# Kimi K3 Architecture and Inference Support Project Plan

- **Date**: 2026-07-21
- **Last upstream audit**: 2026-07-27 06:07 UTC
- **Status**: Proposed; the 2026-07-22 vLLM implementation preview and its already-public prerequisite code are incorporated, while the K3-specific vLLM release branch, checkpoint, and technical report were not yet public at the audit timestamp
- **Scope**: Llaminar V2 model loading, graph construction, Kimi Delta Attention, Attention Residuals, Gated MLA, Stable LatentMoE, SiTU, native vision, MXFP4/MXFP8 execution, prefix caching, MTP, and first-party CPU/CUDA/ROCm kernels with distributed parity
- **Primary upstream**: [MoonshotAI](https://github.com/MoonshotAI)
- **Related Llaminar projects**: [vLLM-Style MTP](../2026-06/MTP_VLLM_STYLE_PROJECT_PLAN.md), [Multi-Domain Pipeline Execution](../2026-06/MULTI_DOMAIN_PIPELINE_EXECUTION_PLAN.md), and [Cross-Backend NativeVNNI Dispatch](NATIVE_VNNI_BATCH_INVARIANT_LEARNED_DISPATCH_POLICY.md)
- **Normative language**: `MUST`, `MUST NOT`, `SHOULD`, `SHOULD NOT`, and `MAY` are requirements levels.

---

## 1. Objective

Add complete, parity-verified Kimi K3 inference support to Llaminar V2 without
weakening the per-device symmetric graph architecture, recurrent-state
correctness, prefix-cache semantics, accepted-state MTP transaction, or
cross-backend requirements.

The project must support the public K3 checkpoint and tokenizer, reproduce the
released reference model's prefill and decode results, and provide practical
execution paths for:

- CPU through Llaminar-owned scalar and vector kernels;
- NVIDIA CUDA through Llaminar-owned CUDA kernels;
- AMD ROCm through Llaminar-owned HIP kernels;
- single-device inference;
- local and global tensor parallelism;
- pipeline parallelism;
- expert-parallel and routed-expert placement;
- prefix-cache save and restore;
- grouped request execution;
- speculative verification and accepted-state publication; and
- the released native-vision input contract.

K3 is a 2.8-trillion-parameter model with 16 of 896 experts active per token.
Single-device semantic support is required, but practical full-model deployment
is expected to be distributed. The first correctness fixtures therefore do not
need to load all 2.8T parameters: synthetic operator fixtures, extracted layer
fixtures, and any smaller architecture-compatible checkpoint released by
Moonshot are valid bring-up vehicles.

## 2. Executive Decision

Llaminar will implement K3 as a new model family under `src/v2/models/kimi3/`
while generalizing existing Qwen 3.5 GDN, hybrid-cache, MTP, and MoE
infrastructure where the semantics are genuinely shared.

KDA will **not** be represented as a renamed Qwen Gated DeltaNet kernel. Both
mechanisms use a delta-rule recurrent state, but their gate contracts differ:

```text
Qwen GDN
  alpha:   [tokens, value_heads]
  dt_bias: [value_heads]

Kimi Delta Attention
  gate:    [tokens, value_heads, key_dim]
  dt_bias: [value_heads, key_dim]
```

The recurrent-state owner, prefix snapshot machinery, grouped-request state
banks, MTP row capture, and TP handoff may be generalized. Projection shapes,
buffer contracts, kernel interfaces, and arithmetic must remain distinct.

The implementation strategy is:

```text
public K3 config + tensor manifest
                 |
                 v
Kimi3GraphConfigBuilder + Kimi3Schema + Kimi3Graph
                 |
       +---------+----------+------------------+
       |                    |                  |
       v                    v                  v
 KDA layers          Gated MLA layers     AttnRes depth bank
       |                    |                  |
       +--------------------+------------------+
                            |
                            v
               SiTU + Stable LatentMoE
                            |
                            v
              routed expert graph collectives
                            |
                            v
          final norm / LM head / native vision bridge
```

All boxes in this execution path are implemented inside Llaminar. External
kernel projects may inform equations, tensor layouts, fusion boundaries,
test vectors, and performance targets, but they are not production
dependencies. In particular, K3 support MUST NOT link to, dynamically load,
vendor, invoke at runtime, or require TRTLLM-Gen, TensorRT-LLM, FlashInfer,
DeepGEMM, FlyDSL, AITER, Triton, FLA, FlashKDA, FlashMLA, or their generated
kernel binaries. The CUDA, ROCm, and CPU implementations are independent
Llaminar source code compiled by the existing Llaminar build.

Reference-source priority is fixed as follows:

1. The released K3 checkpoint, config, model code, tensor manifest, and
   technical report are authoritative.
2. Moonshot's official component repositories are the preferred optimized
   references.
3. FLA's naive KDA implementation is the initial mathematical oracle.
4. SGLang and vLLM are serving-state and integration references.
5. FlashInfer is the optimized recurrent-decode reference.
6. FlashMLA and Megatron-LM are component precedents, not evidence of exact K3
   semantics.

This priority ranks evidence, not dependency or reuse preference. A higher
ranked project can define the expected behavior without becoming part of the
Llaminar build or runtime.

No K3-specific schema, layer schedule, tensor name, quantization block layout,
or equation may be frozen from a precursor model when the public K3 artifact
can answer the question directly.

## 3. Release State and Upstream Audit

### 3.1 Confirmed release facts

Moonshot launched K3 through its hosted products and API on 2026-07-16. The
[Kimi K3 technical blog](https://www.kimi.com/en/blog/kimi-k3) states that the
full weights and further architecture, training, and evaluation details will be
released by 2026-07-27.

The following high-level facts are public:

| Property | Public statement | Planning consequence |
|---|---|---|
| Parameter count | 2.8T total | Full-model testing requires a large distributed topology or extracted fixtures. |
| Context length | 1M tokens | KDA state is essential; full KV allocation cannot be the universal path. |
| Modalities | Native vision plus text | The model boundary must support multimodal preprocessing and vision features. |
| Linear attention | Kimi Delta Attention | Fine-grained per-key-dimension decay and recurrent-state caching are first-class. |
| Depth residual | Attention Residuals | The graph must retain and selectively aggregate block-level representations. |
| Sparse FFN | Stable LatentMoE, 16 of 896 experts | Top-16 routing and much wider expert identity/runtime tables are required. |
| Other blocks | Gated MLA and SiTU | Exact equations remain release-blocked. |
| Quantization | QAT with MXFP4 weights and MXFP8 activations | A correctness fallback and native microscaling roadmap are both required. |
| Serving | vLLM reports KDA-aware partial prefix caching; generic hybrid-cache prerequisites are merged, while the K3 branch is not public | Adopt the public logical-prefix design now and diff the K3-specific state encoding when released. |
| Recommended deployment | Supernodes with 64 or more accelerators | Expert-parallel communication is a primary design dimension, not an afterthought. |

Quantile Balancing and Per-Head Muon are training mechanisms. They are relevant
to interpreting checkpoint metadata and router behavior, but they do not by
themselves require inference kernels. The inference project must not reproduce
optimizer or load-balancing training code unless the released runtime contract
depends on it.

### 3.2 Hugging Face Transformers audit

As of this research snapshot, public branches, pull requests, commits, and model
directories contain no K3 implementation. The public
[`k3` branch search](https://github.com/huggingface/transformers/branches/all?query=k3)
is empty.

The only public Kimi-named feature branch is `exportable-kimi`. Its open
[Transformers PR #47096](https://github.com/huggingface/transformers/pull/47096)
changes Kimi K2.5 multimodal and exporter files. It is not K3 support and is not
a KDA reference.

There is also no public K3 checkpoint in
[Moonshot's Hugging Face model inventory](https://huggingface.co/moonshotai/models)
at the time of this plan.

### 3.3 Moonshot component repositories

No public `Kimi-K3` source repository exists yet. The official component
repositories relevant to K3 are:

| Repository | Contents | Use in this project | Current limitation |
|---|---|---|---|
| [FlashKDA](https://github.com/MoonshotAI/FlashKDA) | MIT CUTLASS/CuTe KDA forward kernels and exact tests | Read-only NVIDIA prefill design and parity reference | SM90+, CUDA 12.9+, BF16, `K=V=128`, no GVA, forward only |
| [Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear) | Architecture paper, released precursor checkpoints, links to FLA | Projection/model-layout precursor and component parity vehicle | Its exact tensor and layer indexing are not authoritative for K3; the vLLM preview independently confirms MLA every four layers |
| [Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals) | Paper and Block-AttnRes pseudocode | Functional AttnRes oracle | No production kernel and no repository license file |

### 3.4 Serving and kernel ecosystem

The public ecosystem already contains substantial KDA work even though no
complete K3 model is public:

| Project | Public implementation | Project use |
|---|---|---|
| [FLA KDA](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda) | Naive recurrent oracle, Triton chunk and recurrent kernels, variable lengths, GVA, forward/backward | Offline mathematical oracle, test-vector generation, and algorithm reference only |
| [FlashInfer KDA](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/kda_kernels/recurrent_kda.py) | SM100 CuTe DSL decode, speculative decode, GQA, indexed state pools | Read-only CUDA decode and state-indexing reference |
| [FlashInfer PR #4001](https://github.com/flashinfer-ai/flashinfer/pull/4001) | Active one-warp/grouped recurrent decode branch | Watch and re-benchmark before freezing Blackwell dispatch |
| [SGLang KDA backend](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kda_backend.py) | Triton, FlashKDA, CuTe DSL, FlashInfer dispatch; MTP; radix cache | Best public end-to-end state/cache integration reference |
| [SGLang PR #31474](https://github.com/sgl-project/sglang/pull/31474) | KDA prefix-cache state tracking and Kimi-Linear enablement | Cache boundary, intermediate-state, and fallback reference |
| [vLLM KDA layer](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py) | Continuous batching, hybrid state, Kimi-Linear integration | Model wiring and state-pool reference |
| [Liger AttnRes PR #1161](https://github.com/linkedin/Liger-Kernel/pull/1161) | Merged fused Triton RMSNorm, depth score, softmax, and weighted-sum kernel | Public optimized AttnRes reference and fixture source |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | Optimized standard MLA kernels, including 512-wide configurations | Read-only CUDA MLA design reference |
| [Megatron-LM MoE layer](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_layer.py) | Latent down/up projections around routed experts | LatentMoE structural precedent only |

SGLang contains the only public K3-specific integration clue found outside
Moonshot's site: its server warm-up recognizes
`KimiK3ForConditionalGeneration` and `model_type == "kimi_k3"`, and uses a
448x448 image described as a 32x32 grid of 14x14 patches. This is not a public
vision-tower implementation and must remain an unconfirmed hint until the K3
processor config and source appear.

### 3.5 vLLM preview and public code audit

The
[vLLM K3 preview](https://vllm.ai/blog/2026-07-22-kimi-k3-preview)
describes a working release candidate shared with approved deployment partners.
It says that the K3-specific release branch includes the language and vision
models, cache integration, serving parsers, and hardware kernels. That branch
was **not public** at 2026-07-27 06:07 UTC.

The public-ref audit found:

- no `kimi`, `k3`, or K3-model branch in the official
  [`vllm-project/vllm` branch list](https://github.com/vllm-project/vllm/branches/all);
- no `KimiK3ForConditionalGeneration` or `kimi_k3` model implementation on
  public vLLM `main`;
- only [PR #49895](https://github.com/vllm-project/vllm/pull/49895), which adds
  `kimi` and `k3` automatic labels in preparation for the release;
- the public branch named
  [`release`](https://github.com/vllm-project/vllm/tree/6d98f91c13e2a0bcc97646bfb98cba7165742381)
  points to a 2026-02-16 Qwen 3.5 fix and is unrelated to K3;
- no K3 implementation branch in the public vLLM forks checked for the blog
  author and public Inferact contributors (`youkaichao`, `jcotant-inferact`,
  `inferact-vllm-bot`, `robin-inferact`, `BugenZhao`, `jeejeelee`, `njhill`,
  and `simon-mo`), and no public `MoonshotAI/vllm` repository; and
- public `kimi-exp`, `kimi-exp-rebase`, and `kimi-opt` branches in an Inferact
  contributor fork contain Kimi K2.5/NVFP4 work, not K3.

The likely conclusion is that “release branch” in the blog refers to a
private or access-controlled integration branch that will be published with
the model artifacts. This conclusion is timestamped, not permanent; Phase 1
must repeat the ref and code search before importing any implementation.

The blog nevertheless fixes several architecture and performance targets:

| Area | vLLM preview evidence | Llaminar planning consequence |
|---|---|---|
| Layer schedule | 93 layers, with MLA every four layers | Plan for a three-KDA/one-MLA cadence, but load the exact layer index list from the released config. |
| KDA prefill | FlashKDA and Triton/FLA; fused input projections and causal convolution; initial-state gather in the surrounding operation | Preserve a standalone recurrence oracle, then promote a fused projection/convolution/state-gather envelope. |
| KDA decode | NVIDIA fusion covers convolution, recurrent update, output gate, and normalization | Make this the CUDA fusion target; keep a portable staged fallback. |
| Prefix cache | Match granularity is independent of physical state-block size; fine-grained chained hashes, exact token hits, partial-block copy-on-write, safe same-step deferral, and transfer support | Replace the earlier checkpoint-only cache plan with the concrete logical-prefix design in Section 14.2. |
| AttnRes | Triton and NVIDIA paths fuse residual update, depth mixing, and output RMSNorm; sequence parallelism shards depth traffic | Add a three-operation fused contract and an explicit sequence-parallel shard/reduction design. |
| Gated MLA | Separate manually fused prefill/decode paths; gate projection overlaps attention in decode and is fused into the gate epilogue in prefill | Design separate phase plans and an explicit stream fork/join for decode. |
| MoE | SiTU is wired into MXFP4 TRTLLM-Gen and DeepGEMM; large token-by-top-k grids are chunked | Implement Llaminar-owned parameterized SiTU kernels and bounded physical-grid chunking; use the external paths only as behavioral and performance references. |
| NVIDIA validation | Optimized MXFP4 path passed correctness on DP16+EP16 | Add DP16+EP16-equivalent route/quantization coverage to the distributed gate. |
| AMD | FlyDSL MLIR provides tuned A16W4/A8W4 fused MoE operators with SiTU | Use its published semantics, shapes, and measurements to guide independent Llaminar HIP kernels; do not import or depend on FlyDSL/AITER. |
| Vision | The initial vLLM path supports image-only preprocessing and a vision tower | Make image input mandatory for initial parity and reject video unless the released processor expands the contract. |

The following relevant code is already public and should be read before
designing Llaminar equivalents:

| Public ref | State | What it exposes |
|---|---|---|
| [Hybrid partial-cache RFC #45702](https://github.com/vllm-project/vllm/issues/45702) | Open design record | Logical hash granularity, partial aliases, exact hit lengths, copy-on-write, scheduler-split versus backend-checkpoint tradeoff, and same-step reuse policy. |
| [PR #45939](https://github.com/vllm-project/vllm/pull/45939) / [merge commit `6bc6f2d`](https://github.com/vllm-project/vllm/commit/6bc6f2d86d7800f878a4a13495b51a6e3b728c37) | Merged | Fine-grained prefix-chain hashes, coarse group views, partial block aliases, promotion, eviction, and event lifecycle. |
| [PR #46384](https://github.com/vllm-project/vllm/pull/46384) / [merge commit `481e481`](https://github.com/vllm-project/vllm/commit/481e481be786c1ca3229e26aa34c15ffd22375af) | Merged | `prefix_match_unit`, hybrid-group convergence, scheduler boundary creation, exact matched-token return, partial-tail materialization, worker copies, copy ordering, and lifetime fences. |
| [PR #43833](https://github.com/vllm-project/vllm/pull/43833) / [head commit `e4734b8`](https://github.com/vllm-project/vllm/commit/e4734b8233c9b33b95bf8383553347e8e14b7041) | Closed, unmerged proof of concept | FlashKDA dispatch, the `lower_bound` gate variant, shape/platform checks, variable-length adapter, and Triton fallback. |
| [PR #45804](https://github.com/vllm-project/vllm/pull/45804) | Open | Per-cache-group local hit accounting for PD-disaggregated hybrid models. |
| [PR #49762](https://github.com/vllm-project/vllm/pull/49762) | Open | Public `kda-mla-nixl-pd` branch for hybrid MLA plus recurrent-state transfer. |
| [Liger PR #1161](https://github.com/linkedin/Liger-Kernel/pull/1161) / [merge commit `3f85b72`](https://github.com/linkedin/Liger-Kernel/commit/3f85b7206b42d09152504e9cb005bd0f24963755) | Merged | A one-pass Triton AttnRes implementation with register-resident depth scores for at most 32 blocks. |

These refs are implementation evidence, not a substitute for the hidden K3
model code. In particular, the public FlashKDA vLLM PR is unmerged and the
public AttnRes kernel does not expose vLLM's K3-specific fused residual-update
and output-normalization wrapper.

### 3.6 Known unknowns

The following facts are release blockers for exact model integration:

- model architecture string and Transformers registration names;
- exact KDA/Gated-MLA layer indices and treatment of the first and last layers,
  despite the reported 93-layer network and MLA-every-four-layers cadence;
- KDA Q/K/value head counts, dimensions, convolution width, safe-gate mode,
  state layout, and state precision;
- AttnRes block count, block boundaries, projection orientation, norm epsilon,
  and whether any implementation detail differs from the public paper;
- Gated MLA equations, projections, cache representation, head dimensions, and
  gating placement;
- Stable LatentMoE latent size, router formula, shared-expert behavior,
  renormalization, expert dimensions, and stability changes;
- SiTU equation, parameterization, and exact insertion point;
- vision-tower family, projector, token layout, patch merge, and image
  placeholder protocol;
- checkpoint tensor names and sharding axes;
- MXFP4/MXFP8 block sizes, scale encoding, scale layout, rounding, packing, and
  accumulator precision;
- tokenizer vocabulary, special tokens, chat template, tool-call format, and
  preserved-thinking-history requirements;
- whether Moonshot releases BF16 weights, microscaled weights, or both; and
- exact KDA state encoding, checkpoint cadence, cache-transfer wire format, and
  release-branch changes beyond public PRs #45939 and #46384.

These questions MUST be answered from public release artifacts and recorded in
an evidence manifest before the corresponding schema is promoted.

## 4. Current Llaminar Baseline

### 4.1 Model and graph infrastructure

Llaminar V2 already provides the correct architectural boundary for a new
model:

- `GraphBuilderRegistry` and `SchemaFactoryRegistry` select a model-specific
  graph and schema without changing core orchestration;
- `DeviceGraphOrchestrator` executes participant-local graphs;
- `RankOrchestrator` coordinates TP/PP participants;
- `GraphResolver` binds tensor shapes, sharding, cache, and device placement;
- `BufferArena`, `StageBoundBuffers`, and `TransferEngine` provide explicit
  activation ownership and coherence; and
- graph collectives represent multi-device synchronization.

K3 MUST follow this model-registration path. It must not introduce an operator
layer, a nested multi-device graph, or model-specific orchestration branches in
core execution.

### 4.2 Hybrid recurrent attention

`Qwen35Graph` already supports heterogeneous GDN and full-attention layers. Its
GDN path includes:

- fused QKV/Z/alpha/beta projections;
- short convolution with persistent per-request state;
- recurrent delta-rule state;
- CPU, CUDA, and ROCm kernels;
- local-TP state sharding and live-state handoff;
- grouped-request state banks;
- effective-length padded prefill;
- MTP row snapshots and accepted-row restoration; and
- hybrid KV/GDN cache ownership.

This state lifecycle is the strongest reusable foundation for KDA. The existing
arithmetic interface is not sufficient because it carries one decay scalar per
head instead of one decay value per key channel.

### 4.3 Prefix caching and speculative state

The prefix-cache framework already stores hybrid state, terminal hidden/logits,
MTP sidecars, and MoE runtime fingerprints. KDA support must extend those typed
payloads rather than create a parallel cache service.

The existing grouped verifier transaction captures and publishes KV, GDN
recurrence, and short-convolution state by accepted row. KDA must join the same
transaction. Rejected verifier rows must never mutate live KDA state.

### 4.4 MoE infrastructure

The Qwen 3.5 MoE path already supports:

- device router logits and top-k selection;
- grouped expert prefill and decode;
- shared experts;
- expert placement and overlay domains;
- hot-expert residency and dynamic rebalancing;
- sparse dispatch and return reduction;
- prefix-cache fingerprints for runtime placement; and
- CPU, CUDA, and ROCm expert kernels.

K3's `top_k=16` exactly matches Llaminar's current maximum route width. Expert
identity capacity does not:

- the general CUDA top-k router allows up to 1,024 experts;
- `MoERuntimeTable` and device overlay metadata allow 256 experts;
- the ROCm device runtime also uses a 256-expert limit; and
- participant metadata currently allows eight participants in the device
  runtime table, far below K3's recommended 64-accelerator deployment.

The K3 distributed path therefore requires a deliberate runtime-table redesign
or domain-segmented representation. Raising a constant without auditing every
fixed array, launch shape, serialization record, fingerprint, collective, and
test is prohibited.

### 4.5 Quantization and vision gaps

The repository contains float8 NCCL datatype declarations but no complete
MXFP4-weight/MXFP8-activation tensor, loader, dequantization, GEMM, or MoE path.
Microscaling is a major workstream.

Llaminar V2 also has no current multimodal Kimi graph. K3 native vision will
require a processor/runtime boundary and graph stages that are not present in
the text-only Qwen graphs.

## 5. Non-Negotiable Invariants

1. Graphs remain per-device and symmetric. A participant with no local expert
   work no-ops until the next collective.
2. K3 support enters through model registries and typed graph configuration;
   core orchestration remains model-agnostic.
3. FLA's naive recurrence is the temporary KDA oracle only until the public K3
   implementation is available. The released K3 implementation then becomes
   authoritative.
4. KDA and GDN share state infrastructure only through explicitly generalized
   contracts. KDA arithmetic must not be forced through scalar GDN gate shapes.
5. Prefill, decode, grouped requests, prefix restore, and MTP verification use
   the same KDA recurrence and state-layout contract.
6. Padded rows and rejected speculative rows never advance short-convolution or
   recurrent state.
7. Prefix hits restore both KDA recurrence and convolution state at the exact
   matched token boundary.
8. AttnRes block representations are explicit graph state. They are not hidden
   inside model-global mutable objects.
9. PP transfers carry every depth representation needed by downstream AttnRes
   blocks, with an explicit layout and lifetime.
10. Expert routing remains a graph collective. Expert-parallel execution must
    not introduce nested remote expert subgraphs.
11. `top_k=16`, 896 expert identities, and the promoted participant count are
    validated at every runtime-table, dispatch, return, and fingerprint boundary.
12. Quantized execution may change storage and kernel implementations, never
    the defined dequantized model semantics beyond an approved parity budget.
13. A BF16/FP32 correctness path remains available until native MXFP promotion
    gates pass.
14. CPU, CUDA, and ROCm implement the same functional model. Backend-specific
    fast paths may be policy-disabled, but semantic gaps must remain visible.
15. GPU work uses the caller's explicit stream and arena-backed workspaces. Hot
    paths have no implicit default-stream dependency or per-token allocation.
16. Weight sharding is declarative and tensor-specific. No backend infers a K3
    sharding axis from a precursor model name.
17. Model, tokenizer, processor, chat template, and quantization revision are
    part of the runtime and prefix-cache fingerprint.
18. Every implementation informed by upstream work records the source commit,
    license, consulted behavior, and independently implemented Llaminar
    component.
19. Performance optimization cannot waive parity, state restoration, route
    correctness, or collective participation requirements.
20. K3 is not promoted on an API-only approximation. Public weights and a
    reproducible reference are mandatory.
21. Every production K3 compute path is implemented and compiled in the
    Llaminar repository. External kernel libraries are reference and benchmark
    inputs only; they are not linked, loaded, vendored, or invoked by the
    shipping runtime.
22. CPU, CUDA, and ROCm each retain a Llaminar-owned correctness path. A
    platform is not declared supported merely because an external project
    implements the operation on that platform.

### 5.1 First-party kernel ownership policy

The following implementation boundary is normative:

| Backend | Required Llaminar implementation | Permitted foundations | Prohibited production dependency |
|---|---|---|---|
| CPU | Portable scalar oracle plus Llaminar-owned threaded and ISA-specialized vector kernels where beneficial | C++ compiler, OpenMP, compiler intrinsics, existing Llaminar tensor/GEMM infrastructure | oneDNN, external BLAS, or another framework as the K3 execution path |
| CUDA | Llaminar-owned `.cu`/`.cuh` kernels, launchers, prepared-weight layouts, workspace contracts, and dispatch policy | CUDA compiler/runtime, documented CUDA intrinsics/PTX, existing Llaminar CUDA utilities | TRTLLM-Gen, TensorRT-LLM, FlashInfer, DeepGEMM, FlashKDA, FlashMLA, Triton, imported cubins, or runtime JIT modules from those projects |
| ROCm | Llaminar-owned `.hip` kernels, launchers, prepared-weight layouts, workspace contracts, and dispatch policy | HIP/ROCm compiler/runtime, documented AMD intrinsics, LLVM ISA inspection, existing Llaminar ROCm utilities | FlyDSL, AITER, Composable Kernel, Triton, imported code objects, or runtime JIT modules from those projects |

Existing Llaminar-owned kernels and shared helpers MAY be generalized and
reused. CUDA and HIP source may share backend-neutral headers for equations,
metadata, and test vectors, but neither GPU backend is implemented as a wrapper
around the other or around an external library. CPU is a production semantic
backend, not merely a fixture generator.

External projects MAY be used outside the shipping process to:

- resolve public equations and packing conventions;
- generate frozen reference fixtures in an isolated tool environment;
- identify useful fusion boundaries, tilings, and dispatch dimensions;
- establish comparative performance targets; and
- cross-check edge cases.

They MUST NOT appear in the production link graph, runtime search path,
container requirements, model-serving Python environment, or graph dispatch.
No source file, generated binary, or JIT artifact from an external kernel
project is copied into the K3 implementation. Any algorithmic technique
learned from public code is re-expressed under Llaminar's kernel interfaces
with its provenance and license recorded.

## 6. Target Model Representation

### 6.1 Proposed directory layout

```text
src/v2/models/kimi3/
  Kimi3Graph.h
  Kimi3Graph.cpp
  Kimi3GraphConfigBuilder.h
  Kimi3GraphConfigBuilder.cpp
  Kimi3Schema.h
  Kimi3BufferSpec.h
  Kimi3BufferSpec.cpp
  Kimi3WeightManifest.h
  Kimi3ArchitectureEvidence.h
```

The exact file split may change, but the responsibilities must remain separate:

- `Kimi3GraphConfigBuilder` parses model metadata and constructs typed model
  configuration and weights;
- `Kimi3Schema` declares stages, buffers, sharding, and tensor aliases;
- `Kimi3Graph` selects the layer subtype and builds participant-local DAGs;
- `Kimi3BufferSpec` owns activation formulas and arena identifiers; and
- `Kimi3ArchitectureEvidence` records the upstream revision and the evidence
  source for every non-obvious mapping.

### 6.2 Typed configuration

The K3 config should contain explicit substructures instead of a flat set of
custom formulas:

```text
Kimi3Config
  model dimensions and vocabulary
  layer_types[]
  KDAConfig
  GatedMLAConfig
  AttnResConfig
  StableLatentMoEConfig
  SiTUConfig
  VisionConfig
  MicroscalingConfig
  tokenizer / processor identities
```

`layer_types[]` MUST come from the checkpoint configuration or a digest-bound
manifest generated from it. It must not be reconstructed from a presumed ratio
such as Kimi-Linear's earlier 3:1 KDA-to-MLA pattern.

### 6.3 Weight manifest

Before loader implementation begins, generate a machine-readable manifest with:

- source tensor name;
- logical semantic name;
- shape and dtype;
- quantization block and scale metadata;
- owning layer/block;
- TP/EP sharding axis;
- replicated versus participant-local status;
- expected GGUF name, if GGUF conversion is supported;
- optional fusion grouping; and
- reference-source line or configuration field.

The manifest is a release artifact and a parity-test input. Silent wildcard
loading of unknown K3 tensors is prohibited.

## 7. Workstream A: Kimi Delta Attention

### 7.1 Functional recurrence

The initial KDA oracle will use the public FLA recurrence. For value head `h`
at token `t`, with state `S[h, K, V]`:

```text
q, k        = L2Normalize(raw_q), L2Normalize(raw_k)
decay       = exp(gate[h, K])
prediction  = k @ S
S           = decay[:, None] * S
              + beta[h] * outer(k, v - prediction)
output      = scale * q @ S
```

The released K3 source may define safe-gate activation, beta range, state
orientation, scaling, or operation ordering more precisely. Those details must
replace this provisional description without changing the public stage
ownership model.

### 7.2 New kernel interface

Introduce `ITensorKimiDeltaAttention` or a more general typed recurrent-attention
interface capable of expressing:

- Q/K head count distinct from value-head count;
- `d_k` distinct from `d_v`;
- raw per-key gate `[tokens, value_heads, d_k]`;
- beta logits `[tokens, value_heads]`;
- `A_log[value_heads or qk_heads]` as confirmed upstream;
- `dt_bias[value_heads, d_k]`;
- optional Q/K normalization;
- optional fused safe-gate activation;
- state `[request, value_heads, d_k, d_v]` or the released transposed layout;
- variable-length packed requests;
- indexed live-state pools;
- final-state and intermediate-state output;
- effective-length padded prefill;
- MTP verifier snapshots; and
- explicit stream/workspace binding.

The existing `ITensorGatedDeltaNet` remains valid for Qwen GDN unless a common
base can be introduced without weakening either type's compile-time and runtime
shape validation.

### 7.3 Projection and convolution graph

The released model will determine fusion boundaries. The graph must be able to
represent at least:

```text
input norm
  -> Q/K/V/Z/gate/beta projections
  -> short convolution over released channels
  -> split or view Q/K/V/gate/beta
  -> KDA recurrence
  -> released gated norm / output gate
  -> output projection
  -> AttnRes-aware block update
```

Projection fusion is a performance choice. The initial parity graph may use
separate projections if that makes tensor mapping and snapshots unambiguous.
Fusion must preserve reference order and released bias semantics.

The vLLM preview establishes the promoted prefill fusion envelope:

```text
input projections
  -> causal convolution
  -> initial recurrent-state gather
  -> KDA recurrence backend
```

Llaminar SHOULD preserve raw gate and beta values until the selected backend
has applied the released gate transforms. That lets a fused backend consume raw
projections while the reference path materializes each transform explicitly.
The unfused path remains required for stage snapshots and cross-backend parity.

### 7.4 Prefill implementation ladder

1. CPU scalar/vector reference using the recurrent equation.
2. Llaminar CUDA and HIP sequential recurrent references for cross-backend
   fixtures.
3. Llaminar-owned chunk algorithm informed by the published FLA recurrence,
   with chunk sizes supported by the released model.
4. Llaminar-owned SM90 CUDA fast path covering the verified FlashKDA reference
   domain when shape and gate restrictions match.
5. Llaminar-owned SM100 CUDA path using architecture-appropriate intrinsics
   where it is measurably superior.
6. Backend policy selecting sequential versus chunked execution by token count,
   request packing, head geometry, state precision, and accelerator capability.

FlashKDA's fixed 16-token internal chunk and FLA's 32/64-token chunks are
implementation facts, not K3 model semantics. The dispatch contract must allow
multiple kernels without exposing their tile sizes in the model schema.

The closed vLLM FlashKDA proof of concept provides a concrete candidate-domain
check that Llaminar should reproduce for its own CUDA kernel before
dispatching:

- the applicable Llaminar CUDA candidate is compiled and registered;
- CUDA device capability at least SM90;
- `d_k == d_v == 128` for the published kernel;
- released `lower_bound` gate configuration present;
- supported BF16 input/state types; and
- supported variable-length sequence-index width.

That proof of concept distinguishes:

```text
legacy Kimi-Linear gate
  g = -exp(A_log) * softplus(a + dt_bias)

lower-bound gate family exercised by the FlashKDA reference
  g = lower_bound * sigmoid(exp(A_log) * (a + dt_bias))
```

The vLLM blog's statement that K3 uses FlashKDA is strong indirect evidence
that the released K3 configuration uses the second family, but Llaminar MUST
confirm the exact formula, signs, broadcasting, clamp behavior, and
`lower_bound` value from K3 artifacts. Unsupported cases must select the
Llaminar chunked or sequential fallback rather than silently reinterpret the
gate.

### 7.5 Decode implementation ladder

1. Exact CPU recurrent step.
2. Llaminar CUDA/HIP head-parallel recurrence for correctness.
3. Packed multi-request device-state indexing.
4. Llaminar-owned architecture-specialized CUDA decode informed by the public
   FlashInfer scheduling and state-indexing evidence.
5. Fused multi-token verifier path with independent post-row state snapshots.
6. Shape-trained dispatch across batch, heads, `d_k`, `d_v`, verifier depth,
   state precision, and GPU generation.

For supported NVIDIA shapes, the promoted single-token fusion target is:

```text
short convolution update
  -> KDA recurrent update
  -> output gate
  -> output normalization
```

It SHOULD keep recurrent and convolution state device-resident, avoid
intermediate global-memory round trips, and optionally emit post-row snapshots
for grouped MTP verification. Projection GEMMs remain separate unless
measurement and the released reference justify a larger fusion boundary.

The decode kernel must support low-batch occupancy. One CTA per request/head can
underfill large GPUs; V-tiled or grouped-warp variants should be candidates, not
hard-coded universal choices.

### 7.6 State ownership and prefix caching

Each request requires, per KDA layer:

- short-convolution window state;
- recurrent matrix state;
- live state-pool slot identity;
- optional intermediate verifier rows; and
- prefix-cache checkpoint metadata.

The state cache must support:

```text
fresh request
  -> zero or checkpoint initial state
  -> prefill updates
  -> terminal live state
  -> zero or more decode updates

prefix hit
  -> restore exact matched conv + recurrence state
  -> prefill unmatched suffix
  -> continue decode

MTP verify
  -> read committed live state
  -> write post-row snapshots to speculative workspace
  -> publish accepted row only
```

SGLang's full-precision active pool plus compressed idle checkpoint pool is a
useful future optimization. Llaminar's first K3 prefix-cache implementation
must store a lossless state representation. INT8 or other compressed prefix
state requires a separate measured quality and continuation-parity decision.

## 8. Workstream B: Attention Residuals

### 8.1 Functional contract

The public Block-AttnRes pseudocode stacks completed block representations and
the current partial block, RMS-normalizes them for key scoring, computes a
learned pseudo-query score over depth, applies softmax over depth, and returns a
weighted sum of the unnormalized representations.

For `N` completed blocks:

```text
V       = stack(completed_blocks + current_partial)  [N+1, tokens, d_model]
K       = RMSNorm(V)
score   = dot(depth_projection, K)                    [N+1, tokens]
weight  = softmax(score, depth_axis)
output  = sum(weight[..., None] * V, depth_axis)
```

AttnRes is applied before attention and before the MLP in the public design.
Exact K3 block boundaries and parameter sharing remain release-blocked.

### 8.2 Graph and buffer design

Add typed buffers for:

- completed block bank;
- current partial block;
- normalized depth keys;
- depth logits and weights; and
- aggregated hidden state.

The initial implementation may use separate RMSNorm, depth-dot, softmax, and
weighted-reduction stages. The promoted CUDA path should expose a fused
contract covering:

```text
residual/block update
  -> AttnRes depth mixing
  -> output RMSNorm
```

The merged Liger kernel is the first public optimized reference for the inner
RMSNorm, depth score, softmax, and weighted sum. Its register-resident score
strategy supports at most 32 depth blocks; Llaminar must derive its limit from
the released block count and provide a tiled/two-pass fallback rather than
assuming that bound. The vLLM release kernel must be compared when it becomes
public because its outer residual-update and output-normalization fusion is
broader than Liger's current public interface.

AttnRes state is depth-local, not autoregressive sequence state. During decode,
it lives only while the current token traverses model depth. During prefill, it
holds block representations for every active token row. It is not part of the
KDA prefix payload.

### 8.3 PP implications

If a PP boundary occurs after one or more completed AttnRes blocks, downstream
stages may need more than the ordinary current hidden tensor. The PP contract
must carry:

- completed block count;
- block-bank tensor or a layout-equivalent packed payload;
- current partial block if the boundary splits an AttnRes block; and
- block-configuration fingerprint.

Layer placement should prefer AttnRes block boundaries when feasible, but
correctness must not depend on that placement.

For sequence parallelism, depth-bank traffic SHOULD be sharded over the
sequence dimension when each rank can compute independent token rows. The
design must specify whether any depth-score reduction crosses ranks, the
collective order relative to residual updates, and how padded rows are masked.

## 9. Workstream C: Gated MLA

The exact Gated MLA equations remain unpublished, so no implementation may
claim K3 parity before the report and source are available. The vLLM preview
does confirm a 93-layer network with MLA every fourth layer and provides the
production execution shape:

```text
prefill
  main attention path
  gate projection -> fused sigmoid and elementwise multiply epilogue

decode
  main attention path ----------------------+
  gate projection on an optional stream ----+-> dependency join -> output
```

The implementation tasks are:

1. Ensure K3 layer dispatch can select a distinct global-attention subtype.
2. Generalize attention buffer formulas for large head dimensions and latent
   cache representations.
3. Audit FlashMLA's 512-wide dense/sparse prefill and decode kernels as
   read-only design and benchmark evidence.
4. Keep gating as an explicit stage parameter rather than folding it into
   standard MLA assumptions.
5. Preserve an unfused mathematical path for parity snapshots.
6. Create separate prefill and decode execution plans rather than forcing one
   launch order on both phases.
7. Model decode gate overlap as an explicit stream fork, dependency event, and
   join owned by the stage; never depend on implicit default-stream ordering.
8. Add a prefill gate-projection epilogue candidate that fuses the released
   sigmoid and elementwise multiply only after the reference operation order is
   known.
9. Keep native PD-disaggregation launch policy separate from semantic model
   configuration so local execution can choose a different schedule.

FlashMLA is a performance reference only. Its standard MLA cache and equations
must not be used as evidence for Gated MLA semantics.

## 10. Workstream D: Stable LatentMoE and SiTU

### 10.1 Provisional LatentMoE structure

NVIDIA's public LatentMoE implementation uses:

```text
full hidden state
  +-> full-dimension router
  +-> shared expert in full hidden dimension, if configured
  |
  v
hidden-to-latent projection
  -> expert dispatch
  -> expert gate/up/activation/down in latent representation
  -> weighted combine in latent representation
  -> latent-to-hidden projection
  -> add shared-expert result
```

This is a useful provisional graph shape. It is not proof that K3 uses the same
ordering, projection sharing, or shared-expert behavior.

### 10.2 Expert capacity redesign

The runtime redesign must support at least:

- 896 logical experts;
- 16 route slots per token;
- the released shared-expert count;
- at least 64 participants for the recommended topology;
- expert ownership and replicas;
- device-resident route histograms;
- sparse dispatch/return descriptors;
- hot-expert cache and rebalance state;
- prefix fingerprints; and
- serialization without fixed 256-entry inline arrays.

Preferred design options, in order of evaluation, are:

1. variable-size arena-backed runtime tables with small fixed headers;
2. domain-segmented expert tables with global expert IDs and local segments;
3. a paged expert directory; or
4. a larger fixed maximum only if memory, launch, ABI, and participant scaling
   remain acceptable.

The design must be sized from measured metadata and graph-capture behavior. A
2.8T checkpoint is not a reason to make every small-MoE request carry 896-entry
inline tables.

### 10.3 Routing and static shapes

Moonshot describes balanced expert-parallel execution with static shapes and no
host synchronization on the critical path. Llaminar should target:

- fully device-resident top-16 selection;
- device-resident per-destination counts and offsets;
- bounded static dispatch capacity per EP wave;
- explicit overflow handling;
- all-to-all or equivalent sparse graph collective;
- device-resident return reduction; and
- no decode-time D2H expert-count decision.

The existing host-assisted fallbacks remain useful for debugging but cannot be
the promoted 64-accelerator path.

### 10.4 SiTU

SiTU must be introduced as a parameterized activation contract and kernel only
after the public formula is known. The vLLM preview confirms that the release
configuration passes SiTU parameters into optimized MXFP4 expert backends; it
does not publish the formula. Required work includes:

- scalar oracle;
- CPU vector path;
- CUDA and ROCm fused activation kernels;
- fusion into grouped expert gate/up kernels where parity permits;
- MXFP8 input/output behavior; and
- comparison with unfused activation at boundary values.

The NVIDIA reference targets are the observable semantics and performance
envelopes of TRTLLM-Gen and DeepGEMM MXFP4 expert paths. The AMD reference
target is the observable parameterization, supported-shape surface, and
performance envelope of FlyDSL/AITER A16W4/A8W4 fused experts. Llaminar will
not add adapters to these projects. It will implement one SiTU model contract
through independent CPU, CUDA, and HIP kernels, including unfused correctness
paths and backend-native fused candidates.

The physical launch plan MUST chunk large `tokens * top_k` workloads before
grid dimensions or kernel index widths overflow. Chunking must preserve global
token ID, route slot, expert ID, route weight, and accumulation order across
the logical workload.

## 11. Workstream E: MXFP4 Weights and MXFP8 Activations

### 11.1 Bring-up strategy

Quantization support will be staged:

1. Parse and validate checkpoint microscaling metadata.
2. Add storage tensors and host dequantization or conversion tools.
3. Run the K3 graph through BF16/FP32 activations and dequantized weights for
   functional parity.
4. Add device dequantization and BF16 compute.
5. Add native MXFP4 × MXFP8 GEMM/GEMV candidates where hardware supports them.
6. Add grouped expert kernels and router/latent projections.
7. Train and certify dispatch policy by backend, shape, and phase.

The correctness path may be too slow for full-model service. It exists to
separate model-architecture failures from packed-kernel failures.

### 11.2 Required tensor metadata

Every microscaled tensor needs:

- element format;
- block size;
- packing order;
- scale element format;
- scale block shape and stride;
- zero handling;
- NaN/Inf policy;
- dequantization equation;
- logical versus physical shape;
- transpose state;
- sharding interaction; and
- padding requirements.

These fields belong in tensor metadata and prepared matrix descriptors, not in
K3-only kernel branches.

### 11.3 Backend expectations

- CUDA Blackwell should implement and evaluate Llaminar-native MXFP4
  grouped-expert candidates with parameterized SiTU. TRTLLM-Gen and DeepGEMM
  inform layouts, fusion boundaries, and comparison targets only.
- CUDA Hopper/Ampere should retain Llaminar-native unpack/dequantize paths and
  may add hardware-appropriate packed kernels where the ISA supports them.
- ROCm should implement Llaminar-native HIP A16W4 and A8W4 fused expert
  candidates, using published FlyDSL/AITER behavior only as reference evidence;
  a Llaminar-owned BF16/FP16 expansion path remains the portability fallback.
- CPU must implement Llaminar-owned unpack/dequantize, dense GEMM/GEMV, grouped
  expert, SiTU, and combine paths. It may begin as a semantic path, but its
  supported formats cannot depend on a GPU or external library.

Native-format promotion requires byte-accurate pack/unpack tests, layer parity,
model parity, and a measured throughput or memory benefit. The distributed test
matrix MUST include a DP16+EP16-equivalent case because that is the topology on
which vLLM reports K3's optimized MXFP4 backend passed correctness. Llaminar
does not need to reproduce vLLM's exact backend choice, but it must validate
that all ranks choose a compatible format and activation contract.

## 12. Workstream F: Native Vision

### 12.1 Processor boundary

The K3 processor implementation must own:

- image input normalization;
- resize/crop/tiling policy;
- patch grid and merge policy;
- placeholder and special-token expansion;
- multimodal position IDs;
- batching constraints;
- chat-template rendering; and
- deterministic metadata passed to the graph.

The HTTP/application layer should produce a typed multimodal request. It must
not embed model-specific image token expansion throughout the generic server.
The initial public vLLM target is image-only. Llaminar MUST reject video inputs
with an actionable error unless the released K3 processor explicitly supports
them.

### 12.2 Vision graph

Once public, the vision architecture should be represented as participant-local
stages and a defined bridge into the language graph. Required decisions include:

- whether vision weights are replicated, TP-sharded, or assigned to a separate
  named domain;
- whether the vision graph runs once per media item or once per prompt;
- feature packing for one or multiple images;
- projector precision and quantization;
- feature-cache ownership; and
- PP placement relative to language embeddings.

Kimi K2.5/MoonViT code may inform preprocessing and warm-up behavior. K3's
public processor and checkpoint remain authoritative.

## 13. Workstream G: Parallelism and Placement

### 13.1 Tensor parallelism

The schema must declare sharding for:

- KDA Q/K/V/gate/beta projections;
- KDA recurrent state and short-convolution state;
- Gated MLA projections and latent cache;
- AttnRes projections and block bank;
- LatentMoE down/up projections;
- router weights;
- routed experts;
- shared experts;
- vision tower/projector; and
- LM head/vocabulary.

KDA value heads and their state should normally share ownership. Any Q/K head
replication or modular mapping must follow the checkpoint and be tested at
non-divisible TP boundaries.

### 13.2 Expert parallelism

Expert parallelism must use named execution domains and explicit graph
collectives. The first large-scale target is one static expert placement. Hot
replication and dynamic rebalancing are later promotion layers because they
multiply the 896-expert metadata and prefix-fingerprint surface.

### 13.3 Pipeline parallelism

PP layer splits must understand:

- AttnRes block-bank transfer;
- hybrid KDA/MLA layer state ownership;
- vision-to-language handoff;
- expert-domain collectives within a stage; and
- final hidden/logit ownership.

Pipeline placement should prefer whole AttnRes blocks and avoid moving KDA live
state between stages after initialization. These are policy preferences, not
semantic restrictions.

### 13.4 Heterogeneous execution

CPU/CUDA/ROCm pipeline combinations remain valid where the normal orchestration
model supports them. The K3 graph must not assume one accelerator family. A
heterogeneous deployment may be far from speed-positive; it is still useful for
component validation and spill/offload experiments.

## 14. Workstream H: Prefix Cache, MTP, and Request Batching

### 14.1 Prefix payload

The K3 prefix payload is expected to include:

- MLA KV or latent cache segments for global-attention layers;
- KDA short-convolution state;
- KDA recurrent state;
- terminal hidden and logits where required;
- tokenizer/processor/template fingerprint;
- K3 architecture and quantization fingerprint;
- expert placement/runtime fingerprint; and
- MTP sidecar state if the released model includes or supports MTP.

AttnRes depth-bank state is not included because it is recomputed for each token
as it traverses model depth.

### 14.2 Cache granularity

KDA recurrent state summarizes every prior token. A cache entry at token `P`
must therefore contain the state exactly after token `P-1`. Partial block hits
must either restore a checkpoint at the matched boundary or replay the suffix
from an earlier valid checkpoint. An ordinary KV block match without matching
KDA state is not a valid K3 prefix hit.

The public vLLM implementation resolves the central granularity problem by
separating:

```text
physical block size
  allocation, ownership, and amortized KDA-state storage

scheduler alignment
  legal execution stops at which all hybrid cache groups represent one prefix

prefix match unit
  finer token interval at which a complete-prefix hash can be registered/hit
```

Llaminar SHOULD adopt these semantics, expressed through its own cache and
graph abstractions:

1. Compute one chained prefix hash at each match-unit boundary. The hash at
   token `P` fingerprints the complete prefix `[0, P)`, not only the last token
   chunk.
2. Require the match unit to divide every physical block size in the hybrid
   cache layout. Derive a safe default from the GCD of the resolved KDA and MLA
   group sizes when no explicit value is supplied.
3. Let each cache group expose a coarse physical-block view over the same
   fine-grained hash chain. Do not recompute an unrelated hash for the larger
   block.
4. Return the exact logical matched-token count separately from the list of
   physical blocks. A physical block containing a partial tail does not imply
   that all its token slots are valid.
5. Make the hybrid coordinator converge KDA and MLA groups on one
   `num_computed_tokens`. A hit is publishable only at the longest boundary
   every group can represent exactly.
6. Register at most the useful final partial-tail alias per request/block
   rather than retaining a recurrent snapshot at every fine boundary.
7. Track reverse block-to-alias metadata so promotion, eviction, reset, and
   reuse remove every hash pointing at the physical allocation.
8. Include the exact partial token range in cache store/remove events and
   external-transfer metadata.

#### Partial-tail checkpoint materialization

The first implementation SHOULD use the simpler scheduler-split strategy from
merged vLLM PR #46384:

```text
scheduled prefill
  -> stop at the final reusable match-unit boundary
  -> publish exact KDA recurrence + convolution state
  -> resume the remaining prompt tail
```

This creates one additional model boundary only when the final reusable prompt
tail crosses a fine match boundary inside a physical KDA block. A later
optimization MAY keep one scheduler step and ask the KDA backend to emit an
intermediate checkpoint offset, but it must prove identical batching,
state-lifetime, graph-capture, and continuation behavior.

#### Partial-block copy-on-write

A request extending a shared partial block MUST NOT update that allocation in
place. Two cases are required:

```text
new request hits partial prefix
  shared cache source -> private request destination
  request continues on private destination

running request publishes partial tail and immediately continues
  request keeps original writable block
  cache aliases move to a cache-owned destination
  original -> cache-owned destination copy preserves the published prefix
```

The destination cannot participate in same-step lookup until its copy is
complete. The initial policy SHOULD defer same-step reuse and recompute the
small suffix rather than introduce a dependency that breaks batching.

The device execution order is normative:

```text
zero newly allocated blocks
  -> copy partial KDA/MLA block payloads
  -> execute model forward
  -> release copy lifetime fence after completion
```

Both copy endpoints remain retained until the non-empty execution step
carrying the copy completes. Aliased layer views of one backing allocation are
deduplicated so the physical page is copied once.

#### Disaggregated prefill/decode

Cache-transfer metadata must carry the same logical matched-token count and
per-group coverage used locally. Dense MLA hits must not be collapsed merely
because a recurrent group needs transfer or reconstruction. Conversely, a
dense hit cannot advance `num_computed_tokens` beyond the transferred KDA
recurrence and convolution state. The transfer fingerprint includes match
unit, physical layouts, state dtype/orientation, model revision, and
quantization revision.

The public K3 branch must still be audited for its exact KDA state encoding,
checkpoint cadence, convolution-state packing, copy kernel, transfer wire
format, and any changes beyond the merged generic cache work.

### 14.3 MTP and speculative verification

If K3 exposes MTP weights or is used as a target for an external proposer, KDA
must support the existing all-position verifier contract:

- run verifier rows from a speculative copy of the committed state;
- capture complete KDA and convolution state after each verifier row;
- preserve request isolation for grouped verification;
- publish the selected accepted row entirely on device; and
- leave committed state unchanged on failure or zero accepted draft rows.

Numerical equivalence between grouped verifier rows and serial decode is a
promotion gate. Chunk-prefill approximations are not acceptable for state that
can become live decode state.

### 14.4 Request batching

Production batching must use indexed state pools and packed real lengths. A
host loop that invokes one recurrent kernel per request is a debugging fallback.
The graph-captured path should have stable bucket shapes and device-resident
effective-length metadata.

## 15. Implementation Phases

### Phase 0: Release-blocked foundations

Can proceed while the public checkpoint or K3-specific vLLM branch is absent:

- freeze this project plan;
- add an upstream evidence collector for repository refs, config fields, and
  tensor manifests;
- build standalone FLA-naive KDA vector generation;
- define KDA tensor/layout contracts independent of guessed K3 dimensions;
- prototype a CPU recurrent KDA oracle;
- design generalized linear-attention live-state and snapshot ownership;
- specify chained match-unit hashes, exact logical hit lengths, partial aliases,
  scheduler-split checkpoints, copy-on-write, and copy lifetime fences using
  public vLLM PRs #45939 and #46384 as references;
- audit every 256-expert and eight-participant assumption;
- design the 896-expert runtime-table replacement;
- define AttnRes buffer lifetimes and PP payloads;
- add synthetic KDA and AttnRes unit fixtures; and
- prepare a release-intake checklist.

Phase 0 MUST NOT add a fake `kimi_k3` model registration or inferred tensor
mapping.

### Phase 1: Release intake and architecture freeze

Triggered by public weights, report, model source, or K3-specific vLLM release
branch:

1. Pin exact upstream revisions and artifact digests.
2. Archive `config.json`, generation config, tokenizer, processor, chat
   template, model code, and tensor index.
3. Generate the complete tensor manifest.
4. Resolve every known unknown in Section 3.6 or record it as an explicit
   unsupported release feature.
5. Produce a readable architecture map with layer schedule and state shapes.
6. Generate reference activations for embedding, every new primitive, one full
   layer of each type, and final logits.
7. Freeze the initial Llaminar K3 semantic specification.

Exit gate: two reviewers can trace every model field, tensor, and equation to a
pinned public source or extracted checkpoint evidence.

### Phase 2: Primitive semantic implementation

- KDA CPU oracle and Llaminar-owned CUDA/HIP sequential references;
- KDA projections and short convolution;
- AttnRes functional stages;
- Gated MLA unfused functional path;
- SiTU scalar and backend kernels;
- Stable LatentMoE latent projections and router/combine semantics;
- MXFP unpack/dequantize reference;
- processor and vision primitive reference; and
- typed state/cache payloads.

Exit gate: every primitive passes isolated public-reference tests on CPU, CUDA,
and ROCm, or has a documented hardware/compiler blocker that prevents that
backend from being promoted. Passing through an external kernel implementation
does not satisfy this gate.

### Phase 3: Single-device text graph

- implement Kimi3 config builder, schema, buffer spec, and graph;
- load text-model tensors;
- run embedding through LM head;
- support prefill and serial decode;
- add KDA/MLA hybrid layer dispatch;
- add AttnRes and Stable LatentMoE;
- support lossless prefix save/restore; and
- capture layer/stage snapshots.

Initial weights may be expanded to BF16 for this phase.

Exit gate: greedy token parity and bounded per-stage error on a public text-only
prompt subset, with prefix-hit continuation matching a fresh run.

### Phase 4: First-party CPU, CUDA, and ROCm production kernels

- Llaminar-native chunked KDA prefill derived from the published recurrence;
- fused KDA prefill projection/convolution/state-gather candidate;
- packed recurrent decode;
- Llaminar CUDA prefill candidate covering the verified FlashKDA shape domain;
- Llaminar Blackwell decode candidate informed by public FlashInfer behavior and fused
  convolution/recurrence/gate/norm candidate;
- Llaminar HIP KDA prefill/decode candidates;
- Llaminar CPU vectorized KDA prefill/decode candidates, with portable scalar
  fallbacks;
- fused AttnRes residual-update/depth-mixing/output-RMSNorm candidate;
- separate Gated MLA prefill/decode kernels and explicit decode gate-stream
  overlap;
- first-party CPU/CUDA/ROCm SiTU kernels and eligible fusions;
- first-party CPU/CUDA/ROCm grouped LatentMoE kernels with bounded
  token-by-top-k grid chunking;
- per-backend prepared-weight layouts and dispatch policies owned by Llaminar;
- production-link and runtime-dependency audits proving that no external kernel
  project is required; and
- graph capture for prefill buckets and decode.

Exit gate: CPU/CUDA/ROCm parity, no hot-path allocation/sync violations, a
speed-positive Llaminar kernel selected for each promoted backend surface, and
a clean dependency audit. External kernels may be benchmark comparators but
cannot be selected by production dispatch.

### Phase 5: Distributed execution

- TP weight and state sharding;
- KDA live-state handoff and replication policies;
- 896-expert/64-participant runtime metadata;
- static EP dispatch and return collectives;
- shared-expert and latent projection placement;
- sequence-parallel AttnRes depth-bank traffic;
- AttnRes PP payload;
- pipeline stage placement; and
- distributed prefix-cache coordination.

Exit gate: single-device versus TP parity on extracted fixtures; EP ownership
and return-reduction parity; PP parity across an AttnRes boundary; collective
deadlock and empty-participant tests; and DP16+EP16-equivalent quantized route
and backend-selection coverage.

### Phase 6: Native microscaling and vision

- MXFP4 loader and prepared matrix descriptors;
- MXFP8 activation quantization/dequantization;
- Llaminar-owned CPU, CUDA, and ROCm native-format dense and grouped expert
  kernels where the relevant hardware ISA supports the format;
- Llaminar-owned expand/dequantize fallbacks on every backend;
- native vision processor and graph;
- multimodal prompt/token integration;
- multi-image cases and video rejection unless the released processor adds
  video support; and
- multimodal prefix/request fingerprinting.

Exit gate: released-format model load without offline expansion, multimodal
reference parity, and native-format kernels meeting accuracy and performance
gates.

### Phase 7: MTP, tuning, and promotion

- KDA verifier snapshots and accepted-row publication;
- grouped request batching;
- prefix cache plus MTP interaction;
- per-backend kernel profiling;
- learned or generated dispatch where candidate count justifies it;
- end-to-end benchmark matrix; and
- documentation, examples, and release notes.

Exit gate: the complete Definition of Done in Section 20.

## 16. Test and Evidence Plan

### 16.1 Reference fixture hierarchy

1. FLA naive KDA vectors for provisional primitive tests.
2. Moonshot FlashKDA exact tests for compatible CUDA shapes.
3. Public K3 framework outputs from the pinned release revision.
4. Extracted K3 layer fixtures small enough for CPU/CUDA/ROCm CI.
5. Full-model prompts on qualified distributed hardware.

Every fixture records source revision, input seed, dtype, device, operation
mode, tensor layout, tolerance, and whether recurrent state is initial, final,
or intermediate.

Reference-framework fixture generation runs as an offline, reproducible tool
step. Generated fixture tensors and provenance may be committed; the framework
or kernel library that generated them is not a test or runtime dependency for
normal Llaminar builds.

### 16.2 KDA matrix

Cover at minimum:

- prefill and decode;
- zero state and nonzero initial state;
- fixed and variable lengths;
- one and multiple requests;
- padded buckets;
- Q/K heads equal to and different from value heads;
- released `d_k` and `d_v`, plus nearby unit shapes;
- all released gate modes;
- `lower_bound` gate present/absent and exact broadcast shapes;
- cases inside and outside the public FlashKDA reference domain, proving the
  Llaminar candidate falls back rather than reinterprets unsupported inputs;
- FP32 and released lower-precision state;
- prefix restore at aligned and unaligned points;
- verifier depths used by MTP;
- accepted row zero, middle, and terminal;
- TP-local and replicated state; and
- CPU/CUDA/ROCm.

Tests compare both output and complete final state. Output-only KDA tests are
insufficient.

### 16.3 AttnRes matrix

- first block, interior block, and final block;
- attention-side and MLP-side application;
- prefill and decode;
- one and many completed blocks;
- released depth-block count and a case above any register-resident fast-path
  limit;
- uniform and sharply peaked depth logits;
- residual-update/mix/output-RMSNorm fused versus staged comparison;
- sequence-parallel versus unsharded comparison;
- PP split at and inside a block boundary;
- padded request rows; and
- fused versus unfused byte/tolerance comparison.

### 16.4 Gated MLA matrix

- released MLA layer indices across the 93-layer schedule;
- prefill and decode launch plans;
- sequential gate projection versus explicit multi-stream overlap;
- dependency completion before elementwise gating;
- fused sigmoid/multiply epilogue versus staged reference;
- local and native PD-disaggregated schedules;
- zero-length/padded rows and mixed prefill/decode batches;
- latent-cache save, restore, and transfer; and
- CPU/CUDA/ROCm functional parity with backend-specific overlap disabled where
  unavailable.

### 16.5 MoE matrix

- 896 experts and top-16 selection;
- ties and near-ties in router logits;
- renormalization and released score transformation;
- tokens selecting experts across every participant;
- participants with zero selected tokens;
- shared-expert and latent-path interaction;
- dispatch capacity boundary and overflow;
- token-by-top-k physical grid chunk boundaries;
- static placement and later hot replicas;
- prefix fingerprint changes on placement changes;
- prefill, decode, and verifier rows; and
- CPU/CUDA/ROCm plus EP collectives.

### 16.6 Quantization matrix

- every released MXFP4 code and scale edge;
- positive/negative zero, finite extrema, subnormal policy, NaN/Inf policy;
- partial physical blocks and logical tails;
- transpose and shard boundaries crossing scale blocks;
- host versus CUDA versus ROCm unpack;
- dense GEMV/GEMM and grouped expert GEMM;
- SiTU parameter mapping through every promoted backend;
- DP16+EP16-equivalent all-rank backend agreement;
- fused versus dequantized reference; and
- repeated graph replay.

### 16.7 Vision matrix

- minimum, representative, and maximum supported image sizes;
- non-square images;
- multiple images;
- explicit video rejection unless supported by the released processor;
- processor token/feature count agreement;
- image-free text requests through the same model;
- mixed request batches;
- vision feature caching if implemented; and
- CPU preprocessing versus reference processor outputs.

### 16.8 Prefix-cache matrix

- match unit equal to and smaller than each physical cache-group block;
- derived GCD match unit and rejected non-divisor configurations;
- chained boundary hashes and coarse group views;
- exact matched-token count for full and partial blocks;
- KDA/MLA group convergence on one logical prefix;
- aligned and unaligned prompt tails;
- partial-to-partial and partial-to-full promotion;
- new-request copy-on-write and running-request cache-owned copies;
- zero, copy, forward, and deferred-free ordering;
- same-step cache publication and deliberately deferred reuse;
- eviction/reset with no stale aliases;
- local, offloaded, and disaggregated prefill/decode transfer;
- prefix hit followed by grouped decode and MTP verification; and
- continuation parity against a fresh full prefill.

### 16.9 End-to-end parity

The parity suite should include:

- prompt token IDs and rendered chat template;
- embedding output;
- pre/post AttnRes state at selected blocks;
- KDA projections, convolution output, recurrence output, and final state;
- Gated MLA Q/K/V or latent cache and output;
- router logits, selected experts, route weights, latent representation, and
  combined output;
- final hidden state and logits;
- greedy decoded tokens;
- prefix-hit continuation;
- MTP accepted-token continuation; and
- multimodal prompts.

## 17. Performance and Promotion Gates

### 17.1 Kernel gates

Each promoted fast path must demonstrate:

- that its executable kernel, launcher, packing path, and dispatch are
  Llaminar-owned source built by the normal CPU, CUDA, or ROCm build;
- parity within the semantic budget;
- complete state parity for recurrent kernels;
- no unsupported-shape silent fallback;
- no implicit synchronization;
- graph-capture compatibility where the surrounding path is captured;
- stable workspace ownership;
- performance no worse than the retained fallback over its promoted domain;
  and
- profiler evidence explaining the chosen dispatch boundary.

The release gate also inspects build metadata, link maps, dynamic dependencies,
runtime logs, and source includes/imports for prohibited external kernel
dependencies. Reference-only benchmark jobs are kept separate from production
acceptance binaries so an installed comparison library cannot mask an
undeclared dependency.

### 17.2 End-to-end measurements

Measure separately:

- text prefill throughput;
- decode throughput and per-token latency;
- 1M-context memory growth and long-context decode;
- prefix-hit TTFT and cache payload size;
- prefix match unit versus physical block size, hit rate, and scheduler-split
  overhead;
- partial-block copy bytes, copy latency, and deferred-free pressure;
- KDA state checkpoint store/restore cost;
- KDA launch count per layer for staged and fused decode;
- native vision encoder and bridge latency;
- MoE router, dispatch, expert compute, and return time;
- EP communication and load imbalance;
- AttnRes overhead;
- Gated MLA prefill epilogue fusion and decode gate-stream overlap;
- token-by-top-k chunk count and grouped-MoE launch efficiency;
- MXFP native versus expanded-BF16 memory and throughput;
- grouped request throughput;
- MTP acceptance and speedup; and
- executor/host synchronization overhead.

Benchmark modes include single request, moderate continuous batch, saturated
throughput, cold prefix miss, hot prefix hit, and mixed multimodal traffic.

### 17.3 Promotion order

Promotion is incremental:

1. CPU semantic primitives.
2. Single-device CUDA text correctness.
3. Single-device ROCm text correctness.
4. CUDA/ROCm optimized KDA.
5. TP text graph.
6. EP static placement.
7. PP including AttnRes payload.
8. Native MXFP.
9. Native vision.
10. Prefix cache and MTP on every promoted topology.

A later surface cannot retroactively mark an earlier untested surface as
supported.

## 18. Observability and Diagnostics

Add stage and runtime telemetry for:

- KDA kernel/backend selection;
- KDA state layout, precision, and bytes per request;
- prefix checkpoint source and matched token;
- match unit, physical block sizes, exact hit tokens, and contributing cache
  groups;
- partial alias promotion/eviction and copy-on-write source/destination;
- scheduler-split and same-step-reuse deferral counts;
- KDA prefix restore/replay/fallback counts;
- AttnRes block count and stage time;
- Gated MLA gate-stream overlap enabled/disabled and join wait time;
- KDA versus Gated-MLA layer timing;
- router and top-k timing;
- expert load histogram and participant imbalance;
- dispatch/return bytes and time;
- logical token-by-top-k work and physical chunk count;
- expert runtime-table capacity and occupancy;
- MXFP expansion/native route;
- vision patch/token count and encoder time;
- MTP verifier state capture/publication; and
- any host synchronization or uncaptured path.

Debug snapshots must name K3 semantics rather than reuse misleading GDN labels.
State hashes should make prefix and MTP divergence localizable to layer,
request, and state component.

## 19. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| The vLLM K3 release branch remains private or changes before publication | Blog detail is mistaken for stable code | Record the 2026-07-27 audit boundary; re-pin the public branch and diff it against the blog and merged prerequisites during Phase 1. |
| Public report changes assumptions from precursor code | Rework or silent wrong model | Freeze exact semantics only after release intake; keep provisional code behind unregistered fixtures. |
| 896 experts overflow 256-entry device tables | Incorrect routing or unsupported EP | Redesign metadata before model integration; add exact 896/top-16 tests. |
| 64-participant target exceeds current device metadata | Deployment cannot match recommended topology | Make participant capacity variable/domain-scoped and test communicator subsets. |
| KDA per-key gating is forced through GDN interfaces | Incorrect recurrence | Create a distinct typed KDA contract. |
| The unmerged FlashKDA proof-of-concept gate formula is treated as authoritative | Wrong decay despite a working kernel | Use `lower_bound` only as a dispatch/evidence clue until the released config and model code confirm every transform. |
| Chunk prefill differs from serial decode at publishable rows | Later token divergence | Require exact/decode-equivalent verifier state capture and serial continuation tests. |
| Prefix cache restores KV without KDA state | Catastrophic continuation error | Treat hybrid state as atomic prefix payload; fail or replay on missing components. |
| Partial cache aliases survive promotion, eviction, or reuse | Stale state can be returned for an unrelated request | Maintain reverse alias metadata and test every lifecycle transition. |
| A partial cache hit appends to shared physical state | Cross-request corruption | Require copy-on-write, exact copy ordering, endpoint lifetime fences, and deferred same-step reuse. |
| AttnRes bank is omitted at PP boundary | Cross-stage parity failure | Make bank/partial state an explicit PP payload and test inside-block splits. |
| FlashMLA is mistaken for Gated MLA | Incorrect attention | Use it only as read-only design and performance evidence after equations are known; implement the kernel inside Llaminar. |
| Decode gate projection overlaps MLA without an explicit dependency | Nondeterministic or stale gate output | Represent stream fork/join and events in the stage contract; compare overlapped and serialized execution. |
| NVIDIA LatentMoE is mistaken for Stable LatentMoE | Incorrect FFN graph | Trace exact K3 source/config; retain unfused reference stages. |
| `tokens * top_k` exceeds a physical MoE grid/index limit | Dropped or misindexed routes | Chunk the logical grid with global route metadata and test exact boundaries. |
| Native MXFP format is unavailable on older hardware | Backend gap | Maintain expand/dequantize correctness path; promote hardware-specific native kernels separately. |
| Quant scale blocks and TP shards misalign | Silent weight corruption | Represent scale geometry in tensor metadata and test shard boundaries. |
| No public ROCm KDA reference | Slower bring-up and tuning risk | Port from naive/chunk equations, compare full state, and profile independently. |
| Full model is too large for routine CI | Weak regression coverage | Publish extracted layer/operator fixtures and a manifest-driven miniature graph. |
| Vision processor evolves independently of model code | Token/feature mismatch | Pin processor revision and include it in fingerprints and fixtures. |
| Upstream KDA cache contribution lands after local design | Duplicate/rework | Keep cache interfaces typed and audit the contribution before promotion. |
| AttnRes repository lacks an explicit license | Provenance risk | Implement from paper-level semantics; do not copy unlicensed source verbatim. |
| A reference backend becomes an accidental build or runtime dependency | K3 binaries are not portable or independently maintainable | Enforce the first-party ownership policy, isolate fixture/benchmark environments, scan includes/imports and link/runtime dependencies, and run acceptance tests without reference libraries installed. |

## 20. Definition of Done

Kimi K3 support is complete only when all applicable requirements below pass:

1. The public K3 checkpoint, tokenizer, processor, and chat template load from a
   pinned revision with verified digests.
2. Every checkpoint tensor is consumed, intentionally ignored with evidence,
   or rejected. No unknown tensor is silently skipped.
3. The model architecture map and tensor manifest are committed.
4. KDA prefill/decode outputs and complete recurrent states match the public
   reference across the required test matrix.
5. Prefix-cache hits produce the same continuation as a fresh prefill.
6. AttnRes matches the public K3 implementation and survives PP boundaries.
7. Gated MLA, Stable LatentMoE, SiTU, and native vision match released
   reference activations.
8. 896-expert top-16 routing is correct on CPU, CUDA, and ROCm.
9. Static expert-parallel execution works with empty participants and the
   promoted participant count.
10. Single-device, TP, PP, and EP paths pass their declared parity matrices.
11. MXFP4/MXFP8 checkpoint loading is correct; every promoted native kernel
    passes expanded-reference parity.
12. Text and multimodal greedy token parity pass on the published validation
    prompts.
13. Grouped request execution isolates all KDA/KV/MLA state.
14. MTP verification publishes only accepted KDA, convolution, MLA/KV, and MoE
    runtime state.
15. Graph-captured paths perform no forbidden allocation, implicit default
    stream work, or per-step host decision sync.
16. Prefix, model, processor, quantization, and expert-placement fingerprints
    invalidate incompatible state.
17. Benchmark and profiler evidence supports each promoted fast path.
18. Unsupported backend/topology/format combinations fail with an actionable
    reason rather than silently changing semantics.
19. User documentation includes supported hardware, quantization, topology,
    model acquisition, CLI examples, limitations, and memory guidance.
20. Upstream source revisions, licenses, and attribution are recorded.
21. Fine-grained prefix hits return exact logical token counts, converge all
    KDA/MLA cache groups, and pass partial-block copy-on-write lifecycle tests.
22. Every Llaminar-owned optimized KDA candidate is selected only inside its
    verified dtype, shape, gate, and accelerator domain; external KDA
    implementations are never production dispatch candidates.
23. Gated MLA's fused prefill epilogue and overlapped decode gate path match the
    serialized reference and have explicit dependency tests.
24. SiTU parameter mapping and token-by-top-k grid chunking match the unfused
    expert reference on every promoted NVIDIA and AMD backend.
25. Normal CPU, CUDA, and ROCm builds, tests, and serving binaries succeed in
    environments without TRTLLM-Gen, TensorRT-LLM, FlashInfer, DeepGEMM,
    FlyDSL, AITER, Triton, FLA, FlashKDA, FlashMLA, or their generated
    artifacts.
26. The production link graph, dynamic dependency list, source-dependency
    scan, and runtime-dispatch logs demonstrate that every K3 kernel selected
    in production is a Llaminar implementation.
27. CPU, CUDA, and ROCm each pass the declared primitive and model parity
    matrix using only Llaminar kernels; unsupported native instructions select
    a Llaminar fallback or fail with an actionable message.

## 21. Deliverables

- this normative project plan;
- K3 release-intake evidence manifest;
- architecture and tensor-name map;
- Kimi3 graph/config/schema/buffer implementation;
- KDA kernel interfaces and first-party CPU/CUDA/ROCm implementations;
- KDA prefix/MTP state support;
- AttnRes stages and optimized kernels;
- Gated MLA implementation and kernels;
- first-party CPU/CUDA/ROCm Stable LatentMoE and SiTU implementations;
- 896-expert/64-participant runtime-table redesign;
- MXFP4/MXFP8 tensor, loader, and first-party CPU/CUDA/ROCm compute paths;
- native vision processor and graph;
- extracted public-reference fixtures;
- unit, integration, parity, distributed, and performance tests;
- profiling reports and dispatch policy artifacts;
- production dependency-audit manifest and clean-environment build evidence;
- K3 runtime documentation and examples; and
- release notes describing promoted and unsupported surfaces.

## 22. Immediate Backlog

### Until the public checkpoint and K3 branch appear

1. Add a standalone KDA oracle test generator from FLA's naive implementation.
2. Draft the typed KDA kernel and state-layout contracts.
3. Add synthetic per-key gate recurrence tests for CPU/CUDA/ROCm.
4. Inventory all `kDeviceMoEMaxExperts`, participant-count, route-width,
   serialization, and fingerprint assumptions.
5. Write the variable-size/domain-segmented MoE runtime-table design.
6. Add 896-expert/top-16 router-only tests where existing kernels already
   support that shape.
7. Prototype Block-AttnRes as unfused graph stages with synthetic tensors.
8. Define AttnRes PP payload layout and lifetime.
9. Add an upstream release-intake script/checklist that records digests and
   generates a tensor inventory without loading the full model.
10. Prepare storage and hardware estimates for BF16-expanded and native MXFP
    checkpoints.
11. Write the Llaminar cache design for chained match-unit hashes, exact
    logical hit lengths, hybrid-group convergence, partial aliases, and
    copy-on-write.
12. Add cache-only synthetic tests for scheduler-split checkpoints, same-step
    reuse deferral, alias promotion/eviction, copy ordering, and endpoint
    lifetime.
13. Add old-gate and `lower_bound`-gate KDA fixtures plus explicit rejection
    tests for shapes outside the Llaminar SM90 candidate's verified domain.
14. Regenerate license-compatible AttnRes fixture tensors from the merged Liger
    reference in an isolated environment; import no kernel source or runtime
    dependency. Test depths at, below, and above its 32-block fast-path limit.
15. Draft phase-specific Gated MLA stage contracts and the explicit decode
    gate-stream fork/join.
16. Add a backend-neutral logical token-by-top-k chunk planner and boundary
    tests for K3's top-16 routing.

### When public release artifacts appear

1. Pin the Moonshot, Hugging Face, Transformers, vLLM, SGLang, FLA,
   FlashInfer, and FlashKDA revisions consulted for the reference audit and
   offline fixture generation.
2. Capture all K3 configuration and processor artifacts.
3. Search again for official K3 branches/PRs and Moonshot reference code.
4. Diff the public vLLM K3 branch against merged cache commits `6bc6f2d` and
   `481e481`, FlashKDA proof-of-concept `e4734b8`, and every fusion boundary
   stated in the preview blog.
5. Record which K3 code was newly published, which prerequisite was already on
   `main`, and which preview feature remains absent or vendor-binary-only.
6. Generate the tensor manifest and architecture evidence map.
7. Run trusted remote/reference inference to capture primitive and layer
   fixtures before attempting optimization.
8. Resolve the model's released weight format and viable conversion path.
9. Update this plan's known-unknown and phase status sections with evidence.

## 23. Upstream Watchlist

The following sources should be checked during release intake and before each
major promotion:

- [Kimi K3 technical blog](https://www.kimi.com/en/blog/kimi-k3)
- [vLLM K3 implementation preview](https://vllm.ai/blog/2026-07-22-kimi-k3-preview)
- [MoonshotAI GitHub repositories](https://github.com/orgs/MoonshotAI/repositories)
- [MoonshotAI Hugging Face models](https://huggingface.co/moonshotai/models)
- [Transformers K3 branches](https://github.com/huggingface/transformers/branches/all?query=k3)
- [Transformers Kimi pull requests](https://github.com/huggingface/transformers/pulls?q=is%3Apr+kimi)
- [vLLM public branches](https://github.com/vllm-project/vllm/branches/all)
- [vLLM K3-labeled pull requests](https://github.com/vllm-project/vllm/pulls?q=is%3Apr+label%3Ak3)
- [vLLM Kimi pull requests](https://github.com/vllm-project/vllm/pulls?q=is%3Apr+kimi)
- [vLLM hybrid partial-cache RFC #45702](https://github.com/vllm-project/vllm/issues/45702)
- [vLLM partial-cache primitives PR #45939](https://github.com/vllm-project/vllm/pull/45939)
- [vLLM hybrid partial-hit PR #46384](https://github.com/vllm-project/vllm/pull/46384)
- [vLLM FlashKDA proof of concept PR #43833](https://github.com/vllm-project/vllm/pull/43833)
- [vLLM hybrid PD cache PR #45804](https://github.com/vllm-project/vllm/pull/45804)
- [vLLM KDA/MLA NIXL PR #49762](https://github.com/vllm-project/vllm/pull/49762)
- [SGLang KDA backend](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kda_backend.py)
- [FLA KDA](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda)
- [Moonshot FlashKDA](https://github.com/MoonshotAI/FlashKDA)
- [FlashInfer KDA pull requests](https://github.com/flashinfer-ai/flashinfer/pulls?q=is%3Apr+kda)
- [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals)
- [Liger AttnRes kernel](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/attn_res.py)
- [Kimi Linear](https://github.com/MoonshotAI/Kimi-Linear)

The watchlist is evidence collection, not an authority override. The pinned K3
release artifacts remain the source of truth for model semantics.
