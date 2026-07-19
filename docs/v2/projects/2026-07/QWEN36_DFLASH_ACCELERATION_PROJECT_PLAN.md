# Qwen 3.6 DFlash Acceleration Project Plan

Status: research complete; implementation not started

Research snapshot: 2026-07-18

Scope: Llaminar V2, Qwen3.6-27B dense and Qwen3.6-35B-A3B MoE

## Objective

Add lossless DFlash speculative decoding for Qwen 3.6 to Llaminar V2. The
implementation must support the released Z-Lab Qwen3.6 DFlash checkpoints,
reuse Llaminar's accepted-state speculative transaction, preserve target-model
output semantics, and become a speed-positive alternative to both ordinary
decode and the existing MTP proposer where measurements justify it.

The first promotion target is SingleDevice on CUDA, ROCm, and CPU. LocalTP,
NodeLocalTP, PP, and ExpertOverlay follow only after the SingleDevice contract
is correct, graph-capturable, prefix-safe, and benchmarked. CPU is a required
semantic backend even where the DFlash drafter is not speed-positive.

This plan is intentionally separate from the existing
[`MTP_VLLM_STYLE_PROJECT_PLAN.md`](../2026-06/MTP_VLLM_STYLE_PROJECT_PLAN.md).
DFlash and MTP are different proposers, but they must converge on one verifier,
rejection-sampling, and accepted-state publication transaction.

## Executive Decision

Implement DFlash as a new `ISpeculativeProposer` behind Llaminar's existing
lossless speculative transaction. Do not build a second target verifier or a
second GDN/KV publication path.

The target design is:

```text
per-device Qwen 3.6 target graph
  prefill or B-row verifier
        |
        +-- selected post-layer residual taps ---------------------+
        |                                                          |
        v                                                          v
existing target logits/state slots                    concat -> FC -> RMSNorm
        |                                                          |
        |                                             per-draft-layer K/V + RoPE
        |                                                          |
        |                                               DFlash context KV cache
        |                                                          |
        |                                      [ready anchor, MASK x (B - 1)]
        |                                                          |
        |                                      one parallel DFlash query graph
        |                                                          |
        |                                      shared target embedding + LM head
        |                                                          |
        +<---------------- B - 1 draft tokens ---------------------+
        |
        v
existing B-row grouped target verifier
        |
        v
existing device rejection sampler
        |
        +-- publish only accepted target KV/GDN/short-conv state
        +-- append only accepted target-feature K/V to DFlash cache
        +-- discard the rejected suffix and every proposal-query K/V row
```

For a trained block size `B`, DFlash proposes `d = B - 1` tokens. The clean
target-produced anchor occupies block row zero. Qwen 3.6 block 16 therefore
requires 15 draft tokens and exactly 16 target verifier rows. This matches the
current Llaminar grouped-verifier certification range rather than requiring a
new hidden depth limit.

Block-parallel matrix acceleration is a first-class part of this design, not a
late CUDA-only optimization. DFlash converts sequential `M=1` drafting into
fixed small-`M` GEMMs, with `B=16` matching an Ampere Tensor Core row tile.
Llaminar will expose one backend-neutral matrix-plan contract and lower it to:

- BF16/FP16 Tensor Core GEMMs and the existing WMMA/IMMA families on CUDA
  Ampere, including RTX 3090 (`sm_86`);
- FP16 packed-dot/VALU GEMMs and attention on ROCm MI50 (`gfx906`), which has no
  MFMA matrix cores and no native BF16 arithmetic; and
- the ordinary floating/quantized CPU routes as the semantic fallback.

The graph, cache, masking, and accepted-state transaction stay identical across
backends. Activation storage, packed-weight representation, physical kernel,
and preferred block size are backend policy dimensions. Proposal-side numeric
differences are acceptance/performance concerns; they never relax the target
verifier's serial-equivalence contract.

## Research Findings

### What DFlash does

DFlash is speculative decoding with a small block-diffusion transformer as the
drafter. It differs from autoregressive MTP/EAGLE-style drafting in three ways:

- It predicts all masked positions in a block in one forward pass instead of
  running one draft forward per proposed token.
- It conditions every draft layer on hidden features extracted from several
  layers of the target model.
- It projects those target features into persistent Key/Value context for every
  draft layer rather than fusing them only into the draft input.

The speed model remains the normal speculative-decoding equation: draft time
plus target verification time, divided by the expected number of committed
tokens including the target correction/bonus token. The distinctive advantage
is that the draft cost is approximately insensitive to moderate block size,
whereas an autoregressive drafter's cost grows with draft depth. The paper's
algorithm, training method, and original Qwen3 results are described in
the [DFlash paper](https://arxiv.org/html/2602.06036). The paper predates the
released Qwen 3.6 checkpoints; Qwen 3.6 support must be derived from the later
checkpoint configurations and serving implementations, not inferred from the
paper's Qwen3 benchmark tables.

### Exact inference transaction

Let the target cache contain positions `[0, P)`, and let `x_P` be the ready
target-produced token that has not yet been processed by the target model.
For block size `B`, set `d = B - 1`.

1. Form the DFlash query block `[x_P, MASK, ..., MASK]` at absolute positions
   `[P, P + B)`.
2. Embed all `B` rows with the target model's token embedding.
3. Run one DFlash draft forward. Each draft layer attends to its persistent
   target-feature K/V context and the current query block.
4. Apply the target model's LM head to draft rows `1..B-1` and greedily select
   proposals `y_1..y_d` in parallel.
5. Run the target model once on `[x_P, y_1, ..., y_d]`, producing target
   posterior rows `z_1..z_{d+1}`.
6. Let `a` be the number of consecutive proposals for which `y_i == z_i` in
   greedy mode, or the number accepted by the exact rejection sampler in
   stochastic mode.
7. Emit `[y_1, ..., y_a, z_{a+1}]`.
8. Publish target input-state rows `0..a`, append the corresponding target
   feature rows to the DFlash context K/V cache, and discard everything else.
   `z_{a+1}` becomes the clean anchor for the next cycle.

The upstream reference implements the same shifted-cache transaction in
[`dflash/model.py`](https://github.com/z-lab/dflash/blob/main/dflash/model.py).
The following row vocabulary must be explicit in Llaminar so an off-by-one
cannot hide behind a generic `accepted_count` field:

| Quantity | Value |
|---|---:|
| Block/query rows | `B` |
| Proposed draft tokens | `d = B - 1` |
| Accepted draft tokens | `a`, in `[0, d]` |
| Emitted tokens | `a + 1` |
| Target state rows committed | `a + 1`, verifier rows `0..a` |
| Target feature rows appended to DFlash | `a + 1`, verifier rows `0..a` |
| Persistent proposal-query K/V rows | `0` |
| Next ready anchor | target sample from verifier logit row `a` |

If `a = 0`, the anchor input row is still committed and the target correction
is emitted. If `a = d`, all `B` verifier input rows are committed and the final
target row supplies the bonus token. EOS and maximum-output truncation must
shorten emission and publication consistently rather than commit invisible
suffix state.

### Target feature semantics

The reference reads Transformers `hidden_states[layer_id + 1]`. Because
`hidden_states[0]` is the embedding output, this is the residual stream after
target transformer layer `layer_id`, equivalently the input to target layer
`layer_id + 1`.

Llaminar will name this semantic `PostLayerResidual(layer_id)`. In the Qwen 3.6
graphs, the tap belongs immediately after the layer's final FFN residual update.
It applies identically to target GDN and full-attention layers.

For GGUF interoperability, `dflash.target_layers` will store the residual-stream
extraction indices used by llama.cpp: raw Hugging Face `target_layer_ids` plus
one. The converter and loader must test this mapping explicitly. Internally,
Llaminar converts each value back to `PostLayerResidual(value - 1)`.

The selected rows are concatenated in checkpoint order, projected once by
`fc.weight`, and normalized by the DFlash hidden-context RMSNorm. The initial
implementation must use a real concatenated buffer and one GEMM, matching the
reference operation order. Decomposing the FC into per-tap partial GEMMs is a
later optimization because changed FP parenthesization can alter draft logits
and acceptance economics.

### Draft attention semantics

The released Qwen 3.6 checkpoints are not uniformly noncausal. Their draft
layers are mixed:

- `sliding_attention` layers use causal attention and the configured sliding
  window.
- `full_attention` layers allow bidirectional attention within the current
  masked query block.
- Both layer types attend to all retained past target-feature K/V rows allowed
  by their cache policy.

The Z-Lab PyTorch reference uses a broad noncausal switch, while its MLX
implementation and the current vLLM/SGLang implementations resolve causality
per layer. Llaminar must follow the checkpoint's per-layer `layer_types`, not a
global `causal=false` flag. This is an important Qwen 3.6 compatibility rule.

For a sliding layer with window `W`, a query at position `p` may see positions
`[p-W+1, p]`; future query rows remain masked. A full layer may see the entire
committed context plus all `B` rows of the current block. Context K/V rows use
their target token positions, and query rows use `[P, P+B)` for RoPE.

### How the draft cache works

Every DFlash layer uses the same learned K/V projections for two sources:

- fused target features, which form persistent context K/V; and
- current block embeddings/hidden states, which form transient query K/V.

The context path does not execute Q projection, attention output projection, or
FFN. It performs K/V projection, K RMSNorm, RoPE on K, and cache storage. The
query path performs the normal draft transformer layer.

The simple reference temporarily appends context and proposal K/V and then
crops back to the committed target position. Production Llaminar should expose
`CachePlusInlineKV` attention: persistent context comes from
`DFlashContextKVCache`, while the block's query K/V remains inline and is never
published. This removes rollback traffic and makes accepted-state ownership
obvious.

After target verification, Llaminar projects and appends only target feature
rows `0..a`. It never commits accepted proposal-query K/V. Accepted positions
are represented by the more accurate target feature K/V produced by the
verifier.

### Training facts relevant to inference

The DFlash drafter is trained against a frozen target. Training blocks start
with a clean randomly selected anchor and mask the remaining positions. Blocks
are isolated from one another, early positions receive higher exponentially
decaying loss weight, and the target embedding and LM head remain frozen and
shared. The paper's default was a five-layer, block-16 drafter with target
features selected from shallow through deep layers.

Two operational consequences follow:

- The checkpoint embedding and LM head must not be duplicated; Llaminar must
  bind the target model's resident weights.
- A checkpoint trained at block 16 may run at smaller blocks. Larger-than-trained
  blocks are invalid. Dynamic policy candidates should initially be drawn from
  `{4, 8, 16}` rather than inventing a larger capacity.

### Released Qwen 3.6 checkpoint inventory

The source of truth is the pinned checkpoint config, not a hard-coded model
name. The two current checkpoints have different config nesting, so the
converter must accept `block_size` at the top level or inside `dflash_config`.

| Field | Qwen3.6-27B-DFlash | Qwen3.6-35B-A3B-DFlash |
|---|---:|---:|
| Target | `Qwen/Qwen3.6-27B` | `Qwen/Qwen3.6-35B-A3B` |
| Pinned HF revision | `0919688658996800f86b895034249700e9481106` | `f181eece646affea2c38b2765f1aaa01a9734ccd` |
| BF16 parameters | 1,730,213,120 | 385,906,176 |
| Approximate BF16 weight bytes | 3.22 GiB | 0.72 GiB |
| Draft hidden / FFN | 5120 / 17408 | 2048 / 6144 |
| Q heads / KV heads / head dim | 32 / 8 / 128 | 32 / 8 / 128 |
| Draft layers | 5 | 6 |
| Layer types | 4 sliding + 1 full | 5 sliding + 1 full |
| Sliding window | 2048 | 4096 |
| Trained block size | 16 | 16 |
| Raw HF target layer ids | `[1,16,31,46,61]` | `[1,6,11,16,22,27,32,37]` |
| GGUF extraction ids | `[2,17,32,47,62]` | `[2,7,12,17,23,28,33,38]` |
| Mask token id | 248070 | 248077 |
| Vocabulary | 248320 | 248320 |
| RoPE theta / max positions | 10,000,000 / 262144 | 10,000,000 / 262144 |

The [27B checkpoint card](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash)
currently describes evolving serving support and gives no Qwen 3.6 performance
acceptance target. The
[35B-A3B checkpoint card](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash)
reports a joint 40K-context retrain and upstream SGLang/B200 results up to 3.61x
at concurrency one and 2.89x at concurrency 32. Those numbers are useful priors,
not Llaminar promotion gates: the hardware, target precision, kernels, scheduler,
and workload differ.

### Implementation review

| Implementation | What Llaminar should adopt | What not to copy blindly |
|---|---|---|
| [Z-Lab reference](https://github.com/z-lab/dflash) | Exact block/anchor shift, target feature order, shared embedding/head, draft-cache crop semantics, layer-level oracle | Python host control and a global noncausal assumption |
| [vLLM proposer](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/dflash.py) and [model](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_dflash.py) | Stable query buffers, separate context-KV preparation, per-layer causal resolution, stacked all-layer context K/V projection, paged-cache writes | Backend-specific cache restrictions or CUDA-only assumptions |
| [SGLang model](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/dflash.py) and [worker](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/dflash_worker_v2.py) | Persistent fixed buffers, compact sliding caches, accepted-prefix direct writes, deferred Qwen hybrid-state commit, batching/overlap scheduler, ROCm coverage | Framework-specific pool metadata and unsupported-feature restrictions unrelated to Llaminar |
| [llama.cpp model](https://github.com/ggml-org/llama.cpp/blob/master/src/models/dflash.cpp), [speculative driver](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp), and [merged DFlash PR](https://github.com/ggml-org/llama.cpp/pull/22105) | GGUF architecture/tensor naming, target-layer `+1` conversion, target embedding/head sharing, separate encoder/context injection graph | Snapshot/replay of hybrid target state and host-oriented sampler loops |

The upstream implementations converge on the same production shape: materialize
target context K/V separately, keep the block query graph fixed, verify once,
and publish only the accepted prefix. That convergence is stronger evidence
than any one framework's temporary API or backend limitation.

## Llaminar Starting Point

Llaminar already owns the hardest losslessness machinery:

- `src/v2/execution/mtp/MTPSpecTransactionDriver.*` plans accepted outcomes and
  atomic publication.
- `src/v2/execution/runner/MTPVerifierForwardExecutor.*` runs the `d+1` target
  verifier.
- `DeviceGraphOrchestrator` owns persistent speculative metadata, device
  sampling, verifier slots, and accepted-state publication.
- Qwen 3.6 target KV, GDN recurrence, short-conv state, terminal hidden/logits,
  sampler history, and positions already have accepted-only publication rules.
- Grouped verifier kernels are certified against serial `M=1` decode through
  `M=2..16` plus a deeper `M=31` sentinel across CPU, CUDA, ROCm, dense, MoE,
  and supported weight formats.
- Prefix cache payloads already include target hybrid state, terminal state,
  and MTP sidecar state with fingerprints.
- `Qwen35Schema` and `Qwen35MoESchema` are the active Qwen 3.6 dense/MoE target
  graphs, including GDN/full-attention hybrid execution.

The reusable target verifier is the main schedule advantage. DFlash block 16
uses the already-important `M=16` verifier shape. The new implementation should
not modify target math merely because the proposer changed.

The missing pieces are:

- a generic proposer interface and proposer-neutral names for the shared MTP
  transaction;
- a separate DFlash GGUF loader and target/draft compatibility manifest;
- post-layer target feature taps that remain on device;
- feature fusion and per-layer context K/V materialization;
- a mixed causal/noncausal DFlash transformer graph;
- per-layer full/sliding DFlash context K/V ownership;
- DFlash prefix payloads and fingerprints;
- graph-captured block preparation, parallel sampling, and request batching;
- DFlash-specific policy, observability, and benchmark training.

## Non-Negotiable Invariants

1. Graphs remain per-device and symmetric. DFlash does not introduce a nested
   multi-device graph.
2. The target verifier remains the production target model. DFlash can change
   proposal quality, never the requested target distribution.
3. Grouped target verifier rows remain serial-`M=1` equivalent for every
   continuation state that is publishable.
4. Rejected target rows, rejected draft rows, and the ready bonus token never
   mutate live target or DFlash state.
5. DFlash proposal-query K/V is always transient. Only K/V projected from
   accepted target feature rows becomes persistent.
6. Target feature extraction is `PostLayerResidual`, with an explicit tested
   checkpoint-index conversion.
7. `sliding_attention` is causal/windowed and `full_attention` is noncausal
   within the DFlash query block unless an explicit future checkpoint override
   says otherwise.
8. Embedding and LM head weights are borrowed from the target model and share
   its lifetime, precision, placement, and TP vocabulary ownership.
9. GPU work uses explicit non-null streams, arena/workspace-backed storage, and
   event ordering. There are no hot-path allocations or implicit default-stream
   dependencies.
10. Host/device movement goes through graph buffer contracts and
    `TransferEngine`. The GPU proposal/verify/sample/publication loop has no
    per-step D2H decision sync.
11. Prefix restores require a compatible DFlash sidecar or take a named,
    counted policy action. Missing draft state is never treated as valid state.
12. CPU, CUDA, and ROCm implement the same transaction and mask semantics. A
    backend may remain policy-disabled for performance, but not semantically
    divergent.
13. DFlash proposal math may use backend-specific activation precision, packed
    weights, and matrix kernels only through common graph-stage contracts. The
    selected physical route and precision are fingerprinted, counted, and
    covered by eager/captured parity tests.
14. Reduced-precision proposal routes never enter the target verifier's
    publishable state path. Target verifier GEMM and stage execution retain
    `beginVerifierDecodeEquivalentScope()` and the complete grouped-row
    serial-byte gates.

## Target Runtime Architecture

### Proposer-neutral transaction

Phase 1 extracts the existing MTP transaction into proposer-neutral types while
retaining compatibility aliases so the refactor is behavior-only for MTP.

```cpp
enum class SpeculativeProposerKind
{
    None,
    MTP,
    DFlash,
};

struct SpeculativeStepShape
{
    int request_count;
    int draft_tokens_per_request;
    int target_rows_per_request; // draft + 1
};

struct SpeculativeAcceptedOutcome
{
    int request_id;
    int accepted_draft_count;
    int committed_target_rows;   // accepted_draft_count + 1
    int emitted_token_count;     // accepted_draft_count + 1
};

class ISpeculativeProposer
{
public:
    virtual bool prepareAfterTargetForward(const TargetFeatureBatch &) = 0;
    virtual bool propose(const SpeculativeStepShape &,
                         SpeculativeDraftBatch *) = 0;
    virtual bool publishAcceptedContext(const DeviceAcceptedOutcomes &) = 0;
    virtual void clearRequestState() = 0;
    virtual ~ISpeculativeProposer() = default;
};
```

Planned moves/aliases:

- `MTPSpecTransactionDriver` -> `SpeculativeDecodeTransactionDriver`
- `MTPSpecDecodeMetadata` -> `SpeculativeDecodeMetadata`
- `MTPRejectionSampler` -> `SpeculativeRejectionSampler`
- `MTPVerifierForwardExecutor` -> `SpeculativeVerifierForwardExecutor`
- MTP names remain aliases during migration; current CLI and tests must not
  change behavior.

MTP and DFlash are mutually exclusive proposers in the first release. A later
portfolio controller may choose among baseline, MTP, and DFlash per request,
but it is not allowed to compose two proposers in one transaction until a new
correctness contract is designed.

### Checkpoint and weight ownership

DFlash is a separate GGUF model context, not a trailing Qwen `nextn` block. Add:

- `scripts/convert_dflash_hf_to_gguf.py`
- `src/v2/models/dflash/DFlashGraphConfigBuilder.*`
- `src/v2/models/dflash/DFlashSchema.h`
- `src/v2/models/dflash/DFlashGraph.*`
- `src/v2/execution/dflash/DFlashWeightManifest.*`
- `src/v2/execution/dflash/DFlashModelCompatibility.*`

The converter should align with upstream llama.cpp GGUF conventions where they
exist:

- `general.architecture = dflash`
- `dflash.block_size`
- `dflash.target_layers` using extraction ids (`HF id + 1`)
- mask token id
- standard Qwen3 hidden/head/FFN/RoPE/window metadata
- ordered `layer_types`
- `fc.weight`, context hidden norm, final norm, and `blk.N.*` transformer
  tensors

The DFlash file must not duplicate `token_embd.weight` or `output.weight`.
`DFlashWeightBindings` borrows those target bindings explicitly. The target
context owns their lifetime, and the memory planner counts them once.

The converter also writes a compatibility manifest containing target family,
target layer count, target hidden width, vocabulary/tokenizer hash, extraction
semantics, checkpoint revision, and trained block size. It should accept the
target GGUF as an input so these fields are not guessed from a model name.
Target quantization is allowed to differ from training; it may change acceptance
but not the speculative correctness proof. Prefix fingerprints still include
the actual target and DFlash precisions.

The converted BF16 tensors are the canonical checkpoint/oracle representation.
The runtime may derive persistent backend-packed views during load or model
preparation, never during a decode step. In particular, CUDA may retain BF16 or
materialize FP16 Tensor Core layouts, while MI50 may materialize FP16 weights
from canonical BF16 after acceptance certification instead of repeatedly
emulating BF16 through FP32. Dry-run and memory accounting include every
retained canonical or packed view; prefix and graph fingerprints identify the
effective proposal activation, weight, and KV precisions.

Loader validation rejects:

- target dense/MoE family mismatch;
- wrong target layer count or hidden width;
- invalid or unordered extraction layers;
- tokenizer, vocabulary, mask-token, or special-token mismatch;
- unsupported draft layer type;
- runtime block larger than the trained block;
- missing target embedding/LM-head bindings; or
- a draft context whose complete tensor inventory is not loaded.

### Runtime configuration

Add a separate `DFlashRuntimeConfig` under `RuntimeConfig` and
`OrchestrationConfig`. Preserve existing MTP options unchanged.

Initial CLI surface:

```text
--dflash-model <draft.gguf>             enable DFlash with a separate draft
--dflash-block-size <B>                 total block rows; proposals are B - 1
--dflash-verify-mode <mode>             greedy or speculative-sampling
--dflash-max-request-batch <N>          persistent request capacity
--dflash-block-policy <mode>            fixed, observe, or dynamic
--dflash-min-block-size <B>
--dflash-initial-block-size <B>
--dflash-max-block-size <B>
--dflash-activation-precision <precision> auto, bf16, or fp16
--dflash-matrix-policy <mode>            auto, portable, or accelerated
--dflash-kv-cache-precision <precision> initial release: fp16 or bf16
--dflash-prefix-policy <policy>         require-sidecar or bypass-request
```

`--dflash-model` implies enablement. Fixed block zero means use the checkpoint's
trained block. `--dry-run` must print both `B` and `B-1`, selected target taps,
layer causality/window, canonical and prepared draft weight bytes, requested and
effective activation/weight/accumulator precision, selected matrix route/layout,
per-request DFlash KV bytes, feature scratch, graph buckets, and target/draft
compatibility. `portable` forces the common non-accelerated oracle route;
`accelerated` requires a certified backend route and fails rather than quietly
falling back. `auto` may choose or bypass only through installed policy.

`--mtp` and `--dflash-model` are initially a validation error. Unsupported
requested DFlash configurations fail with an actionable reason; a runtime
policy bypass is legal only after successful initialization and must increment
an explicit counter.

### Target feature tap graph

Add a generic `LayerOutputTapStage`/`StageType::LayerOutputTap` rather than
embedding DFlash-specific logic into Qwen target schemas.

For selected target layer `L`, the resolver inserts the tap after the final FFN
residual stage and writes the current `[M, target_hidden]` rows into slice `k`
of a stable `[M, K * target_hidden]` feature buffer. `K` is the number of
selected layers. The stage is a device-local copy/scatter with a graph-capturable
fixed destination.

Properties:

- The tap is disabled when no feature consumer is registered.
- It captures real rows only; bucket padding is poisoned in tests and ignored.
- It is valid in prefill, ordinary decode, and speculative verifier graphs.
- The concatenated buffer has a lifetime spanning the selected target layers
  and cannot alias ordinary attention/FFN scratch.
- Prompt prefill is chunked. A chunk's selected features are fused and appended
  to DFlash context K/V before the buffer is reused, avoiding an unbounded
  full-prompt hidden-feature allocation.
- Verifier capture uses the fixed `B`-row bucket and later materializes only the
  accepted prefix.

The first oracle compares every tap and the concatenated tensor to the pinned
Z-Lab PyTorch model before implementing fused or decomposed feature paths.

### Feature fusion and context K/V graph

`DFlashFeatureFusionGraph` performs:

1. one `fc.weight` GEMM from `[M, K * target_hidden]` to
   `[M, draft_hidden]`;
2. DFlash hidden-context RMSNorm;
3. K and V projection for each draft layer;
4. K RMSNorm;
5. absolute-position RoPE on K; and
6. append of the selected prefix to the layer's DFlash context cache.

The initial path uses one ordinary K/V projection per layer. The performance
path stacks all draft-layer K and V weights and issues one grouped/fused GEMM,
then normalizes/rotates and writes each layer's cache directly. vLLM and SGLang
both demonstrate this optimization. Promotion requires counter evidence that
the fused path is used, not just a timing improvement.

The matrix plan must benchmark both accepted-row compaction (`M=sum(a+1)`) and
fixed bucket execution (`M=request_bucket*B`) followed by a prefix-only cache
write. Fixed `B=16` execution can remove compaction and align CUDA Tensor Core
tiles, but computing rejected feature rows is not assumed to be free. Dispatch
chooses from measured eager/captured evidence and reports useful versus padded
rows; it never changes the accepted prefix that becomes persistent.

For verifier output, the graph receives device-resident accepted counts and row
starts. It writes only `a+1` rows per request. It must not materialize accepted
counts on the host or copy the entire `B`-row feature tensor into live caches.

### DFlash draft graph

The DFlash schema is a Qwen3-style dense transformer with two named templates:

- `sliding_attention`: causal, configured window;
- `full_attention`: noncausal within the query block, full persistent context.

Each layer contains input RMSNorm, Q/K/V projections, per-head Q/K RMSNorm,
RoPE, context-plus-inline attention, output projection/residual, post-attention
RMSNorm, SwiGLU FFN, and residual. The graph finishes with DFlash final RMSNorm.

The query input is a persistent `[requests, B]` token buffer. A device kernel
writes one ready anchor and `B-1` mask ids per active request, along with
positions, row starts, cache slots, and active masks. The query graph embeds via
the target embedding table.

Only output rows `1..B-1` pass through the borrowed target LM head. Greedy draft
sampling is batched and device-resident. It produces one-hot proposals for the
existing exact stochastic rejection sampler; full draft probability tensors
are not part of the initial design.

Add `KVSourceMode::CachePlusInline` to the attention contract. The attention
kernel reads committed context K/V plus current query K/V without calling
`KVCacheAppendStage` for query rows. CPU, CUDA, and ROCm tests cover every
combination of causal/full mode, window boundary, context length, and block
bucket.

The proposal graph is a separate activation-precision island from the target
graph. Target taps enter through an explicit conversion boundary. Draft GEMM
inputs/outputs and query K/V may remain BF16 or FP16 resident, while RMSNorm,
softmax, and other reductions accumulate in FP32. No draft layer should
round-trip through FP32 storage merely because the target graph uses FP32
hidden rows. The canonical BF16 path is the numerical oracle; every alternative
storage route is judged by layer tolerances, proposal acceptance, and final
target correctness.

### Backend-specific matrix acceleration

DFlash's primary hardware advantage is weight reuse across the block: one
weight tile feeds `B` query rows instead of being streamed once for each
autoregressive draft token. For a 16-bit weight-dominated GEMM, the approximate
weight-side arithmetic intensity is `M` FLOP/byte, so `M=16` may still be
bandwidth-bound even with Tensor Cores. Promotion therefore requires measured
weight bandwidth and end-to-end committed-token throughput, not Tensor Core
occupancy or isolated TFLOPS alone.

The required Qwen 3.6 matrix inventory is:

| Logical bundle | 27B dense `(M,K,N)` | 35B-A3B `(M,K,N)` | Initial physical plan |
|---|---|---|---|
| Feature FC | `(M,25600,5120)` | `(M,16384,2048)` | One GEMM; compact and fixed-`B` variants |
| Stacked context K/V | `(M,5120,10240)` | `(M,2048,12288)` | One packed GEMM for every draft layer's K and V |
| Fused draft QKV | `(B,5120,6144)` | `(B,2048,6144)` | One packed GEMM with split output views |
| Attention output | `(B,4096,5120)` | `(B,4096,2048)` | Floating matrix route |
| Fused FFN gate/up | `(B,5120,34816)` | `(B,2048,12288)` | One packed GEMM with split output views |
| FFN down | `(B,17408,5120)` | `(B,6144,2048)` | Floating matrix route with fused SwiGLU input where profitable |
| Shared LM head | `(B-1,5120,248320)` | `(B-1,2048,248320)` | Target-owned floating or quantized route; compare natural and padded `M` |

Request batching flattens the first dimension to `requests*B`, or to
`requests*(B-1)` for draft logits. The supported inventory includes natural
rows `{3,7,15}`, padded buckets `{4,8,16}`, accepted-row `M=1..16`, and their
request-batched multiples. Padding is poisoned in parity tests and excluded
from logits, sampling, cache counts, and useful-work telemetry.

Add a backend-neutral `DFlashMatmulPlan`/descriptor bundle. It owns logical
matrix dimensions, packed-weight lifetime, output views, optional epilogues,
workspace requirements, activation/weight/accumulator precision, and a stable
dispatch id. Graph construction requests the bundle; backend factories select
the physical implementation. There are no backend `#ifdef`s in DFlash graph
semantics and no source-level one-shape overrides in production dispatch.

CUDA `sm_80+` lowering:

- start with preplanned cuBLASLt BF16/FP16 input and FP32-accumulate algorithms;
- create descriptors, choose algorithms, pack weights, and reserve arena
  workspace before capture; hot replay performs no heuristic search, handle
  creation, packing, `cudaMalloc`, or `cudaFree`;
- bind every launch to the graph's explicit non-default stream and maintain
  stable addresses across eager and captured execution;
- extend the existing Ampere WMMA FlashAttention path for
  `CachePlusInlineKV`, including the mixed causal/full masks and `B=16` query
  tile; and
- route quantized shared-head and later quantized draft matrices through the
  existing NativeVNNI IMMA candidate/training pipeline after adding the exact
  DFlash shapes to the shared inventory.

If cuBLASLt is not competitive at the exact small-`M` shapes, add CUTLASS or
local MMA candidates behind the same plan. Source spelling is not proof of
Tensor Core execution: production-selected floating and integer candidates
need isolated Nsight Compute HMMA/IMMA instruction evidence, while canonical
latency remains measured outside profiler replay.

ROCm `gfx906` lowering:

- use FP16 packed-dot (`v_dot2_f32_f16`) and vectorized VALU kernels with FP32
  accumulation; do not pretend MI50 has an MFMA/rocWMMA route;
- fuse QKV, gate/up, and all-layer context K/V so weight and activation tiles
  are reused even without matrix cores;
- compare canonical BF16-through-FP32 execution with load-time BF16-to-FP16
  prepared weights and an FP16-resident proposal graph; and
- extend the existing pure-VALU attention path for `CachePlusInlineKV` rather
  than forcing CUDA's WMMA tile geometry onto wave64.

CPU, CUDA, and ROCm share one candidate observation schema and common logical
shape inventory. Generated dispatch keys include backend architecture,
activation/weight format, `M/N/K`, request bucket, eager/captured mode, and
relevant layout/epilogue features. Exact shape winners are overlays above a
generic fallback. The runtime fails or takes a named counted proposer bypass
when no certified route exists; it never quietly switches to a hot-path
migration or allocation fallback.

Matrix acceleration is proposal-only unless separately certified through the
grouped-verifier suite. In particular, a faster CUDA Tensor Core algorithm may
change draft logits within declared tolerances, but target verifier rows and
every publishable target continuation byte retain the backend's canonical
serial-`M=1` arithmetic identity.

### DFlash context KV ownership

Add `DFlashContextKVCacheSet`, one cache per draft layer and request:

- sliding layers retain only their configured recent context window;
- full layers retain the entire context;
- K/V precision is explicit and initially floating point;
- cached-token count is logical per request, even when physical sliding storage
  rotates;
- TP records local KV head start/count;
- query K/V has no live-state slot;
- clear, truncate, prefix export/import, and graph-capture reset have explicit
  semantics.

For BF16/FP16 Qwen 3.6, each layer's K+V costs:

```text
2 * 8 KV heads * 128 head_dim * 2 bytes = 4096 bytes per cached token
```

Approximate per-request context KV memory before TP sharding is therefore:

| Checkpoint | Sliding portion | Full-attention portion | At 40K context | At 262144 context |
|---|---:|---:|---:|---:|
| 27B DFlash | `4 * 2048 * 4 KiB = 32 MiB` | `L * 4 KiB` | ~188 MiB | ~1056 MiB |
| 35B-A3B DFlash | `5 * 4096 * 4 KiB = 80 MiB` | `L * 4 KiB` | ~236 MiB | ~1104 MiB |

The planner must expose this cost per request and concurrency. DFlash cannot
quietly allocate full-capacity caches for every configured request when the
memory plan only fits active requests. Paged/on-demand full-layer storage and
compact rotating sliding storage are required before high-concurrency
promotion.

### Prefix cache integration

The existing fixed `mtp_*` fields in `PrefixPayloadLayout` are insufficient for
mixed full/sliding DFlash layers. Generalize speculative sidecar payloads to a
named, versioned descriptor with per-layer layout, logical token count,
precision, KV head shard, window, and byte extent. MTP remains one sidecar type;
DFlash becomes another.

Add:

- `dflash.bin` storage for disk/tiered backends;
- DFlash checkpoint/config/precision fields in `PrefixCacheFingerprint`;
- per-layer DFlash cached counts in state snapshots and probes;
- DFlash bytes, hits, misses, incompatibilities, and policy actions in stats;
- atomic target + DFlash snapshot ordering on one completion event.

Policy behavior:

- `require-sidecar` treats a legacy or incompatible prefix entry as a prefix
  miss, reruns target prefill, and regenerates DFlash context K/V.
- `bypass-request` may restore the target prefix but disables DFlash for that
  request, with a reason and counter.
- A partial DFlash payload is never restored.

Storing raw target features is not required. A compatible prefix payload stores
the already-projected DFlash context K/V. This avoids replaying the target just
to rebuild draft context.

### Qwen 3.6 target hybrid state

DFlash does not change Qwen 3.6 target state semantics. Target verification
continues to use the existing speculative KV/GDN/short-conv slots. Publication
commits verifier input rows `0..a`; rejected rows and the ready target token are
not published.

This is preferable to snapshot/rollback approaches seen in some external
implementations. The existing accepted-only slot machinery is both safer and
faster. DFlash target feature taps are scratch outputs of the same verifier
rows; the accepted prefix is projected into DFlash K/V only after the device
sampler has produced the compact outcome.

### Streams and graph capture

Use stable addresses and explicit events:

1. Target stream completes prefill/verifier plus selected feature taps.
2. DFlash stream waits, runs feature fusion/context K/V publication, prepares
   the block, runs the query graph, shared LM head, and draft argmax.
3. Target stream waits for proposal readiness, runs the grouped verifier.
4. Sampler/publication stream reduces accepted outcomes and publishes target
   state.
5. DFlash stream waits for the accepted outcome and appends accepted feature
   K/V for the next cycle.

After warmup, fixed block buckets `{4, 8, 16}` use captured query graphs and
captured target verifier graphs. Context-KV publication may be a captured
fixed-capacity graph with device row counts or a graph-capturable kernel chain;
it must not force a host sync.

Every `DFlashMatmulPlan` is finalized before the first capture. Its cuBLASLt or
backend-native descriptors, selected algorithm id, packed-weight owner, and
maximum workspace slice have the same lifetime as the captured graph. Eager and
captured modes are separate dispatch features because launch overhead and
library algorithm economics can differ, but both consume the same logical
buffers and produce the same visible rows. A capture miss may rebuild outside
the hot transaction; it may not allocate or run an algorithm heuristic inside
an active decode step.

The graph cache key includes target/draft model fingerprints, backend/device,
request bucket, `B`, target verifier bucket, layer-causality vector, sliding
window, proposal activation/weight/accumulator precision, matrix dispatch and
packed-layout ids, KV precision/layout, TP shard, and prefix state shape.

### Multi-device lowering

SingleDevice comes first, but the data ownership model must not block later
topologies.

LocalTP:

- target post-layer hidden is replicated after the normal row-parallel
  reductions, so each child can tap the same rows;
- shard DFlash QKV/gate/up column-wise and Wo/down row-wise using the existing
  Qwen3 policies;
- shard the feature FC on its input dimension and allreduce its draft-hidden
  output, or retain a measured replicated-FC option;
- bind each child's target embedding and LM-head shard; use the existing
  mirrored-full-head option only when it wins and preserves sampler ownership;
- DFlash KV caches record local KV heads and publish the same accepted count on
  every participant.

PP/NodeLocalTP:

- each target stage taps only locally owned selected layers;
- selected feature slices move through explicit graph PP transfers/collectives
  to the DFlash proposer domain;
- no rank hosts a nested remote-device DFlash graph;
- the continuation owner coordinates the proposer-neutral transaction, while
  every participant publishes its local accepted target state symmetrically.

ExpertOverlay:

- the DFlash drafter remains dense and does not enter expert routing;
- target MoE routing and expert state continue through the normal target
  verifier graph;
- feature taps occur after the complete routed/shared FFN residual, never from
  a partial expert result.

## Implementation Phases

### Phase 0: Freeze sources, artifacts, and the oracle

Tasks:

- Pin the DFlash paper revision, Z-Lab repository commit, the two HF revisions
  listed above, and reviewed vLLM/SGLang/llama.cpp source commits in an evidence
  manifest.
- Mirror the two original safetensor checkpoints in the model corpus.
- Add a Python reference runner that dumps target taps, concatenated/fused
  features, per-layer context K/V, draft layer outputs, draft logits/tokens,
  target verifier logits, acceptance length, emitted tokens, and post-commit
  cache counts.
- Generate deterministic fixtures for prompt prefill, `B={2,4,8,16}`, every
  rejection position, all-accepted, EOS, and long-context window boundaries.
- Record checkpoint tensor inventories and hashes.

Exit gate:

- The reference reproduces Z-Lab generation for both Qwen 3.6 checkpoints and
  exposes every tensor boundary Llaminar will test.

### Phase 1: Extract the common speculative transaction

Tasks:

- Introduce proposer-neutral metadata, verifier, sampler, and transaction names
  with MTP compatibility aliases.
- Add `ISpeculativeProposer` and adapt MTP without changing its execution.
- Separate shared verifier configuration from proposer-specific MTP depth
  configuration.
- Add `DFlashRuntimeConfig`, CLI/YAML parsing, validation, dry-run reporting,
  and mutual exclusion.
- Preserve every current MTP counter or provide an explicit compatibility map.

Exit gate:

- Full MTP unit, grouped-verifier, Qwen 3.6 prefix/MTP, and benchmark smoke gates
  are unchanged. No new DFlash execution is enabled yet.

### Phase 2: Convert, load, and plan DFlash checkpoints

Tasks:

- Implement the HF-to-GGUF converter and conversion tests for both config
  nesting variants.
- Register the `dflash` graph/schema factory.
- Implement draft model context, weight manifest, target-shared bindings, and
  compatibility validation.
- Extend `ModelMemoryProfile`, placement explanation, and weight streaming to a
  target + draft composite.
- Load canonical BF16 first and distinguish checkpoint precision from effective
  backend storage/compute precision in configuration, fingerprints, and memory
  plans.
- Add load-time prepared-weight ownership for later CUDA Tensor Core layouts and
  MI50 BF16-to-FP16 views; no preparation is enabled until the BF16 oracle
  passes. Add quantized draft formats only after that oracle.

Exit gate:

- Both pinned checkpoints load on CPU/CUDA/ROCm, every tensor is accounted for,
  shared weights are counted once, invalid target/draft pairings fail, and
  `--dry-run` predicts weight/scratch/KV bytes without executing inference.

### Phase 3: Target feature taps and feature fusion

Tasks:

- Add graph-declared post-layer residual taps and stable concatenation buffers.
- Wire the selected tap vector into Qwen 3.6 dense and MoE prefill/verifier
  graph construction without changing normal graph math.
- Implement one-GEMM feature FC and context RMSNorm.
- Add the explicit target-FP32 to proposal-BF16/FP16 conversion boundary and
  preserve FP32 accumulation for feature/norm reductions.
- Stream prompt chunks into the fusion graph; do not retain full-prompt raw
  features.
- Add CPU/CUDA/ROCm tap/fusion dump points and counters.

Exit gate:

- Each selected target row, concatenation order, fused feature, padding guard,
  and absolute position matches the Python oracle for dense/MoE prefill and
  verifier shapes `M=1,2,4,8,16`.
- DFlash-disabled target logits and state remain unchanged.

### Phase 4: DFlash context KV and draft graph semantics

Tasks:

- Implement `DFlashContextKVCacheSet` with full and rotating sliding layers.
- Implement context K/V projection, K norm, RoPE, and append.
- Extend attention with context-plus-inline K/V.
- Build the two-template DFlash graph and borrowed embedding/LM head.
- Introduce `DFlashMatmulPlan` and descriptor bundles for feature FC, context
  K/V, QKV, output, gate/up, down, and LM-head execution. Start with ordinary
  backend GEMMs behind this interface before tuning physical candidates.
- Keep proposal activations/KV resident in their configured 16-bit format with
  FP32 reductions; do not inherit the target verifier's FP32-storage/fixed-order
  projection path.
- Implement fixed device block preparation and batched greedy draft argmax.
- Start with eager CPU as the mathematical debugger, then the same graph/stages
  on CUDA and ROCm.

Exit gate:

- Feature K/V, every draft-layer output, final hidden, logits, and proposal ids
  match the reference within declared BF16/FP32 tolerances.
- Causal SWA and noncausal full layers pass adversarial future-token and window
  boundary tests on all backends.
- Query K/V never changes persistent cache counts.

### Phase 5: Dense SingleDevice greedy transaction

Tasks:

- Connect `DFlashProposer` to the common `B`-row verifier and device outcome
  reduction.
- Publish target state through the existing accepted slots.
- Append exactly `a+1` accepted target-feature rows to DFlash context K/V.
- Cover ready-anchor, correction, bonus, EOS, maximum-output, and clear-cache
  transitions.
- Add captured graphs for `B={4,8,16}` after eager parity.

Exit gate:

- Qwen3.6-27B DFlash greedy output is token-identical to ordinary target decode
  on CPU/CUDA/ROCm for short, long, math, code, chat, all rejection positions,
  and repeated clear-cache cases.
- Target KV/GDN/short-conv/terminal state and DFlash K/V counts match serial
  reference after every cycle.
- Captured and eager paths match; GPU hot decode has no decision D2H sync.

### Phase 6: MoE and stochastic losslessness

Tasks:

- Enable the same proposer for Qwen3.6-35B-A3B.
- Reuse device-resident one-hot-draft rejection sampling for non-greedy target
  sampling.
- Preserve request/batch-invariant RNG addressing and target sampler history.
- Verify accepted-only GDN/short-conv publication for block 16.
- Exercise routed/shared expert verifier rows through the existing grouped
  serial-equivalence gate.

Exit gate:

- Dense and MoE greedy sequences match their baseline targets.
- Seeded stochastic runs satisfy the existing sampler replay contract; broad
  distribution tests pass token-frequency/KL thresholds against ordinary
  target sampling.
- Rejected target and draft suffixes leave byte-identical live state.

### Phase 7: Prefix cache and long context

Tasks:

- Generalize prefix speculative sidecar descriptors and storage.
- Export/import DFlash full/sliding K/V with fingerprints and atomic events.
- Test full, partial, split-prefill, disk, RAM, device-hot, tiered, legacy, and
  incompatible prefix cases.
- Certify sliding-window transitions at `W-1`, `W`, `W+1`, multiple rotations,
  and long full-attention cache growth.
- Treat Qwen3.6-35B-A3B up to 40K as the initial performance-promotion range;
  measure, but do not assume, 27B or beyond-40K acceptance.

Exit gate:

- Prefix restore is output/state equivalent to uninterrupted DFlash decode.
- Incompatible entries follow the configured named policy with counters.
- Memory growth follows the planned sliding-plus-full formula and has no raw
  target-feature leak.

### Phase 8: Production GPU path and request batching

Tasks:

- Pack/fuse QKV, gate/up, and all-layer context K/V bundles and use split output
  views without intermediate copies.
- Implement preplanned CUDA cuBLASLt BF16/FP16 Tensor Core candidates with FP32
  accumulation for every exact dense/MoE shape and `M` bucket. Add custom
  CUTLASS/MMA candidates only where exact-shape evidence beats cuBLASLt.
- Extend CUDA WMMA attention to mixed causal/full `CachePlusInlineKV`; prove
  production-selected HMMA/IMMA routes with isolated Nsight instruction
  evidence.
- Implement MI50 FP16 packed-dot/VALU matrix candidates and pure-VALU
  `CachePlusInlineKV`, including load-time BF16-to-FP16 prepared weights behind
  an acceptance gate.
- Compare compact accepted-row context materialization with padded fixed-`B`
  Tensor Core execution and record useful/padded rows.
- Add natural/padded DFlash LM-head and draft projection shapes to the shared
  floating/NativeVNNI measurement inventory; train generated dispatch rather
  than hardcoding backend/shape choices.
- Capture block prep, draft graph, shared LM head, argmax, target verifier,
  sampling, and publication where backend APIs permit.
- Flatten active requests into fixed request/block buckets with device row
  starts and per-request caches.
- Add compact sliding caches and paged/on-demand full-layer caches.
- Overlap accepted-context K/V materialization with response handling without
  crossing the next proposal dependency.
- Remove eager migration fallbacks once captured paths pass parity/perf gates.

Exit gate:

- CUDA and ROCm profiles show no hot allocations, default-stream launches,
  runtime library heuristic searches, per-step weight packing, per-step host
  outcomes, or whole-`B` accepted-state copies.
- Request batches with unequal prompt lengths, acceptance lengths, EOS, and
  prefix states are batch-invariant.
- Counters prove the selected matrix route, fused bundles, useful/padded row
  counts, prepared-weight ownership, and captured query/verifier reuse.
- RTX 3090 evidence proves actual HMMA/IMMA instructions for every promoted
  Tensor Core family. MI50 evidence proves the intended packed-dot/VALU route;
  neither backend is promoted from a source-level kernel name.
- Real-model accepted/committed-token throughput improves beyond the noise band;
  an isolated matrix-kernel win alone does not pass the phase.

### Phase 9: Block policy and performance promotion

Tasks:

- Add fixed, observe, and dynamic block controllers over `{4,8,16}`.
- Record first-rejection histograms, accepted drafts, committed tokens,
  draft/verify/publication time, matrix route, useful/padded rows, context,
  request batch, and memory pressure.
- Train deterministic offline tables per backend, hardware profile, target
  dense/MoE class, target/draft precision, matrix dispatch family, context
  bucket, and concurrency.
- Allow backend-specific economics: CUDA Ampere may prefer `B=16` to feed MMA
  tiles, while MI50 may prefer `B=4`, `B=8`, `B=16`, or bypass. Shared semantics
  do not imply a shared block decision.
- Compare dynamic DFlash with the best fixed DFlash block, existing MTP policy,
  and ordinary decode on held-out evidence.
- Permit block zero as an explicit dynamic bypass state.

Exit gate:

- Generated policy is reproducible from checked-in evidence and cannot select
  an uncertified block/backend/precision lane.
- Default eligibility meets the promotion gates below on held-out runs.

### Phase 10: Multi-device promotion

Order:

1. dense LocalTP;
2. MoE LocalTP;
3. NodeLocalTP;
4. LocalPP/global PP;
5. ExpertOverlay and named heterogeneous domains.

Each topology repeats target feature ownership, target/draft weight sharding,
grouped verifier, stochastic sampling, accepted publication, prefix restore,
graph capture, and performance gates. A topology does not inherit promotion
from SingleDevice merely because its target baseline already works.

Exit gate:

- Per-device graphs remain symmetric, collectives/transfers are graph-native,
  all participants agree on accepted counts and positions, and no hidden host
  coordinator moves feature tensors.

### Phase 11: Default enablement and cleanup

Tasks:

- Document checkpoint conversion, configuration, memory planning, profiling,
  and operational counters.
- Install generated block policies only for certified hardware/model profiles.
- Keep DFlash opt-in elsewhere and explain the bypass reason.
- Remove compatibility aliases/dead fallbacks only after MTP and DFlash callers
  use the common transaction directly.
- Add DFlash lanes to release smoke, nightly parity, and performance dashboards.

Exit gate:

- The complete Definition of Done is satisfied.

## Verification Plan

### Unit tests

Add at least:

- `Test__DFlashCheckpointManifest.cpp`
- `Test__DFlashModelCompatibility.cpp`
- `Test__DFlashRuntimeConfig.cpp`
- `Test__DFlashTargetFeatureTap.cpp`
- `Test__DFlashFeatureFusion.cpp`
- `Test__DFlashMatmulPlan.cpp`
- `Test__DFlashProposalPrecision.cpp`
- `Test__DFlashPackedWeightLifetime.cpp`
- `Test__DFlashAttentionMask.cpp`
- `Test__DFlashContextKVCache.cpp`
- `Test__DFlashBlockLayout.cpp`
- `Test__DFlashAcceptedContextPublication.cpp`
- `Test__DFlashPrefixPayloadLayout.cpp`
- `Test__DFlashMemoryProfile.cpp`
- `Test__DFlashGraphConstruction.cpp`

Required semantic cells include:

- dense and MoE checkpoint manifests;
- `B={2,4,8,16}` and invalid `B>trained`;
- rejection `a={0,1,...,d}`;
- EOS at every emitted row;
- zero/one/full context and prefill bucket padding;
- sliding boundaries and rotations;
- full-layer noncausal future-mask influence;
- no future influence in sliding causal layers;
- target layer extraction `+1` mapping;
- shared target weight lifetime and missing-weight failure;
- exact dense/MoE feature-FC, stacked-K/V, QKV, output, gate/up, down,
  and LM-head descriptor shapes;
- natural rows `{3,7,15}`, padded rows `{4,8,16}`, accepted `M=1..16`,
  request-batched multiples, poisoned padding, and split-output views;
- canonical BF16, CUDA BF16/FP16, and MI50 FP16 proposal storage with FP32
  accumulation/reduction boundaries;
- eager/captured dispatch ids, explicit non-default streams, declared workspace,
  and no runtime descriptor/packed-weight ownership changes;
- `portable`/`accelerated`/`auto` admission, including hard failure when a
  requested accelerated route lacks certification and a counted auto bypass;
- proof that proposal matrix routes cannot enter verifier decode-equivalent
  scope or mutate publishable target state;
- prefix fingerprint changes for draft revision, precision, block policy, target
  tokenizer, and topology.

### Reference and real-model integration tests

Create `tests/v2/integration/parity/dflash/` with:

- `Test__Qwen36_DFlashReferenceEquivalence.cpp`
- `Test__Qwen36_DFlashSingleDeviceParity.cpp`
- `Test__Qwen36MoE_DFlashSingleDeviceParity.cpp`
- `Test__Qwen36_DFlashGroupedVerifierEquivalence.cpp`
- `Test__Qwen36_DFlashPrefixParity.cpp`
- `Test__Qwen36_DFlashLongContextParity.cpp`
- `Test__Qwen36_DFlashGraphCapture.cpp`
- `Test__Qwen36_DFlashMatrixRouteParity.cpp`
- topology-specific files as phases are promoted.

The operation-equivalence test should compare:

1. every target tap;
2. concatenated and fused target features;
3. each layer's context K/V before/after K norm and RoPE;
4. every draft layer hidden state;
5. natural versus padded matrix-bundle visible rows, DFlash final hidden,
   shared-head logits, and proposals;
6. target grouped verifier logits versus serial target rows;
7. accepted counts and emitted tokens;
8. target KV/GDN/short-conv/terminal state;
9. DFlash context K/V and logical counts.

Draft logits may use a declared numerical tolerance because different valid
proposals do not violate target losslessness. Target grouped verifier math and
publishable continuation state retain the existing serial-equivalence gates.
Greedy end-to-end output must be token-identical to the configured target
baseline.

CUDA route tests use counters plus an isolated Nsight Compute launch to prove
that the production-selected floating/quantized candidates execute HMMA/IMMA.
Profiler replay duration is never a dispatch timing label. ROCm route tests
prove the selected gfx906 packed-dot/VALU family and reject MFMA/rocWMMA claims.
Canonical candidate timing runs on explicit streams outside profiler replay,
with eager and captured execution measured independently. Both routes must then
pass the real-model benchmark; microkernel speed is not a promotion result.

### Quantization matrix

Start with canonical BF16 DFlash weights/KV to prove the architecture. CUDA then
certifies BF16 and FP16 Tensor Core proposal islands independently. ROCm MI50
certifies canonical BF16-through-FP32 against a load-time BF16-to-FP16 prepared
weight/activation path using packed-dot/VALU kernels; the latter may become the
production route only when acceptance-adjusted throughput and memory improve.
Keep FP32 reduction/accumulator boundaries explicit in every case.

Then certify draft Q8_0 and selected Q4/IQ formats using the normal all-format
evidence process. Add the DFlash exact shapes to the shared cross-backend shape
inventory and generated-policy pipeline; do not create a DFlash-only hardcoded
NativeVNNI table. The authoritative corpus, generic-fallback, exact-overlay,
profiler-isolation, and installation rules remain those in the
[`Cross-Backend Batch-Invariant NativeVNNI Learned Dispatch Policy`](NATIVE_VNNI_BATCH_INVARIANT_LEARNED_DISPATCH_POLICY.md).
Keep feature FC, context norm, Q/K norms, and cache precision floating until
evidence proves lower precision is acceptance-positive.

Quantized draft differences are allowed to change proposal/acceptance patterns,
but not target output distribution or accepted-state invariants. Every promoted
format needs its own acceptance, latency, and memory evidence; functional parity
alone does not make it a default. None of these proposer precision choices
changes target verifier format or its grouped serial-byte gate.

### Build and test commands

Use the required Ninja/full-parallelism workflow:

```bash
cmake -B build_v2_integration -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Integration
cmake --build build_v2_integration --parallel

ctest --test-dir build_v2_integration \
  -R '^V2_Unit_(DFlash|Speculative)' \
  --output-on-failure --parallel

ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen36.*DFlash' \
  --output-on-failure

ctest --test-dir build_v2_integration \
  -R '^V2_Integration_GroupedVerifierRows_' \
  --output-on-failure

cmake --build build_v2_release --parallel \
  --target v2_perf_cuda_dflash_matmul v2_perf_rocm_dflash_matmul

ctest --test-dir build_v2_release \
  -R '^V2_Perf_(CUDA|ROCm)_DFlashMatmul$' \
  --verbose
```

Performance tests use `build_v2_release`; parity tests do not use Release. The
isolated matrix tests establish candidate correctness and timing only. Final
route promotion still requires captured and eager real-model DFlash benchmarks
with profiling disabled.

## Benchmark and Promotion Plan

### Required comparison axes

Every performance claim names:

- target: Qwen3.6-27B dense or Qwen3.6-35B-A3B;
- backend/device and topology;
- target quantization and KV precision;
- DFlash revision, canonical checkpoint precision, effective activation/weight/
  accumulator precision, packed layout, matrix dispatch family, and KV
  precision;
- proposer: baseline, MTP fixed/dynamic, DFlash fixed/dynamic;
- `B={4,8,16}` and `d={3,7,15}`;
- natural, padded, compact-accepted, and request-batched matrix `M` where the
  route supports alternatives;
- greedy and recommended stochastic sampling;
- prompt class: math, code, chat, long-context;
- prompt length buckets: short, 2K, 16K, 40K, and supported longer probes;
- output length: 128 and long decode;
- request concurrency/batch: 1, 8, and 32 where memory permits;
- prefix cold, full hit, and partial hit.

### Required metrics

Record in structured JSON/CSV:

- TTFT, decode tokens/s, median/p95 TPOT, and end-to-end latency;
- target feature tap/copy time;
- feature FC/norm time;
- context K/V projection/norm/RoPE/write time;
- DFlash query graph and shared LM-head time;
- time and route id for feature FC, stacked context K/V, QKV, output, gate/up,
  down, attention, and LM head;
- useful rows, padded rows, useful FLOPs, arithmetic intensity, useful prepared-
  weight bandwidth, and physical profiler traffic per matrix bundle;
- CUDA HMMA/IMMA instruction proof and MI50 packed-dot/VALU route proof from
  separate isolated profiler launches;
- target verifier time and verified rows;
- sampler/publication time;
- accepted draft count histogram and committed tokens/cycle;
- zero-accept and full-accept rates;
- target rows and draft FLOPs per committed token;
- graph captures, replays, cache misses, and eager fallbacks;
- host sync and transfer counts;
- target/DFlash weight, KV, prefix, and peak workspace bytes;
- canonical versus prepared/packed weight bytes and one-time preparation cost;
- prefix sidecar hits/misses/bypass reasons;
- dynamic block decisions and reasons.

### Promotion tiers

Correctness is mandatory at every tier.

Functional:

- parity/state/prefix gates pass;
- DFlash remains opt-in even if speed-negative.

Accelerated for a hardware profile:

- the dynamic or best fixed policy achieves at least 1.10x geometric-mean
  decode throughput over ordinary decode on the required concurrency-one
  workload set with confidence intervals excluding 1.0;
- any cell below 0.95x is predictably bypassed by the installed policy;
- TTFT and peak-memory regressions stay inside the profile's declared budget;
- no hot-path allocation, packing, library heuristic search, or outcome D2H
  sync is present;
- every claimed CUDA Tensor Core route has production-selection and HMMA/IMMA
  instruction evidence, and every claimed MI50 optimized route has packed-dot/
  VALU evidence;
- the gain survives the full model with profiling disabled and exceeds its
  measured noise band; isolated GEMM latency or TFLOPS is insufficient.

Default-eligible:

- the installed DFlash policy is at least 1.05x the best of ordinary decode and
  the installed MTP policy on held-out geometric-mean throughput;
- held-out p95 TPOT is no worse than 1.02x the best alternative;
- no required workload/backend/topology silently falls into an uncertified
  implementation;
- prefix and high-concurrency memory behavior remain within planned budgets.

Upstream speedups are never substituted for these gates.

## Risks and Mitigations

| Risk | Consequence | Mitigation / required evidence |
|---|---|---|
| Target tap off by one layer | Low acceptance with plausible tensors | Name `PostLayerResidual`; fixture every selected tap; test HF-to-GGUF `+1` mapping |
| Treating all draft layers as noncausal | Wrong Qwen 3.6 proposals and backend incompatibility | Per-layer named templates; hostile future-token tests |
| Query K/V accidentally committed | Cache drift and growing stale state | `CachePlusInlineKV`; cache-count assertions after every proposal |
| Rejected target feature rows appended | Future drafts depend on rejected tokens | Device prefix-write lengths from accepted outcomes; poison rejected rows |
| GDN/short-conv rollback error | Target output divergence | Reuse current speculative slots; never snapshot/replay as the primary path |
| Shared embedding/head lifetime bug | Use-after-free or duplicate VRAM | Borrowed typed bindings and composite memory/lifetime tests |
| Dense 27B draft weight cost | OOM or poor placement | 3.22 GiB visible in dry-run; streaming; later certified quantization |
| Full draft layer KV at long context/concurrency | Multi-GiB request state | Paged full cache, compact SWA, concurrency admission, explicit policy bypass |
| Long-context acceptance collapse | DFlash slower while still correct | Context-aware observe/dynamic policy; 35B promotion initially bounded to trained 40K evidence |
| Draft quantization reduces acceptance | Loss of speed despite smaller memory | Per-format end-to-end evidence, not kernel speed alone |
| Feature FC optimization changes arithmetic | Acceptance regression | One-GEMM oracle first; A/B logits/acceptance gates before decomposition |
| Treating Tensor Cores as the whole speed model | Small-`M` draft remains weight-bandwidth bound or regresses end to end | Report arithmetic intensity/useful bandwidth and require real-model committed-token gains beyond noise |
| Reusing verifier FP32x16 kernels for the proposer | BF16/FP16 checkpoint never reaches matrix hardware and incurs FP32 storage traffic | Separate proposal precision island and target conversion boundary; retain verifier scope only for target rows |
| Hot cuBLASLt planning or workspace allocation | Capture failure, jitter, and decode synchronization | Finalize descriptors/algorithms/workspace before capture; stable graph-owned plan lifetime |
| Padding to `M=16` computes too much rejected work | Tensor Core utilization improves while cycle latency worsens | Measure compact versus padded paths; count useful/padded rows; dispatch by acceptance/request bucket |
| MI50 executes BF16 through FP32 emulation | ROCm DFlash is needlessly slow and memory-heavy | Preserve BF16 oracle, then certify load-time FP16 prepared weights and packed-dot/VALU execution |
| CUDA tile geometry leaks into ROCm graph semantics | Poor wave64 execution or backend divergence | Common logical bundles; backend-specific physical lowering and block policy; no ROCm MFMA claim on gfx906 |
| Tensor Core route changes publishable target math | Continuation state loses serial equivalence | Proposal-only route boundary; target verifier retains `beginVerifierDecodeEquivalentScope()` and all-format byte gates |
| Packed weight duplication is omitted from planning | OOM despite a passing checkpoint-size estimate | Composite dry-run accounts canonical and every retained prepared view; lifetime/clear tests |
| Dynamic blocks trigger graph churn | Capture overhead dominates | Fixed bucket set `{4,8,16}` and persistent buffers |
| TP/PP hidden ownership ambiguity | Cross-rank mismatch/deadlock | SingleDevice first; explicit per-device tap/collective plan and participant agreement tests |
| Prefix entry lacks DFlash state | Invalid context or permanent bypass surprise | Named `require-sidecar`/`bypass-request` policies and counters |
| Upstream checkpoint/config changes | Reproducibility loss | Pin revisions and tensor hashes; reject unknown manifests until re-certified |

## Expected File Map

New core areas:

```text
src/v2/execution/speculative/
  ISpeculativeProposer.h
  SpeculativeDecodeMetadata.*
  SpeculativeDecodeTransactionDriver.*
  SpeculativeRejectionSampler.*

src/v2/execution/dflash/
  DFlashRuntimeState.*
  DFlashWeightManifest.*
  DFlashModelCompatibility.*
  DFlashTargetFeatureCollector.*
  DFlashFeatureFusionExecutor.*
  DFlashMatmulPlan.*
  DFlashPackedWeightOwner.*
  DFlashContextKVCache.*
  DFlashProposer.*
  DFlashBlockController.*

src/v2/models/dflash/
  DFlashGraphConfigBuilder.*
  DFlashSchema.h
  DFlashGraph.*

src/v2/execution/compute_stages/stages/
  LayerOutputTapStage.*
  DFlashContextKVProjectionStage.*

src/v2/kernels/cuda/gemm/
  CUDADFlashMatmulPlan.*
  CUDADFlashMatmulDispatchGenerated.inc

src/v2/kernels/rocm/gemm/
  ROCmDFlashMatmulPlan.*
  ROCmDFlashMatmulDispatchGenerated.inc

scripts/
  convert_dflash_hf_to_gguf.py

python/reference/
  dflash_reference.py

tests/v2/unit/dflash/
tests/v2/integration/parity/dflash/
tests/v2/performance/dflash/
tests/v2/performance/kernels/cuda/gemm/Perf__CUDADFlashMatmul.cpp
tests/v2/performance/kernels/rocm/gemm/Perf__ROCmDFlashMatmul.cpp
```

Existing areas expected to change:

- `RuntimeConfig`, `OrchestrationConfig`, parser, validator, YAML, and dry-run;
- `GraphSchema`, `GraphResolver`, compute-stage factory, attention source mode;
- `ModelRegistrations`, CMake source/test registration;
- `DeviceGraphOrchestrator` and rank/runner speculative scheduling;
- Qwen graph construction only for generic feature-tap insertion;
- prefix layout/fingerprint/storage/snapshot/probe/stats;
- memory planning, benchmark JSON, PerfStats, and generated policy plumbing.

## Definition of Done

DFlash for Qwen 3.6 is complete only when all of the following are true:

- Both pinned Qwen 3.6 DFlash checkpoints convert, load, validate, and run.
- The target embedding and LM head are shared, not duplicated.
- Target feature taps, feature fusion, per-layer context K/V, mixed attention,
  proposals, verifier, and accepted state match the reference contracts.
- Greedy output is token-identical to ordinary target decode on CPU/CUDA/ROCm
  for dense and MoE.
- Stochastic output preserves the target distribution and existing RNG contract.
- Rejected suffix and bonus-ready rows cannot mutate target or DFlash live state.
- Prefix full/partial/split restores are state-equivalent, fingerprinted, and
  memory-accounted.
- Block 16 uses the existing serial-equivalent 16-row target verifier.
- GPU hot paths have stable buffers, explicit streams, captured buckets, no hot
  allocations, and no per-step outcome D2H sync.
- The DFlash proposal graph is a certified BF16/FP16 activation island with
  FP32 accumulation/reduction boundaries; it does not inherit the target
  verifier's FP32-storage projection route.
- RTX 3090 production dispatch uses preplanned, graph-safe Tensor Core routes
  where real-model evidence selects them, with HMMA/IMMA instruction proof.
- MI50 production dispatch uses certified FP16 packed-dot/VALU routes where
  selected, never claims unavailable MFMA hardware, and may choose a different
  block size or explicit bypass from CUDA.
- Feature FC, stacked context K/V, QKV, gate/up, down/output, attention, and
  LM-head matrix bundles cover natural, padded, accepted-row, and request-batch
  shapes through generated backend dispatch with no hot packing/planning.
- Target verifier and publishable target continuation state still pass the full
  grouped serial-byte-equivalence inventory for every backend/format.
- Structured benchmark evidence compares baseline, MTP, and DFlash across the
  required matrix.
- Default enablement occurs only for profiles that pass held-out performance,
  latency, and memory gates; all other profiles remain explicitly opt-in or
  bypassed.
- Multi-device promotion preserves per-device symmetric graphs and explicit
  collectives.
- Documentation, release tests, observability, and generated policy artifacts
  are checked in and reproducible.

## Primary Sources

- [DFlash paper (arXiv 2602.06036v2)](https://arxiv.org/html/2602.06036)
- [Z-Lab DFlash reference repository](https://github.com/z-lab/dflash)
- [Qwen3.6-27B-DFlash checkpoint](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash)
- [Qwen3.6-35B-A3B-DFlash checkpoint and benchmarks](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash)
- [vLLM DFlash proposer](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/dflash.py)
- [vLLM Qwen3 DFlash model](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_dflash.py)
- [SGLang DFlash model](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/dflash.py)
- [SGLang DFlash V2 worker](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/dflash_worker_v2.py)
- [llama.cpp DFlash model](https://github.com/ggml-org/llama.cpp/blob/master/src/models/dflash.cpp)
- [llama.cpp DFlash implementation PR](https://github.com/ggml-org/llama.cpp/pull/22105)
- [NVIDIA Ampere Tensor Core tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
- [AMD rocWMMA supported architectures](https://rocm.docs.amd.com/projects/rocWMMA/en/docs-6.3.1/api-reference/api-reference-guide.html)
- [AMD/LLVM gfx906 instruction reference](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX906.html)
