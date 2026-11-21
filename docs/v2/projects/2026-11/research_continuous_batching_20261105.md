# Continuous batching in Llaminar: architecture, reference engines, and delivery gates

**Date:** 2026-11-05

**Depth:** Exhaustive; seven research rounds including citation/provenance audits, primary-source inspection, and three session consultations.

**Status:** Research and proposed implementation sequence. No runtime implementation, build, benchmark, deployment, or certification was performed. This is a detailed research record, not a size-limited tuning dashboard.

**Companion design:** [Continuous batching: proposed serving design](design_continuous_batching_20261105.md) records the revised architecture. The [independent review](review_continuous_batching_20261105.md) found material cohort, sampler, memory, publication and lifecycle gaps; its CB-01 through CB-20 dispositions distinguish document corrections from unproved runtime gates.

## Executive summary

**Llaminar should build continuous serving by extending its existing request-batched execution and ownership contracts—not by adding HTTP inference workers or treating a larger batch setting as a scheduler.** At the inspected revision, HTTP requests execute serially on one stable inference worker. There is substantial fixed-cohort MTP batching below that boundary, including request coordinates, grouped verification, per-request outcomes, and device-resident execution. It does not establish online admission, replacement of completed lanes, mixed prefill/decode scheduling, or independent HTTP request lifetimes. [L1–L6]

The most useful reference patterns are vLLM's token-work budget and block ownership, TensorFold's explicit admit/fill/step/retire loop and family-specific cache adapters, and Sarathi-Serve's treatment of long-prefill interference. **Do not import vLLM's host-owned token scheduler or graph downgrades into Llaminar:** its GPU controller must remain device-owned, complete captured execution must remain mandatory, and `PhysicalMemoryAuthority` must remain the sole physical-memory authority. [L8, L10, L13, V1–V6, T2–T5, P3]

An existing **16-token SSE publication window** is a useful design seam: it completes a retained production transaction while preserving resident continuation. It is not currently a cross-request scheduling boundary. A continuous-serving design must first prove slot-local reset, admission, cancellation, output publication, capture identity, and workspace safety at such a boundary. The window size is not an established optimum or a proposal to expose mutable GPU state to a host scheduler. [L2–L4, L10]

The peer evidence points to two practical priorities alongside batching: preserve exact rendered prompt prefixes, and measure **SLO-qualified goodput under long-prefill interference**, not just peak decode tokens/s. The TensorFold prefix improvement was principally a serialization repair. The vLLM memory improvements bought capacity and faster cold prefill, not an observed aggregate-decode speedup. Neither provides a measured speedup prediction for Llaminar. [E1, E2]

### Principal findings

| Finding | Confidence and limit |
|---|---|
| HTTP inference is serialized; existing MTP batching is not continuous HTTP serving. | High: directly traced in local source. No new runtime test. |
| Current batched prefill rejects enabled prefix caching, PP/global TP, and multi-rank MPI; it requires MTP and admitted batch capacity. | High: explicit source checks. Rank-local multi-GPU TP is a distinct case, not equivalent to MPI TP. |
| KV/prefix/recurrent state, sampling, output, and reset need request-scoped lifetimes before lanes can be replaced. | High for the missing public contract; implementation feasibility remains to be proved. |
| vLLM and native TensorFold offer useful but different scheduling and memory patterns. | High at the pinned revisions; not a universal statement about every backend/model. |
| Prefill chunking alone does not establish decode latency isolation. | Strong systems rationale and original research; one corroborating private operational incident, not a controlled causal measurement. |
| A captured service-window design is a plausible Llaminar-native direction. | Proposal, not implemented or certified. No numerical performance forecast. |

## 1. Scope, method, and provenance

### 1.1 Research boundary

The initial task investigated batched serving while a separate session completed ARM work and was restricted to this report. The user subsequently requested a written design and a separate worktree. This report and its companion design now live on `feat/continuous-batching-design`, based on `origin/develop` at `ef1f54b0962d63531a9b33e06ca676e5124d05cc`. Only this session's untracked report was moved out of the ARM checkout after byte verification; unrelated CPU-topology/CMake work was not modified. No accelerator occupancy, model campaign, benchmark, service change, package installation, clock/swap change, or protected Docker service operation was requested or performed.

Source inspection is not a capability certificate. The ARM owner's historical Spark **125/125** and current Mac **151/151** results are focused scopes; they do not establish complete Unit/preflight, whole-engine inference, or model qualification. PR32 remained draft at consultation. GB10 CPU topology must come from actual kernel sibling sets: its 20 cores must not be collapsed into ten pairs by misleading `lscpu` CSV data. Shared UMA capacity must still have one PMA authority. [E3]

### 1.2 Exact identities

| Object | Inspected or reported identity | Interpretation |
|---|---|---|
| Llaminar local source | `5e2a881aa99a9a4b428110a64b8ff64694c8b6b8` | Inspected unmerged descendant, 100 commits ahead of the develop base, on `fork/codex/qfn-ngram-continuation-arm64`; not a released continuous-serving implementation. Source links below pin this revision, excluding unrelated dirty edits. |
| Current native TensorFold | `ashhart/TensorFold` at `09e2f7fa664016ec9647af43ec721be33f9fe44a` | Native Zig source and corresponding docs. Do not merge its claims with legacy Python/MLX API prose. |
| Current upstream vLLM | `vllm-project/vllm` at `cb95d994f4ec066b75672035ca3ae7e1c9745e84` | Upstream comparison point, **not** the deployed PCAI source. |
| PCAI vLLM upstream base | `4ba787ec9959e566d3b587a796dff07673324da5` | Pinned by image-source `642dd15` in `enthus-appdev/vllm-pcai`; scheduler/KV source inspected separately. |
| PCAI base image | `vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e` | Public Dockerfile provenance, not proof of subsequent runtime tests. |
| PCAI deployed image | `3f4e61bcc73cc5bf87d73ab69d46db718016b2ff28d9b5cd77e62c87693de978` | Owner-reported image identity for the measurements below. |
| Historical Spark TensorFold experiment | Engine base `2f8e514b0b7d615df7c971627ce3c0fb7e55d93a`; recipe `jayleaton/glm53-tensorfold-spark` at `f79bf9c72968f79bc8f1dc788ddb48ac974d2858` | **78 recipe patches plus local HTTP-contract changes.** The base SHA alone is not the tested artifact or a complete reproduction recipe. [E2] |

<a id="inspected-source-provenance"></a>

**Inspected descendant versus documentation base:** `5e2a881aa99a9a4b428110a64b8ff64694c8b6b8` descends from develop `ef1f54b0962d63531a9b33e06ca676e5124d05cc` by 100 commits and is not merged into that base. Fourteen of the 25 previously linked literal source paths differ between the revisions; the unchanged `AGENTS.md` symlink also resolves to a changed `.agents/AGENTS.md`, making 15 effective file contents differ. This is not merely a two-header relocation.

L1–L16 source links pin the inspected descendant in the fork. L17 is an explicitly develop-pinned follow-up controller/capacity audit from the second review; it does not imply that the two revisions have identical planning policy. `ProjectionRowWork.h` and the standalone `DeviceGenerationExecutionPolicy.h` exist only in that descendant; the execution-policy enum itself already exists on develop inside `IInferenceRunner.h`. No source files are copied into this documentation worktree.

The recorded-input replay requirement and detailed-document/dashboard size distinction appear in the descendant policy but not in the develop policy text. They are explicitly adopted as requirements of this proposed design, not described as already-landed develop rules. The fork commit was verified through GitHub metadata; source inspection still does not certify its runtime behavior.

Official vLLM `stable` documentation was accessed on 2026-11-05; those URLs are moving references. Source-pinned implementation takes precedence over simplified or version-skewed prose. PR bodies, timestamped public author follow-ups, and private operator measurements are separate evidence classes. [V1, V2, V10]

### 1.3 Evidence discipline

- **Source-verified:** implementation/interfaces at the revisions above, or explicit official documentation. This verifies the described code/contract, not its performance on this machine.
- **Owner-reported:** sanitized observations from `pcai-llm-serving` or `dgx-spark-lab`. Not independently rerun here. Private incident evidence has no invented public citation.
- **Proposed:** architecture, implementation stages, and gates derived from the evidence. Proposed names are illustrative, not existing APIs or registered tests.
- **Unmeasured:** all Llaminar continuous-serving performance, the optimal publication quantum, and cross-engine rankings.

Seven initial research rounds informed this draft. The subsequent independent review identified material gaps and overstatements, now recorded with dispositions. A search checkpoint is not an architectural completeness or correctness certificate; no numerical confidence score substitutes for the still-open implementation proofs.

## 2. What does Llaminar already implement?

### 2.1 The HTTP boundary serializes whole inference jobs

`SerializedInferenceExecutor` supplies bounded admission and one stable inference worker. HTTP discovery and passive stats readers remain independent. SSE reserves admission before provider startup, so a client connection is not permission to start another independent GPU execution owner. Shutdown drains admitted work rather than cancelling an arbitrary GPU transaction. `ServerMode` and `ChatCompletionHandler` use this scalar request-serving arrangement. [L1, L2]

The handler's request cleanup, sampling configuration, generation loop, and runner state assume this ownership. Increasing HTTP threads would not create safe batching: it would put independent requests against shared mutable runner state and reusable arenas. A lock around each decode call would still leave prefill/reset/sampler/KV ownership unresolved and would not itself fuse work.

Current stats also reflect this service model: `HttpRuntimeStats` has one active-request/phase representation and publishes terminal observations into an epoch. Its reset/lifecycle guarantees are valuable, but simply adding multiple workers would invalidate the representation. A future multi-request service must preserve passive `GET /stats`, epoch reset isolation, and publication after cleanup while representing several active requests. [L11, L12]

### 2.2 Real request-batched MTP machinery exists below HTTP

Important existing seams include:

- `OrchestrationRunner::supportsPrefillBatch`, `prefillBatch`, `supportsDecodeStepBatch`, and `decodeStepBatch`.
- `admitRequestBatchDeviceResidentGeneration` and `completeDeviceResidentBatchGeneration`.
- `MTPSpecRequestBatchScheduler` and `MTPSpecRequestBatchOwner`: per-call host compatibility and in-flight commit/release helpers. Their integer IDs, string compatibility key and mutation lockout are not the target GPU serving identity/controller.
- `DeviceRequestRowRanges`: immutable request-major physical strides with device-authored live lengths. Short rows and inactive holes must not collapse another request's coordinates.
- Per-row controller outcomes and position-keyed seed machinery. The draw key is `(resolved seed, logical output position, purpose)`, never RequestId/slot/ordinal. Nonzero seeds are preserved and zero-seed entropy is resolved once per lane. However, public batch setup feeds all rows one `active_sampling_params_`; a lower-level seed array does not establish independently supplied per-request laws/seeds in the serving interface. [L3–L6]

The batched decode interface requires a per-row limit and atomic publication or rollback of the admitted batch, but its current admission ABI carries one common budget, admission kind, depth policy and ordinary sampling law. Initialization carries one kind/session epoch; new C cannot simply join continuing A through that cohort ABI. Completion also assumes the admitted cohort. The host batch owner's focused tests reject mutations while a batch is in flight. These are useful contracts/oracles, not a complete continuous controller. [L3–L5, L12]

**Fixed cohort is the accurate description.** These components prove useful foundations; they do not prove that the HTTP server can admit request C while A continues after B finishes.

### 2.3 Current batched-prefill admission is deliberately narrow

At this revision, `supportsPrefillBatch()` requires all of the following:

1. Initialized runner and more than one logical request.
2. MTP enabled and request count within `mtp.max_request_batch`.
3. Prefix caching not enabled under the resolved plan/config storage policy.
4. No local PP, global TP, or pipeline-parallel plan.
5. No MPI world with more than one rank.
6. Enough admitted runner batch capacity. [L3]

Rank-local TP in one process is not excluded by the same topology checks. Focused Qwen3.6 request-batched resident-prefill tests cover single-GPU and local-TP CUDA/ROCm cases in source, with small contexts and explicit MTP settings. Their existence is not a report that they ran here, nor proof of every format, prefix mode, model family, ordinary non-MTP batch, or continuous admission. Their diagnostic instrumentation also does not replace production-path certification. [L14]

Consequently, **do not recommend turning off default prefix caching or MTP merely to label the present API “continuous batching.”** Those would be different explicit modes and would still lack the serving lifecycle.

### 2.4 Reset is an especially important missing serving contract

`IOrchestrationRunner::clearCache()` resets the live request while preserving reusable graphs, prepared weights, workspaces, and device contexts. The lower-level `InferenceStateResetRequest` distinguishes main/MTP KV, GDN/short-conv state, model-local runtime state, logical positions, and terminal metadata. It does not provide the public HTTP scheduler with a typed “retire only slot B while A remains live” contract. [L4]

A continuous service needs that narrower lifecycle across all state owners. `IKVCache::SequenceRetirement` already provides an explicit one-slot KV seam, so a global absence claim would be wrong; it does not by itself retire recurrence, sampler, MTP, publication and model-runtime ownership. Existing graph-builder reset/prefix snapshots include runtime state that must first be classified as slot, transaction or domain state. A prefix hit or sibling retirement must not restore/reset shared placement or histograms. [L4, L7, L9]

### 2.5 Material prerequisites exposed by independent review

| Current source constraint | Consequence for the proposed design |
|---|---|
| One cohort admission kind/epoch/budget and scalar ordinary sampler identity | Mixed NewRequest/ContinueResponse, independent limits and generation-safe slot reuse need a new per-row ABI. |
| Captured sampler embeds law values and a one-row penalty view; law changes can invalidate/recapture | Per-row runtime law/history tables and complete graph identity are prerequisites, not a handler-only change. |
| Batched verifier rejects penalty-mutated sampling while major Qwen defaults include nonzero presence penalties | Default traffic cannot be qualified by disabling penalties or putting it through hidden singleton cohorts. GPU DRY sequence history is also a known gap. |
| Host thinking-budget loop uses a one-token budget and `forceDecodeToken`; host output counts participate in maintenance calls | Admit a device-owned stop/think/forced-sequence machine and preserve device/domain maintenance authority. The 16-token window is not universal today. |
| SSE `sink.write` runs within the inference worker's admitted job | Decouple shared staging and submission from network consumer lifetimes. |
| Fixed-slot full-context KV and copy-based prefix restore | Do not infer paged storage or zero-copy live-prefix sharing from generic KV/prefix interfaces. |
| Single-request MPI command vocabulary, prefix harvest identity and runner-global summaries | Slot-qualified commands, harvest and accounting are prerequisites; broader topologies remain unqualified. |

These statements concern source contracts at the cited revisions; no failing runtime campaign was run here. The review record preserves the initial findings and their second-pass residuals. [L1–L4, L7–L11, L15]

### 2.6 Second-review source and policy corrections

Further source checks at develop `ef1f54b…` exposed three important distinctions:
- Budget-limited MTP transactions skip depth learning, and zero newly emitted tokens are an error for generation commit. Intrinsic window boundaries therefore matter numerically; lifecycle-only work is a different operation, not an empty decode. [L17]
- Terminal prefill logits, ready/pending conditions and prelaunched sidecar ownership are runner-global. Interleaving must retain them per slot, with native completion events and a BOM that includes logits/hidden/sidecar storage. A sampled request-specific condition is not reusable prefix state. [L17]
- Request slots, generation capacity, HTTP accepted-response capacity and benchmark request batches currently use separate inputs. On develop, terminal-head selection also depends on `max_request_batch`. The HTTP constant 32 counts accepted responses including active and queued, not 32 additional queued clients. A typed resolved serving capacity must replace the disconnected authorities, with physical bytes still owned by PMA. [L17]

**Policy proposal, not a source capability:** revision 3 of the design proposes closed typed admission/cancel/window-grant ingress and terminal completions for bounded serving units in both streaming and non-streaming. The user authorized documenting that proposal, **not adopting an exception to current maxims**. Formal adoption remains a prerequisite independent of native tests. HIP's selector-first choice adds a completion before the named model operation; additional native branch tickets also count. The report does not treat these as already permitted or economical.

The selected first increased-capacity shape is one request's canonical prefill chunk alternating with grouped decode. Windows are request-intrinsic, grants are host-owned destination permissions, and selected participation stays device-owned. Host ticket validation uses an immutable host-owned admitted-binding snapshot, not the selected row set. These are proposed contracts, not current features.

## 3. What scheduling and serving contracts are needed?

### 3.1 Distinguish four different meanings of batching

| Term | Meaning | What it does not prove |
|---|---|---|
| Static/request batch | A fixed cohort is executed together. | Online replacement or fair service. |
| Continuous batching | Membership can change at safe execution boundaries while other requests remain live. | Efficient mixed prefill/decode or bounded latency. |
| Chunked prefill | A long prompt is processed in bounded pieces. | Decode latency isolation, fairness, or cancellation responsiveness by itself. |
| Fused/shared-round execution | Multiple requests contribute useful work to a common production execution round. | Correct capacity, independent state, or sustained lane replacement without additional tests. |

Orca established iteration-level scheduling and selective batching as serving concepts. PagedAttention addressed memory fragmentation and sharing. Sarathi-Serve treats prefill/decode interference and chunked, stall-reducing scheduling as a separate throughput/latency problem. Their published speedups are workload- and implementation-specific and are not estimates for Llaminar. [P1–P3]

### 3.2 Use a work budget, not only an active-request limit

vLLM's pinned scheduler expresses work as the gap between `num_computed_tokens` and required prompt/output/speculative tokens (`num_tokens_with_spec`). The source explicitly does not assign a separate scheduler “prefill phase” and “decode phase.” It schedules running requests first under token, sequence, cache, and other limits, then considers waiting work. Documentation's “decode-first” description is a useful shorthand, not a complete description of all running work. A running long prefill is still running work. [V1, V5]

For Llaminar, the corresponding **device-owned** work policy should account for at least:

- Ready decode requests and their latency/debt budgets.
- Remaining uncached prefill work and its chunk geometry.
- MTP proposal/verification work, not only emitted tokens.
- Context-dependent attention cost, recurrent-state updates, and routed-expert load.
- Available admitted logical state/storage leases and output-buffer credits.
- Fairness/age, cancellation intent, and completion/retirement boundaries.

A cap of eight requests says little about work: eight short answers and eight 70K-token generations have radically different occupancy. The token budget is also only a proxy for time; equal token counts at different context lengths, MTP depths, or MoE routes need not have equal cost. Start with an explicit measurable policy, then calibrate it from isolated evidence. Do not invent another anonymous memory reserve or subtract capacity outside PMA. [L8, E1]

### 3.3 Define separate queues and lifecycle states

A proposed ownership state machine is:

`Queued → Admitted → Prefilling/Ready → RunningWindow → Publishable → Ready or Terminal → Retiring → Free`

Cancellation is an orthogonal intent with explicit observation/commit rules, not an immediate free from every state. Required distinctions:

- **Queued cancellation:** remove external work without allocating execution state.
- **Admitted cancellation:** issue an ordered request-scoped command; the device owns when the active transaction can stop safely.
- **Finished generation:** tokens/state are committed, but output delivery or a prefix snapshot may still hold leases.
- **Retirement:** release a slot only after all execution, publication, transfer, and cache-reader owners are retired.
- **Admission replacement:** use a new generation-qualified identity; never let an old completion, event, or disconnect target a reused slot.

A stable host submitter is still useful. Continuous batching does **not** require one inference thread per request. On CPU, execution control can be host-owned; on GPU, host queue/admission and transport metadata must not become a shadow of live controller, sampler, KV, or response state. [L1, L4, L10, L13]

### 3.4 Fairness, backpressure, and streaming are first-class

Define bounded policies for:

- A short interactive request arriving behind or beside a huge prompt.
- Long generations that retain a slot for minutes.
- Unequal prompt lengths and staggered completion.
- A slow/disconnected SSE consumer without stalling every healthy lane.
- Limits on queued prompt bytes, queued requests, admitted state, and retained terminal output.
- Whether a blocked output consumer suspends its slot, cancels it, or rejects further output—and when that transition is safe.

Do not make early production depend on preempting a live GPU request to host state. If preemption/offload is supported later, it needs its own complete state/transfer/capture contract, including recurrent and MTP state, and explicit economy gates. Otherwise apply bounded admission and a precise failure/backpressure policy. vLLM's allocation/preemption behavior is a reference, not a license to add an automatic Llaminar fallback. [V1, V2, L13]

SSE token bursts complicate latency metrics. A 16-token publication window can make client-visible inter-event gaps much larger than average model time per token. Record both output-event timing and committed-token counts; do not spread burst tokens artificially across the preceding gap or advertise their average as actual client ITL. [L2, V9]

## 4. How should KV, prefix caching, and memory fit together?

### 4.1 Extend existing authorities, not parallel accounting

Llaminar already has sequence-aware KV interfaces, ring/cache layouts, prefix snapshots, recurrent state, physical-memory estimation, and a canonical `PhysicalMemoryAuthority`. Current ring backing reserves full configured context per slot, and prefix restore copies into that private slot. Thus neither “there is no KV support” nor “a shared paged pool is already available” is accurate. [L7–L9]

A useful physical BOM separates:

- Weights/prepared weights and persistent graph/workspace storage.
- Main KV and any MTP sidecar state.
- Recurrent/short-convolution state and rollback/checkpoint images.
- Shared immutable prefix backing versus request-private writable tails.
- Communication, prefix transfer/staging, output, and device-control storage.
- Concurrent graph ownership versus storage reusable by mutually exclusive graphs.

Logical request occupancy must not become a second physical-capacity ledger. PMA's current suballocation claims correspond to backing allocations, not a completed logical-page grant protocol. In the first proposed fixed-slot mode, execution admission leases an already admitted slot. Cache-resident immutable prefix backing can be charged once, but live private slot copies remain separately charged. A future paged/shared-prefix mode requires its own PMA-owned logical grants and native format/backend implementation. [L8, L9, L13]

The revised design explicitly selects fixed-slot, full-context backing as the first increased-capacity physical mode, with its own economy gate and no claim of economical maximum-context concurrency. Unspecified output limits currently use remaining context; “guaranteed growth” in this mode is slot admission, not a new paging gain. Paged KV requires append/attention/export/import, recurrent/MTP and terminal-state contracts on every supported backend/format, plus ordered grant retirement. It remains a separate unqualified mode.

### 4.2 vLLM's useful lessons

vLLM's block pool and KV manager distinguish physical blocks, request references, computed/cacheable blocks, allocation, and release. Prefix hashes incorporate ancestry and additional identity inputs; documented `cache_salt` can isolate reuse across trust groups. This is separate from numerical determinism. [V2, V3]

Hybrid KV management matters: attention layers and recurrent/Mamba state have different storage and reuse characteristics. A common attention prefix is insufficient if the corresponding recurrent state is unavailable at that boundary. The pinned KV manager also deliberately limits an all-hit lookup to obtain the terminal computation/logits it needs. “Every prompt token was cached” does not universally imply zero remaining work. [V2, V4]

For Llaminar, cache compatibility must include the existing model/runtime fingerprint, state layout, precision/codebook, positional semantics, execution topology where relevant, and every model-specific state component. If multi-tenant serving is intended, add an explicit trust-domain reuse policy; exact token equality alone is not authorization to share. This report does not claim the current HTTP surface already has vLLM-style cache salts. [L9]

### 4.3 Native TensorFold's useful lessons and limits

The current native engine has an explicit host scheduling loop: admit pending work, process cancellation, fill available lanes, execute a round, and retire completed work. `--parallel N` is active capacity; additional requests wait. `--parallel auto` asks for up to eight streams where memory and engine support permit it; an unsupported explicit request is refused. [T1–T3]

However, execution is family-specific. The `family_streams`/shared-round path and an engine's capability matter; a visible HTTP slot count is not evidence of a fused batch. The loop also has paths that invoke family work lane by lane. Do not transplant those into Llaminar as production serial-row replay. Prove the actual economical grouped path separately. [T2]

TensorFold's native prompt cache uses exact token/canonical decoded-interval rules and family save/restore/drop hooks. GLM's pool-backed radix sharing, idle-page reclamation, and Qwen's prefix adapter are not one universal implementation. The inspected `TF_GLM_KV_SHARE` and `TF_QWEN_KV_SHARE` pool-sharing paths are off when unset/zero; this does not mean all prompt caching is off. Namespace/layout and terminal-state compatibility remain model-specific obligations. Both adapters also budget retained state snapshots, not only KV pages. More token capacity alone does not prevent live lanes from evicting one another's required state. [T3–T5]

The GLM adapter additionally requires a server-supplied content key for image prompts; matching placeholder token IDs is insufficient. It binds MTP lookahead/state to the cache boundary. This is a concrete reminder to include multimodal content identity and model-specific terminal state, not only text tokens, in any reusable prefix contract. [T5]

Prefix-dependent admission can avoid redundant prefill, but waiting for a potential donor creates another queueing trade-off. A default wait or a successful small cache demo is not a latency SLO. Likewise, native family code that synchronizes during a cache operation is not a pattern to copy into Llaminar's GPU hot path: map it to the authorized restore boundary with TransferEngine and event-aware ownership. [T3–T6, L9, L13]

### 4.4 Prompt serialization can dominate apparent cache performance

Cache keys operate on rendered tokens/state, not an abstract chat history. Moving an instruction into the leading system message, changing tool schemas, changing a template/tokenizer, or adding a timestamp near the beginning can destroy reuse even when most human-readable history is unchanged.

Preserve append-only serialization when semantically valid, but never weaken instruction ordering or tool semantics to win a cache benchmark. Compare rendered token prefixes and model behavior. A cache-hit percentage alone hides restore latency, recomputed terminal work, memory pressure, and cross-request eviction. [T4, T6, E2]

## 5. How can this coexist with captured execution, MTP, and multiple devices?

### 5.1 The execution-policy mismatch is fundamental

Llaminar's `DeviceGenerationExecutionPolicy` distinguishes `NativeConditionalGraph`, `HostScheduledCapturedTransactions`, and `Unsupported`. CUDA-native eligible domains require their complete conditional policy; HIP and explicitly supported boundaries use authenticated scheduler tickets naming retained complete transactions. The host submits the named transaction; it does not reconstruct sampler/verifier decisions or mirror live GPU state. Missing support is fatal. [L10, L13]

vLLM's documented graph modes allow full decode with piecewise mixed work, runtime graph dispatch, and mode downgrades when an attention backend cannot support a requested mode. Its `BatchDescriptor` makes batch shape/mode identity explicit, which is useful. Its eager/piecewise escape paths are **not permitted substitutes** for Llaminar's complete homogeneous captured policy. [V6]

### 5.2 Candidate seam: a complete multi-request service window

The existing handler constant is `kStreamingPublicationWindowTokens = 16`. The documented reason includes accommodating MTP depth 15 plus a target row. Each scalar streaming window uses the retained complete production path and preserves resident continuation. [L2, L3]

**Proposal:** investigate a complete captured multi-request serving policy owned by the existing `OrchestrationRunner`, not another engine above it. Revision 3 chooses single-request canonical prefill chunks alternating with grouped decode, a separately typed lifecycle/selector operation, and one outstanding host-backed window grant per slot-generation, rather than unproved fused-M dispatch. A normative host/device payload table, independent host lease/device slot machines and per-row ABI are prerequisites. The host must not reconstruct scheduling, thinking-budget or maintenance decisions from decoded progress.

Before accepting this seam, prove:

1. A completed window is a true transaction boundary, including every participant's exit frontier.
2. New prompt/sampling ingress is an ordered lifecycle operation into preadmitted stable storage, not allocation/rebinding during capture.
3. Per-slot reset cannot reset another request's KV, recurrence, sampler, MTP, output, or model-local state.
4. The device owns active lengths, stop conditions, MTP acceptance, and scheduling decisions after admission.
5. Membership changes use stable slot addresses and generation-qualified metadata, or another explicitly captured identity scheme. No pointer silently changes meaning under a cached graph.
6. A terminal result for one request does not invalidate continuation or borrowed state for another.
7. Output publication is bounded and immutable; cancellation and backpressure do not expose mutable device state.
8. Replay/reset equivalence holds without recapture, host count readback, or segmented homogeneous execution.

`ForwardGraphTypes` already captures batch/sequence geometry, topology, borrowed addresses, and capture identity. The proposal must extend this authority rather than invent a “close enough” cache key. The existing whole-request reset and fixed-cohort completion contracts mean this work is substantive, not a handler-only change. [L3, L4, L6, L10]

### 5.3 MTP needs per-request economics as well as correctness

In a continuous batch, requests differ in context, proposal length, acceptance history, sampling, stop budget, and completion time. Keep distinct:

- Physical verifier capacity and capture envelope.
- Device-authored live verifier lengths.
- Draft/proposal policy and verification policy.
- Accepted draft tokens, correction/target tokens, and committed output.
- Per-request RNG identity and logical positions.

The RNG key is the request-resolved seed, logical output position and draw purpose. RequestId, slot, generation, packed row and queue ordinal never enter it. Preserve original request coordinates under any compaction. Numerical comparison uses the same request-local chunk/control policy and placement-epoch sequence; dynamic residency is not silently disabled. LLEP assignment itself must be request-local or eligible replicas must be proven byte-equivalent, not merely aligned to the same window grid. Compatibility grouping cannot silently change a sampler or disable MTP. [L3–L6]

Current vLLM adaptive verification is a useful separate reference: it can retain a fixed physical proposal envelope while selecting per-request verification length from confidence/economics. Its documented supported path is opt-in and constrained; it is not the same mechanism as the PCAI concurrency-to-depth table below. Do not copy its defaults or assume long-context calibration from a short-context profile. [V8]

Acceptance percentage alone is not economy. Measure useful committed tokens per wall time, verifier rows/work per committed token, proposal cost, and latency under load. vLLM's standard speculative-decoding distributional claims and optional batch-invariance mode are not identical to Llaminar's serial-row byte-equivalence requirement. [V7, V12, E1]

### 5.4 Multi-device support must preserve the same request coordinates

Separate one-process rank-local TP, cross-rank TP, dense PP, ExpertOverlay, heterogeneous boundaries, and remote MPI. A passing local-TP batch test does not qualify the others. [L3, L14]

Required invariants include:

- All participants agree on request identity, transaction epoch, slot generation, and publication/retirement order.
- Graphs remain participant-local; movement is an explicit collective or TransferEngine boundary.
- Empty/inactive participants exchange only required metadata.
- Collectives transfer live owned data, not slot capacity, verifier capacity, padded inactive rows, or unowned experts.
- Shared workspaces follow actual concurrency; a graph cannot reuse another live graph's scratch merely because their logical requests differ.
- Existing stream/event identities and ordinary collective timeout contracts remain intact. No host barriers, count readbacks, recapture, or enlarged timeout to hide a missing protocol. [L6, L10, L13]

## 6. What did the PCAI vLLM deployment actually establish?

<a id="pcai-owner-evidence"></a>

**Evidence class:** owner-reported measurements and private operational observations, supplied by the `pcai-llm-serving` session on 2026-11-05, plus public author follow-up summaries. The Dockerfile/PRs establish source provenance and patch intent. PR bodies describe GPU validation as pending; timestamped follow-up comments on PR85/88 subsequently report bounded GPU capacity/kernel/serving results. Those comments are author-reported summaries, not independent reproduction or public raw receipts for every combined-rollout claim below. [V10, E1]

### 6.1 Workload and configuration

Reported deployment: **2×H100 NVL, TP2**, DeepSeek-V4-Vision serving mixed coding/agent sessions, vision, and documents up to 1M tokens. Relevant settings:

- `max-model-len=1048576`, `max-num-seqs=8`, `max-num-batched-tokens=4096`.
- Utilization 0.96, FP8 KV, block size 256, V2 runner, `FULL_AND_PIECEWISE` graphs.
- Humming MXFP4 with block-FP8 activations/group128.
- DSpark depth 5 for 1–2 active requests, 3 for 3–4, and 2 for 5–8.
- `SimpleCPUOffloadConnector` with 192 GiB host KV; UVA vision/aligner/embedding offload.
- `indexer_tp_shard_min_rows=1024`. The prior/default 65536 threshold would not activate within a 4096-token chunk. This shares sparse-indexer prefill rows across TP; it is **not generic attention context parallelism**. [E1]

These are a recorded operating point, not suggested Llaminar defaults. In particular, a utilization fraction is not a replacement for Llaminar's typed PMA BOM and admission.

### 6.2 Patch lineage and measured claims

The owner reported a carried vLLM#58074 C128 ring change and image changes including PR78 hostpool initialization, PR80 waits, PR84 packed fused-E8M0 scales, PR88 a native half-width launcher correction, and PR85 omission of an unused DSpark auxiliary-path MTP residual buffer. Public upstream/image discussions establish patch intent and some bounded author-reported results; retain the exact image and flag set as the measurement identity. `HUMMING_USE_PACKED_BS` and `VLLM_PCAI_DSPARK_SKIP_MTP_BUFFER` were enabled in the reported combined comparison. [V10, V11, E1]

The upstream C128 PR explicitly scopes its circular layout to CUDA and includes read-before-tail-save handling to prevent wraparound overwrite. Its author clarifies that the old paged state was **already preallocated**: the benefit is smaller bounded-state reservation, not elimination of per-decode full-history reallocations/copies. This is a physical-liveness lesson, not a universal backend or batching speedup. [V11]

| Metric | Same-image flags off → combined opt-ins | Qualification |
|---|---|---|
| KV capacity | 1,129,474 → 1,511,456 tokens, +33.8% | Capacity, not concurrent full-context guarantees. |
| Model memory | 78.61 → 76.32 GiB/GPU | Owner-reported allocator/model accounting. |
| Graph memory | 0.61 → 0.61 GiB | No reported graph-memory reduction. |
| Cold 1M TTFT | 245.659 → 165.779 s | One reported operating point; not a cross-engine benchmark. |
| Eight-request aggregate decode | 952.8 → 952.6 tok/s | Essentially unchanged point measurements, not statistical equivalence proof. |
| Single-request decode | 208.8 → 210.8 tok/s | Small observed difference, no general speedup claim. |
| Bounded checks | Vision 3/3; boundary 8/8; stability 456/0 | Scope-limited, owner-reported. |
| Quality suite | 48 valid passes versus earlier 49–50 | Does not establish statistical or bitwise quality parity. |

Public author follow-ups report packed-only at about +32% KV capacity with a 1–3% decode cost, and buffer-only at +20,876 tokens (+1.85%). The packed-only cold-1M TTFT was 245.659→245.454 seconds—almost unchanged. Therefore, do not attribute the combined run's larger TTFT difference to packing alone or assume additive/causal synergy from the isolated results. The right lesson is **account for which memory is actually live and prove transformed layouts**, not “all batching optimizations improve decode throughput.” [V10, E1]

StreamK BF16 atomics caused baseline-repeat nondeterminism in the owner's investigation. StreamK-off eager/graph exactness on eight real-weight cases does not certify full default-model byte parity. Treat layout correctness, numeric contracts, free-form output variance, and performance as separate measurements. [E1, V7]

### 6.3 Private 417K-prefill incident: useful, but not a benchmark

**Operator-collected single incident; private retained logs/DB metadata; no payload retained. No public incident report or URL exists.** Generic repository links do not document this event. [E1]

Reported Nov 4 record:

- Start `15:09:12.546113Z`; input **417,036** tokens, reported cached **14,848**, output **406**.
- Total **165,042.750 ms**, upstream **50,934.311 ms**, gateway overhead **114,108.439 ms**.
- Derived end `15:11:57.588863Z`. Subtracting upstream duration gives `15:11:06.654552Z`, but this is **not an observed backend admission timestamp**.
- At `15:11:18/28/38/48Z`, generation was **48.2/35.4/34.9/25.2 tok/s**, with eight running requests and KV usage **60.8→74.3%**.
- At `15:11:58Z`, logged computed-prompt throughput was **40,634.8 tok/s** over the reporting interval and KV usage 19.9%; generation subsequently returned to roughly **900–970 tok/s**.
- Zero pending host stores and no crash were reported.

Near-cold prefill and the decode slowdown are strongly correlated. There is **no profiler-confirmed kernel, thermal, or allocator diagnosis**. Usage subtraction gives **402,188 uncached tokens**; it is not an exact match to roughly 406K logged computed work, which can include other requests/accounting. Prompt statistics appear on first output, explaining the delayed accounting spike; zero logged prompt throughput does not imply GPU idleness.

The research implications are bounded but important:

1. Chunk size 4096 did not by itself guarantee low decode interference in this incident.
2. Gateway delay and backend service must be measured separately.
3. Completed-only p99 omits the worst currently active requests.
4. Backend status counters need timestamp/aggregation semantics before causal interpretation.

### 6.4 Other workload lessons

The owner observed 46K–72K-token outputs occupying slots for 8–15 minutes. For speculation, depth around 5 produced roughly 38–39% acceptance with mean accepted length around 2.9, while depth around 2 produced roughly 55% acceptance with length around 2.26. The higher percentage did not mean more useful progress per verification. These are workload observations, not a general optimal-depth table. [E1]

## 7. What did the Spark TensorFold work actually establish?

<a id="spark-owner-evidence"></a>

**Evidence class:** sanitized historical observations from `dgx-spark-lab`, supplied on 2026-11-05; not rerun here. The tested stack was the historical patched engine/recipe identified in §1.2, not current upstream TensorFold. Model: non-abliterated **GLM-5.3-Flash EXL3 4bpw with native checkpoint MTP**, across two **GB10/SM121 ARM64 UMA** nodes. [E2]

### 7.1 Concurrent decoding was observed, within a bounded scope

Four short capped requests demonstrably overlapped: a native sample showed four decoding requests; all streams completed usage/DONE, and each emitted `mtp_batched` statistics. This supports concurrent batched execution for that fixture.

It does **not** establish sustained lane replacement, arbitrary prefill/decode fairness, long soak behavior, or four full 1M contexts. The shared pool was **1M total**, not 4×1M. Current upstream source shows an explicit admission loop, but that is a different evidence statement from the older deployment's four-request observation. [T1–T3, E2]

### 7.2 The large prefix win was a serialization repair

| Same historical comparison | Leading-system rewrite | Append-preserving serialization |
|---|---:|---:|
| Restored/cached tokens | 0 | 18,176 |
| Uncached suffix | Not comparable as a short suffix | 61 tokens |
| First output | 12.177 s | 0.355 s |
| Native prefill | 11.087 s | 0.282 s |
| Queue | Approximately 0.018 s | Approximately 0.018 s |

The Pi adapter change `supportsMidConvoSystemMessages=true` preserved prompt-prefix structure; it did not accelerate the scheduler. In another observation, adding a tool/schema header reduced common prefix from **3913 to 90 tokens** despite unchanged user history. Preserve exact prompt rendering when interpreting cache results. Bounded semantic checks are not general proof of instruction-hierarchy or tool-call correctness. [E2]

### 7.3 Separate startup, storage, and decode claims

- A restart fixture proved **3520-token SSD restoration** with successful checksummed reads. It did not prove eviction-scale, cross-model, concurrent-retirement, or arbitrary-prefix correctness.
- Prepared weights/direct I/O reduced historical startup **718→35 seconds**. That is not decode throughput.
- A clipped one-slot startup was rejected rather than weakening the reserve. This is an admission behavior, not evidence that the requested larger concurrency had become economical.
- No public upstream PR was established for the complete local findings/HTTP changes. [E2]

The transferable lessons are exact prompt/cache identity, admission that refuses unsupported capacity, explicit model-family snapshot hooks, and evidence that distinguishes slots from useful shared work. The old patched stack is not a substitute for Llaminar's ARM, capture, numerical, or economy gates.

## 8. Comparison: adopt the concepts, not incompatible execution policies

| Dimension | Llaminar inspected state | vLLM reference | Native TensorFold reference | Proposed Llaminar direction |
|---|---|---|---|---|
| HTTP lifecycle | Whole-request serialized worker | Online request scheduler | Admit/cancel/fill/step/retire loop | One submission authority with independent request leases. |
| Batch membership | Existing fixed-cohort MTP APIs | Iteration-level token-deficit scheduling | Explicit refill; execution family-dependent | Replacement only at proved transaction boundaries. |
| Mixed work | No continuous HTTP contract established | Token budget; running-first source policy; chunked prefill | Family/engine-specific rounds | Device-owned budget with decode SLO and prefill-age policy. |
| Memory | PMA, arenas, sequence KV, prefix snapshots | Block pool and cache references | Pools, radix/cache hooks, optional sharing | One physical ledger; explicit logical leases and private tails. |
| GPU control | Device-owned full policy | Host scheduler and flexible graph modes | Native host lane controller | Preserve device ownership; never mirror state for scheduling. |
| Graphs | Exact identity; no homogeneous eager/segmented path | Full/piecewise modes and documented downgrades | Family-specific implementation | Complete retained service-window policy, if proved. |
| Prefix correctness | Model/runtime fingerprint and state leases | Hash identity, optional salt, hybrid-state manager | Exact tokens plus family adapters | Prefix + recurrent + MTP + terminal-state compatibility and trust domain. |
| Sampling | Position-keyed request batching foundations | Default batch invariance not universal | Family-specific contracts | Resolved seed/position/purpose RNG and serial-row byte equivalence under the same policy/placement trace. |
| Performance evidence here | None for continuous serving | Owner measurements on one patched image | Owner historical bounded fixtures | Matched Release SLO/goodput and economy gates. |

No ranking of “fastest engine” follows from these sources. Models, quantization, GPUs, contexts, cache states, patches, graph modes, and output contracts differ.

## 9. Proposed implementation sequence and acceptance gates

These are **development stages, not permission to ship partial support under automatic fallback**. Each admitted mode needs correctness and economy together. An incomplete backend/topology/policy must fail explicitly; passing one stage or backend must not be advertised as complete continuous-serving certification. Default optimized paths, prefix caching, stochastic sampling, and automatic MTP remain obligations before their supported production configurations are qualified. [L13]

### Stage 0 — Set the serving contract and workload

Decide target concurrency, prompt/output/context distribution, SLOs and trust domains. Preserve every existing topology/mode at capacity one before adding higher typed capabilities. Formally resolve the proposed serving-boundary policy; document authorization is not adoption. Record admission/cancel/grant semantics, intrinsic windows, response-lifetime capacity and memory failure. Audit what the present scalar API owns globally. Specify a typed compatibility key and a request/slot generation identity; do not use a stringly typed mode soup.

**Exit evidence:** reviewed ownership/state diagram, physical BOM, immutable graph-envelope definition, and a recorded-input controller trace schema. No performance claim yet.

### Stage 1 — Make request lifetimes independent

Before implementation, classify affected state as slot, transaction or domain state, and retain `OrchestrationRunner` as the sole lifecycle/controller authority. First migrate every existing mode/topology to a capacity-one per-row seam without silently changing numerical/control policy. Include KV/recurrent/MTP state, terminal logits/hidden/condition/sidecar handoffs, sampler/stop/think state, terminal outputs, stats and prefix leases without resetting domain state. Extend the existing owners; retire replaced machinery rather than retaining two unrelated serving paths. Prove slot-local reset and generation-safe reuse without invalidating long-lived graph/workspace assets.

Keep device-free state-machine tests fast. CPU implementations can validate host-owned scheduling semantics, but do not use a CPU mirror as the GPU implementation. ARM execution tests wait for the ARM owner's applicable build/runtime gates.

**Exit evidence:** adversarial lifecycle/contract tests and recorded controller replay; one request's cancellation/reset/retirement cannot affect another.

### Stage 2 — Complete the economical multi-request execution substrate

After formal policy adoption, qualify the common intrinsic-window/phase policy at capacity one before increasing topology capacity. Close ordinary and MTP batch gaps under typed identities: independent laws/budgets, host-backed grants, per-slot history and default penalties on CPU/CUDA/ROCm. Provide live-length-aware KV/recurrence, private live prefix copies, persistent prefill handoff and economical grouped decode/verifier kernels. GPU DRY history is new complete-target work, not an existing cutover-parity claim. Preserve byte equivalence to serial-row references without calling row replay in production.

Design CPU, CUDA, and ROCm interfaces together; qualify each explicitly. Multi-rank, PP, and overlay extensions need their own complete request/communication contracts, not merely the removal of capability checks.

**Exit evidence:** format/backend/geometry sweeps; production captured tests; exact communication extents; PMA BOM reconciliation; isolated kernel profiling where kernels change. Large-to-small replay and partial/empty work must be explicit.

### Stage 3 — Add complete continuous service windows

Implement admission insertion, alternating single-request canonical prefill/grouped decode, typed lifecycle-only retirement, and replacement around the formally approved captured service contract. Keep unqualified capacity at its explicitly reported bound and reject higher authored requests; never silently clamp. Device-owned GPU control selects live work; host transport handles only authorized boundaries. Preserve stable capture identity and workspace ownership without recapture or hot-path allocation.

Connect every application mode to the single runner-owned request seam rather than spawning competing inference runners or retaining separate scalar benchmark loops. Extend streaming/stats to independent requests, detached staging, inclusive output credits and lifecycle-only cancellation/retirement.

**Exit evidence:** A and B begin; B completes; C enters while A remains live; repeat with cancellation, unequal prompts, prefix hits/misses, different samplers, MTP depths, and output budgets. Trace must prove actual shared execution and replacement—not merely overlapping sockets or interleaved logs.

### Stage 4 — Tune fairness and MTP economics

Measure policy choices against admitted workloads: canonical chunk/window sizes, device-counted fairness, MTP work and concurrent cache restore/harvest. A different intrinsic quantum is an explicitly versioned policy experiment, never a per-batch load adjustment. Preserve the applicable numerical/capture contract and include selector/branch receipt costs and I/O stalls. Add cost evidence to existing authorities rather than creating a parallel planner or memory ledger.

**Exit evidence:** lower or acceptable TTFT/ITL tails at the target offered rate, improved SLO-qualified goodput, sensible per-request progress, and no hidden increase in unserved/in-flight work. Acceptance percentage is not the acceptance gate.

### Stage 5 — Qualify all advertised production cells

Extend canonical typed test/benchmark inventory instead of maintaining a second topology/model matrix. Qualify dense and MoE, MTP and non-MTP, CPU/CUDA/ROCm, prefix tiers and quantization, local/remote participant layouts, and supported hybrid models. Every required cell executes without silent skips. Unsupported cells remain explicit and unqualified.

**Exit evidence:** complete required Unit and `ProductionTestPreflight`, focused production regressions, native capture/integration gates, canonical Release HTTP E2E/benchmarks, and actual multi-session application tests. This research does not run or waive those gates. [L12–L14]

### 9.1 Focused regression matrix

Names below describe proposed contracts, not current CTest registrations.

| Area | Required adversarial cases | Required evidence |
|---|---|---|
| Admission/identity | Empty queue; full capacity; duplicate/stale IDs; generation exhaustion fails closed without wrapping; queued cancellation; insertion during an in-flight window | No mutation of live transaction ownership; precise reject/backpressure; controller decisions from recorded inputs. |
| Coordinates/numerics | Request permutations; holes; M=1 through supported range; N/K and positive CPU-thread counts; dispatch totality beyond measured points; unequal contexts; large→small replay; all supported formats/codebooks | Same logical request/position and serial-row byte contract; no serial-row production execution. |
| Sampling/MTP | Nonzero/zero seed semantics; greedy and stochastic; fixed/dynamic depth; rejection/correction; early EOS/stop/length; different per-request budgets | Correct per-request draws and committed state; no other request changes RNG or acceptance. |
| KV/hybrid/prefix | Shared cache entries with private live restored copies in fixed-slot mode; partial blocks; admitted KV formats including TurboQuant; per-slot terminal/sidecar handoff; recurrent checkpoints; MTP rollback; concurrent restore/harvest/cancel/evict races | No state leak/alias; exact compatibility; every physical allocation charged once, including private restored copies; leases outlive consumers. |
| Capture/reset | Slot replacement without recapture; stale pointer/workspace generation; graph-envelope mismatch; repeated recorded replay; partial/empty work | Exact identity; complete captured policy; hard failure for unsupported/stale state. |
| Transfer/collectives | Empty participant; skewed experts; partial buckets; large→small replay; rank-local and cross-rank layouts | Actual live communicated byte counts, not just correct final output; exact stream/event ownership. |
| HTTP/lifecycle | Slow reader; execution Released while response Delivering; grant return/replay; all-no-grant lifecycle-only drain; disconnect/terminal races; shutdown; stats reset; discovery under load | Response-sized bounded output; one grant per generation/window; no empty-generation call; immutable outcomes; old epoch cannot repopulate counters. |
| Economy | Short chat plus huge prefill; long outputs; mixed prefix hit rates; saturated arrival load; memory pressure | SLO goodput, active-age tails, useful work/byte, and bounded prefill/decode interference. |

Every discovered production defect needs a focused regression explicitly registered in `ProductionTestPreflight`, as well as applicable broad sweeps. New/changed controllers and GPU kernels need **recorded-input replay** through the production implementation with policy/version, precision, geometry, ordering, and initial state preserved. Synthetic repetition alone is not replay. Lifecycle and geometry tests must be named accordingly. [L12, L13]

Existing anchors include `DeviceRequestRowRanges`, `IKVCache::SequenceRetirement`, request-batched resident-prefill probes, and `V2_Integration_HTTPRuntimeStatsAccounting`, `V2_Integration_HTTPRuntimeStatsResponsiveness`, and `V2_Integration_HTTPRuntimeStatsHandlerLifecycle`. `MTPSpecRequestBatchOwner` may serve as a host arithmetic/ownership oracle; its per-call integer/string interface and in-flight mutation lockout must not become the GPU serving controller. Replace or retire superseded machinery instead. Registration/source existence is not new passing evidence. [L7, L12, L14]

### 9.2 Measurement protocol

1. **Freeze identity:** Release binary/image, exact source/patch set, model/tokenizer/template/quantization, prompt bytes, hardware/ISA, topology, context, sampling, prefix state, MTP policy, graph mode, and offload policy. Measure production automatic defaults first; make each override explicit and separately measured.
2. **Separate workloads:** short interactive; mixed agent/tool turns; long outputs; near-cold long prompts alongside decode; prefix-rich repeat turns; mixed cache misses; cancellation/slow consumers. Use synthetic controls and sanitized representative request-length/timing traces, without retaining private prompt payloads unnecessarily.
3. **Run both closed and open load:** concurrency sweeps reveal saturation; an open-loop arrival-rate/burstiness sweep reveals queue growth and SLO failure. A client semaphore can lower the achieved arrival rate—do not call that an unconstrained offered-load test. Record scheduled and actual send times. [V9]
4. **Measure the full path:** queue, gateway, admission, restore, prefill, first model token, first nonempty client-visible output, SSE event gaps, decode, final output, cleanup, cancellation latency. Capture active-request age and time since last progress, not only completed-request percentiles.
5. **Count failures and unfinished work:** offered, admitted, rejected, cancelled, failed, completed, and still active at measurement end; queued bytes/tokens; deadline misses. Do not silently omit in-flight outliers or declare a partially drained run successful.
6. **Report throughput and goodput separately:** aggregate committed output tokens/s; per-request output rate; prompt work/cache-hit/restore work; requests/s satisfying every declared SLO. Publish p50/p95/p99 and worst active age with sample counts and distributions, not just means.
7. **Measure useful execution:** live versus admitted rows; verifier work per committed token; actual collective payload bytes; device↔host bytes by authorized boundary; capture/replay/recapture counters; KV/prefix unique physical bytes, logical references, and eviction/restore bytes. Passive stats must not synchronize or probe live GPU state.
8. **Keep evidence clean:** warmup/cache policy fixed; repetitions and uncertainty reported; isolated CPU/GPU profiling separate from canonical unprofiled timing. New kernels require occupancy/register/spill/memory-throughput evidence and the full shipped architecture compilation set. CPU work needs physical-core/NUMA/ISA scaling, not guessed hyperthread budgets.

A practical comparison reports the highest sustainable offered load meeting TTFT/ITL/E2E objectives **and** memory/correctness constraints. No target multiplier is assigned here. Choose SLOs before tuning; do not select them after seeing the fastest result.

## 10. Resolved contradictions, uncertainty, and decisions still needed

### Resolved or scoped apparent contradictions

| Apparent conflict | Resolution |
|---|---|
| “Llaminar supports batching” versus serialized HTTP | Both are true: fixed-cohort lower-level MTP machinery exists; online HTTP membership/lifetimes are not established. |
| vLLM “decode first” versus no prefill/decode phases | Documentation summarizes an effect; pinned source uses token deficits and running-request priority. |
| TensorFold `--parallel` versus universal fused batching | It limits active lanes; family capability and the executed branch determine shared work. |
| Four Spark requests versus four 1M contexts | Observed short-request overlap used a shared total pool, not four independently guaranteed maximum contexts. |
| Large cache/TTFT gains versus scheduler speedup | Prompt serialization restored a long prefix; queue time was nearly unchanged. |
| PR body says GPU pending versus follow-up results | Timestamped public author comments supplement the body; bounded reported tests, private combined-rollout evidence, and independent reproduction remain distinct. |
| Higher speculative acceptance percentage versus better throughput | Depth changes the denominator and work; measure committed tokens and cost. |
| GPU full capture versus host-selected transactions | Authenticated named retained transactions are an explicit policy on supported boundaries; host-owned model control or graph downgrades remain disallowed. |

### Remaining uncertainties and gaps

- The proposed multi-request service window is unimplemented. Its compatibility with every Llaminar full-generation, reset, streaming, MTP, and topology contract requires design review and production proofs.
- No new builds or tests ran. ARM whole-engine qualification and complete gates remain outside this report; focused peer counts cannot substitute.
- Peer measurements were not independently reproduced. The 417K event establishes correlation, not a profiler-confirmed cause, and has no public receipt.
- The historical TensorFold artifact includes a substantial patch stack and local changes. Its base SHA cannot reproduce the observations alone.
- Current vLLM/TensorFold support remains backend/model/version-specific. Moving documentation may later diverge from the pinned source.
- Optimal concurrency, publication quantum, chunk budget, memory layout, prefix-wait policy, and MTP depth are workload-dependent and unmeasured for Llaminar.
- Numerical equivalence between different engines/quantizations is not established. Compare serving economics only with explicit correctness contracts and aligned workloads.

### Product/design decisions to settle before implementation

1. Target concurrency and context/output distributions, including whether million-token prompts are routine or exceptional.
2. TTFT, client-visible inter-event/ITL, E2E, cancellation, and fairness SLOs; acceptable queue/rejection behavior.
3. Single-tenant trusted prefix reuse versus explicit tenant/trust-domain separation.
4. Supported initial production topology/model envelope and the complete gates required before advertising it.
5. Approval of the GPU service-window contract: exactly which immutable ingress/publication/retirement information crosses the host boundary, and how all live decisions remain device-owned.

## 11. Recommendation

**Primary:** extend `OrchestrationRunner` as the single lifecycle/controller authority, migrate all application modes to its request seam, and prove independent slot lifetimes through a complete device-owned serving policy. First preserve every existing topology/mode through capacity-one ownership, then qualify the explicitly versioned common serving policy and raise only proven capacities. Start from the fixed-slot physical envelope; do not assume paging, heterogeneous per-row sampling, persistent handoff banks or concurrent prefix restore already exists. The revised design and independent review make the still-open prerequisites explicit.

**Conditional alternative:** if the 16-token-window seam cannot meet the ownership/latency contract, design a different complete retained serving controller/transaction envelope explicitly. Compare its storage and graph-count costs before implementation. Do not disguise per-token eager submission or host-shadow scheduling as a temporary implementation.

**Not recommended:** multiple HTTP inference workers against one runner, production serial-row loops, implicit prefix/MTP disabling, capacity-sized communication, a second memory ledger, graph recapture on membership changes, unqualified ARM claims, or importing vLLM/TensorFold flags and benchmark numbers as Llaminar defaults.

The next useful engineering artifact is an ownership-and-transition design for admission, service windows, and slot retirement, followed by focused recorded controller replay. It can be designed while ARM work finishes; runtime qualification must remain with the appropriate native backend gates.

## Sources and evidence register

L1–L16 source links pin the inspected unmerged descendant; L17 explicitly pins the develop baseline for the second-review controller/capacity audit. These are not implicit links to whichever working checkout happens to be open. See the [source-provenance note](#inspected-source-provenance) and the review record for baseline differences and source-audit qualifications. External source links are pinned where possible; documentation was accessed on 2026-11-05. The register has 41 evidence groups after the two review source audits, not 41 independent experiments.

| ID | Source | Authority and use |
|---|---|---|
| L1 | [SerializedInferenceExecutor.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/app/modes/SerializedInferenceExecutor.h), [ServerMode.cpp](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/app/modes/ServerMode.cpp) | Primary implementation: bounded admission and stable serialized worker. |
| L2 | [ChatCompletionHandler.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/app/modes/ChatCompletionHandler.h), [ChatCompletionHandler.cpp](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/app/modes/ChatCompletionHandler.cpp) | Primary implementation: scalar HTTP lifecycle and 16-token publication seam. |
| L3 | [OrchestrationRunner.cpp](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/execution/runner/OrchestrationRunner.cpp) | Primary implementation: batched-prefill checks, batch admission/completion, scalar resident generation, seed policy. |
| L4 | [IOrchestrationRunner.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/execution/runner/IOrchestrationRunner.h), [IInferenceRunner.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/execution/local_execution/orchestrators/IInferenceRunner.h) | Primary interfaces: batch atomicity, reset owners, persistent assets. |
| L5 | [MTPSpecRequestBatchScheduler.cpp](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/execution/mtp/MTPSpecRequestBatchScheduler.cpp), [MTPSpecRequestBatchOwner.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/execution/mtp/MTPSpecRequestBatchOwner.h) | Primary host per-call compatibility/ownership helper, not the proposed GPU serving controller. |
| L6 | [DeviceRequestRowRanges.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/kernels/common/DeviceRequestRowRanges.h), [ProjectionRowWork.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/kernels/common/ProjectionRowWork.h) (unmerged descendant) | Primary interfaces: request coordinates, live rows, storage geometry. |
| L7 | [IKVCache.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/kernels/IKVCache.h) | Primary interface: sequence-aware KV contracts; not proof of every backend's continuous serving. |
| L8 | [PhysicalMemoryAuthority.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/planning/PhysicalMemoryAuthority.h), [KVCacheMemoryEstimator.cpp](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/planning/KVCacheMemoryEstimator.cpp) | Primary ownership and typed memory inputs. |
| L9 | [PrefixStateCache.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/execution/prefix_cache/PrefixStateCache.h), [PrefixStateSnapshot.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/execution/prefix_cache/PrefixStateSnapshot.h), [PrefixCacheFingerprint.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/execution/prefix_cache/PrefixCacheFingerprint.h), [PrefixStateBlock.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/execution/prefix_cache/PrefixStateBlock.h) | Primary cache/state identity and lifetime contracts. |
| L10 | [DeviceGenerationExecutionPolicy.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/execution/mtp/DeviceGenerationExecutionPolicy.h) (standalone header in the unmerged descendant), [ForwardGraphTypes.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/execution/local_execution/engine/ForwardGraphTypes.h) | Primary execution/capture policy and complete identity. |
| L11 | [HttpRuntimeStats.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/app/modes/HttpRuntimeStats.h), [HttpRuntimeStats.cpp](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/app/modes/HttpRuntimeStats.cpp) | Primary passive stats/epoch implementation. |
| L12 | [tests/v2/CMakeLists.txt](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/tests/v2/CMakeLists.txt), [Test__MTPSpecRequestBatchOwner.cpp](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/tests/v2/unit/mtp/Test__MTPSpecRequestBatchOwner.cpp) | Registration and focused contracts; no claim of execution in this research. |
| L13 | [AGENTS.md](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/.agents/AGENTS.md), [V2 architecture](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/.github/instructions/llaminar-v2-architecture.instructions.md) | Inspected-descendant policy; recorded replay and dashboard scoping are adopted design requirements, not claims about landed develop wording. |
| L14 | [Test__KVPrefixMTPStateProbe.cpp](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/tests/v2/integration/prefix_cache/Test__KVPrefixMTPStateProbe.cpp) | Existing Qwen3.6 request-batched resident-prefill regression source and its bounded geometry/instrumentation. |
| L15 | [ModelGenerationPolicy.cpp](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/models/ModelGenerationPolicy.cpp), [MTPRequestSamplingPolicy.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/execution/mtp/MTPRequestSamplingPolicy.h), [DeviceGenerationContract.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/execution/mtp/DeviceGenerationContract.h), [DeviceGenerationInitialization.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/kernels/common/DeviceGenerationInitialization.h), [DeviceGraphOrchestratorGeneration.cpp](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestratorGeneration.cpp) | Source-audit additions: model-default penalties, unsupported history, cohort admission and captured scalar sampler. |
| L16 | [NativeCollectiveRows.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/collective/NativeCollectiveRows.h), [NativeAllreduceRequestRows.h](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/src/v2/collective/NativeAllreduceRequestRows.h) | Live-prefix versus ragged ownership and per-bank lowering; no measured economy crossover claimed. |
| L17 | [OrchestrationConfigParser.cpp](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/config/OrchestrationConfigParser.cpp), [RankMemoryPlanInputs.cpp](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/planning/RankMemoryPlanInputs.cpp), [ExecutionPlanBuilder.cpp](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/execution/mpi_orchestration/ExecutionPlanBuilder.cpp), [BenchmarkRunner.cpp](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/utils/BenchmarkRunner.cpp), [SerializedInferenceExecutor.h](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/app/modes/SerializedInferenceExecutor.h), [SamplingMath.h](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/kernels/common/SamplingMath.h), [OrchestrationRunner.h](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/execution/runner/OrchestrationRunner.h), [ChatCompletionHandler.h](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/app/modes/ChatCompletionHandler.h) | Explicit develop-baseline follow-up: learner/empty-generation rules, global handoff state, split capacities, dependent head policy, response admission and benchmark-only calls. |
| V1 | [Deployed-base scheduler](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/sched/scheduler.py), [current comparison scheduler](https://github.com/vllm-project/vllm/blob/cb95d994f4ec066b75672035ca3ae7e1c9745e84/vllm/v1/core/sched/scheduler.py), [current engine core](https://github.com/vllm-project/vllm/blob/cb95d994f4ec066b75672035ca3ae7e1c9745e84/vllm/v1/engine/core.py) | Primary pinned code: token deficits, running/waiting scheduling, engine ownership. |
| V2 | [Deployed-base KV manager](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/v1/core/kv_cache_manager.py), [current block pool](https://github.com/vllm-project/vllm/blob/cb95d994f4ec066b75672035ca3ae7e1c9745e84/vllm/v1/core/block_pool.py) | Primary pinned code: allocation, computed blocks, prefix/state boundary, release. |
| V3 | [vLLM prefix caching design](https://docs.vllm.ai/en/stable/design/prefix_caching/) | Official documentation: hash identity, reuse, cache salts. |
| V4 | [vLLM hybrid KV-cache manager](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/) | Official documentation: heterogeneous attention/recurrent state. |
| V5 | [vLLM optimization guide](https://docs.vllm.ai/en/stable/configuration/optimization/) | Official policy explanation; source refines simplified phase wording. |
| V6 | [vLLM CUDA graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/) | Official modes, descriptors, and downgrade behavior. |
| V7 | [vLLM reproducibility](https://docs.vllm.ai/en/stable/usage/reproducibility/), [batch invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/) | Official distinction between scheduling reproducibility and batch-invariant numerics. |
| V8 | [vLLM adaptive verification](https://docs.vllm.ai/en/stable/features/speculative_decoding/adaptive_verification/) | Official opt-in, constrained per-request verification reference. |
| V9 | [vllm bench serve](https://docs.vllm.ai/en/stable/cli/bench/serve/) | Official request-rate, concurrency, burstiness, percentile, and goodput definitions. |
| V10 | [PCAI Dockerfile at 642dd15](https://github.com/enthus-appdev/vllm-pcai/blob/642dd15/Dockerfile#L3-L6), [image PR85](https://github.com/enthus-appdev/vllm-pcai/pull/85), [image PR88](https://github.com/enthus-appdev/vllm-pcai/pull/88) | Public provenance and bounded author-reported GPU follow-up comments; not independent/raw receipts for every combined-rollout measurement. |
| V11 | [vLLM PR58074](https://github.com/vllm-project/vllm/pull/58074) | Upstream CUDA-scoped ring proposal and author's existing-preallocation clarification; deployment/measurement status remains separate. |
| V12 | [vLLM speculative decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/) | Official standard-sampler guarantees, numerical qualifications, and workload dependence. |
| T1 | [Native TensorFold README](https://github.com/ashhart/TensorFold/blob/09e2f7fa664016ec9647af43ec721be33f9fe44a/README.md) | Primary native documentation: parallel admission and engine limits. |
| T2 | [lane_host.zig](https://github.com/ashhart/TensorFold/blob/09e2f7fa664016ec9647af43ec721be33f9fe44a/zig/src/core/lane_host.zig), [lanes/engine.zig](https://github.com/ashhart/TensorFold/blob/09e2f7fa664016ec9647af43ec721be33f9fe44a/zig/src/core/lanes/engine.zig) | Primary loop and family/shared-round execution distinction. |
| T3 | [lane_host_fill.zig](https://github.com/ashhart/TensorFold/blob/09e2f7fa664016ec9647af43ec721be33f9fe44a/zig/src/core/lane_host_fill.zig), [lane_host_prefix.zig](https://github.com/ashhart/TensorFold/blob/09e2f7fa664016ec9647af43ec721be33f9fe44a/zig/src/core/lane_host_prefix.zig) | Primary admission/refill and prefix-wait behavior. |
| T4 | [prompt_cache.zig](https://github.com/ashhart/TensorFold/blob/09e2f7fa664016ec9647af43ec721be33f9fe44a/zig/src/core/prompt_cache.zig), [prompt_rules.zig](https://github.com/ashhart/TensorFold/blob/09e2f7fa664016ec9647af43ec721be33f9fe44a/zig/src/core/prompt_rules.zig), [cache_modes.zig](https://github.com/ashhart/TensorFold/blob/09e2f7fa664016ec9647af43ec721be33f9fe44a/zig/src/core/cache_modes.zig) | Primary exact-token/decoded-interval and cache-mode contracts. |
| T5 | [kv_pool.zig](https://github.com/ashhart/TensorFold/blob/09e2f7fa664016ec9647af43ec721be33f9fe44a/zig/src/core/kv_pool.zig), [Qwen prefix adapter](https://github.com/ashhart/TensorFold/blob/09e2f7fa664016ec9647af43ec721be33f9fe44a/zig/src/families/qwen3_5/cuda_prefix.zig), [GLM prefix adapter](https://github.com/ashhart/TensorFold/blob/09e2f7fa664016ec9647af43ec721be33f9fe44a/zig/src/families/glm/cuda_prefix.zig) | Primary pool, opt-in family sharing, state snapshots, and content-key contracts; no universal-family claim. |
| T6 | [TensorFold cache benchmarks](https://github.com/ashhart/TensorFold/blob/09e2f7fa664016ec9647af43ec721be33f9fe44a/docs/cache-benchmarks.md) | Project benchmark methodology/claims, not independent reproduction. |
| P1 | [Orca, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu) | Original iteration-level/selective-batching research. |
| P2 | [PagedAttention, 2023](https://arxiv.org/abs/2309.06180) | Original block-management/sharing research. |
| P3 | [Sarathi-Serve, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/agrawal) | Original prefill/decode throughput-latency research. |
| E1 | [PCAI owner evidence, §6](#pcai-owner-evidence) | Private/sanitized session consultation, 2026-11-05. Suggested retained records: `models/deepseek-v4-vision/README.md`, `reports/capacity-context-vs-parallelism_20261103.md`, `research/engines/upgrades.md` in the owner's serving repository. Those records were not independently reproduced here and do not publicly document the 417K incident. |
| E2 | [Spark owner evidence, §7](#spark-owner-evidence) | Private/sanitized session consultation, 2026-11-05; historical patched-artifact observations, not current-upstream certification. |
| E3 | ARM owner consultation, 2026-11-05; scope in §1.1 | Focused test counts, unfinished qualification, topology/UMA cautions, and agreement to this report-only file boundary. |

Citation ranges such as `[L1–L6]` refer to the corresponding rows in this register. Private consultations are attributed evidence, not public reproducible sources.
