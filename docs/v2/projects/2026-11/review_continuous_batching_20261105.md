# Continuous batching: independent design review

**Date:** 2026-11-05

**Status:** Both independent read-only reviews are recorded. Revision-3 author responses are in §7. The user selected documentation of a narrow policy proposal; formal adoption is still pending. Revision 3 has not received a third independent review. **No runtime finding or implementation/economy gate is marked fixed by these document edits.**

**Inputs:** [Design](design_continuous_batching_20261105.md) · [Research](research_continuous_batching_20261105.md).

## 1. Reviewer, scope, and snapshots

The user authorized the existing `batch-inference-reviewer` session (`01a1ad84-9250-7433`) to conduct an independent read-only adversarial review. The design owner is the sole writer in the isolated `feat/continuous-batching-design` worktree.

The review was returned on 2026-11-05, with a subsequent precision note about placement-epoch sequences and request-local LLEP assignment. The reviewer reported read-only source/history inspection, **no builds/tests/benchmarks**, and no re-verification of external vLLM, TensorFold or PCAI observations.

### Exact reviewed and revised inputs

| Document | Initial reviewed SHA-256 | Second reviewed SHA-256 |
|---|---|---|
| `design_continuous_batching_20261105.md` | `bf80e42efe18a60c2d4af56f58df48d5727295b48bdf768f497e47802a4120c1` | `3b12f8859035b0cf54333e6c3c36386a93e4de4eb1d31cc1a2343c072dbe45a1` |
| `research_continuous_batching_20261105.md` | `c40ae90dfafaa84db2aec3f3ae6de6f283ac1f7e307efbea2b639fe35c5dde4f` | `eef1ac3fc9051efb9479fe0c73fe5610eabc242f81e7f36918ae5281a5c41c68` |

Original finding locations refer to the **initial reviewed document**, not revision-2 line numbers. `D` means design and `R` means research. Revision-2 gate identifiers G1–G9 are defined in D §15.

The documentation base is develop `ef1f54b0962d63531a9b33e06ca676e5124d05cc`. The initial research inspected `5e2a881aa99a9a4b428110a64b8ff64694c8b6b8`, an unmerged descendant 100 commits ahead. The reviewer stated that the cited develop facts were also checked at the descendant. The design owner independently checked the principal ABI, sampler, thinking, storage, retirement, snapshot and ownership seams, plus ancestry/blob identities; this is not a claim to have rerun every source audit or any runtime qualification.

No changes to the ARM checkout, runtime files, services, deployment, or build configuration are part of this review response.

## 2. Initial verdict and interpretation

The reviewer found the architectural direction coherent: one shared runtime, generation-qualified slots, distinct response/execution/publication leases, device-owned GPU control, PMA as physical authority, and explicit rejection of shortcuts.

The design was **not implementation-ready**. The reviewer identified CB-01, CB-02, CB-03, CB-05, CB-06, CB-07, CB-12 and the CB-14 feature matrix as decisions/contracts to settle before code.

Priority meanings:

- **P0:** central claim blocked or a forbidden path possible as written.
- **P1:** binding invariant or prerequisite missing.
- **P2:** clarification or explicit economy/implementation gate needed before code.
- **P3:** evidence/provenance correction.

Disposition vocabulary below:

- **Revised; gate open:** the document now states the required contract, but implementation/proof remains open.
- **Document corrected:** a source description or wording error was corrected; no runtime behavior is thereby qualified.
- **Accepted with precision:** the underlying concern is accepted, but an overbroad inference or proposed mechanism is narrowed explicitly.

## 3. First-pass findings and revision-2 dispositions

These are the owner's revision-2 responses. The second review in §6 supersedes any claim of documentary resolution where it identifies a residual.

### CB-01 · P0 · Design defect: undefined host boundary and retained host control

**Original location:** D §4 diagram, §5/§5.1/§5.2 and §8. **Evidence:** `ChatCompletionHandler.cpp` host thinking-budget loops use one-token `step_budget`, `forceDecodeToken`, and host-counted maintenance calls; `OrchestrationRunner.h` exposes global single-request MPI setters. See S1/S2.

**Failure:** publication-sized reasoning counts either overshoot a budget or force the whole cohort to one-token host round trips. Forced tokens have no admitted ingress category. A host `RunningTransaction/Committed` request machine risks mirroring GPU progress.

**Disposition — revised; G1/G2/G7 open:** D §4 separates host lease facts from device slot phases. D §5 defines permitted admission/cancel/credit/policy payloads, terminal output and permitted tickets, and forbids host inputs derived from decoded progress. Stop/think matching, budget enforcement and prescribed injection are request-owned execution state. Host text framing is presentation-only. Policy rows are installed for a new lifetime, not a live sampler update channel. Host SSE-token counting no longer defines maintenance cadence in the target.

The current ticket's narrowly authorized cadence metadata is distinguished from token contents/live model state; it does not justify a general progress mirror. Actual native publication semantics still require approval and proof.

### CB-02 · P0 · Blocking gate: cohort-uniform generation ABI

**Original location:** D §2.1, §3.2 and §5. **Evidence:** `DeviceGenerationContract.h`, `DeviceGenerationInitialization.h`, and `OrchestrationRunner` cohort/budget/transaction-zero checks. See S2/S3.

**Failure:** new C cannot be initialized independently of continuing A; one cohort epoch does not establish safe generation-qualified slot reuse; budgets remain common. Embedding `seq_idx` in captures would multiply topology variants by slot assignment.

**Disposition — revised; G2/G8 open:** D §6 sketches per-row admission kind, limits, law/history, seed, stop/think and MTP learner state. Serving-instance identity, at-least-64-bit non-reused slot generations, window/catalog/workspace identity and participating-binding authentication are required. Slot addressing is device data. The fixed-size authentication encoding and actual backend ABI remain unproved; the sketch is not a binary specification or capability claim.

### CB-03 · P0 · Design defect: deferred numerical contract

**Original location:** D §3.2, §6 and §11. **Evidence:** routed-prefill stable-window policy in `OrchestrationRunner.cpp`, plus the change in arithmetic/dispatch introduced by chunking, aggregate expert M, replica selection and placement. See S2.

**Failure:** fused prefill/decode can change decode dispatch; load-dependent chunks can change attention/recurrence accumulation and harvested state; pooling LLEP windows changes replica assignment; dynamic residency can change arithmetic during a request.

**Disposition — accepted with precision; G3 open:** D §7 compares to a solo reference with the same request-local chunk/control policy **and placement-epoch sequence**, preserving default dynamic movement rather than silently freezing it. The first shape alternates complete prefill/decode transactions. Chunk/checkpoint grids depend on request/configuration, not spare batch capacity. LLEP assignment itself must be request-local or every eligible replica must be proven byte-equivalent over its full live-M dispatch. Alignment alone is insufficient.

No weaker numerical contract is introduced for difficult topologies. Whole-output reproducibility also holds the sampling/MTP control trace fixed; it is not confused with the distributional property of a different speculative policy. These distinctions incorporate the reviewer's follow-up clarification.

### CB-04 · P1 · Wording/research defect: RNG coordinates

**Original location:** D §6 and R §2.2. **Evidence:** `resolveRequestBatchedStochasticSeed`, position/purpose sampling helpers, and common `active_sampling_params_`. See S2.

**Failure:** mixing RequestId/slot into an explicit seed changes draws across otherwise identical submissions. A lower-level seed array was overstated as independent per-request sampler admission.

**Disposition — document corrected; G2/G3 runtime proof open:** D §7.2 specifies `(resolved seed, logical output position, draw purpose)`. Request/slot/generation/ordinal are excluded. Seed-zero entropy is resolved once. R now distinguishes the seed-array foundation from the public batch setup's common law/explicit seed.

### CB-05 · P0 · Blocking gate: captured scalar laws and rejected default penalties

**Original evidence:** captured `SamplerStage` law and one-row history view in `DeviceGraphOrchestratorGeneration.cpp`; graph invalidation when sampler identity changes; batched penalty rejections; Qwen model-default presence penalties; GPU-MTP DRY rejection. See S2/S4.

**Failure:** default requests can be excluded from batched verification; differing laws require recapture or hidden exclusive cohorts. Disabling a default penalty does not qualify the intended path.

**Disposition — revised; G2/G7 open:** D §6 requires row-wise runtime law tables and independent penalty/sequence history on CPU/CUDA/ROCm. A vocab-sized INT32 count bank is an explicit BOM input; it does not supply DRY sequence history by itself. Graph identity includes addresses plus schema, geometry, precision, generations, launch policy and every truly embedded value—not addresses alone. Compatibility is typed/minimal; request laws cannot be convenient serialization keys.

The complete target matrix explicitly requires closing the GPU history gap. That is additional unimplemented work, not a claim current GPU DRY is supported. The penalty/prefix-off first experiment cannot certify default serving.

### CB-06 · P1 · Design defect: assumed paging and PMA logical grants

**Original location:** D §4, §6 and §7. **Evidence:** full-context per-slot ring allocations, copy-based prefix import, PMA allocation claims tied to backing allocation, and default output limit equal to remaining context. See S1/S5.

**Failure:** fixed-slot “growth admission” is only slot admission; a new page free-byte counter could duplicate physical accounting; live restored prefixes are private copies; default full-context requests do not gain fictitious paging capacity; independent rank grant order can diverge.

**Disposition — revised; G4 open:** D §9 names fixed-slot/full-context as the first physical envelope, with its own economy gate and honest default-output semantics. Cache-resident backing and private live copies are charged separately. Serving-owned host buffers are a proposed BOM obligation, not a claim every HTTP allocation is already tracked.

Paged KV is a separate unqualified mode requiring native operations for all formats/backends, a typed PMA-owned logical grant/retirement primitive, and ordered topology-wide admission. PMA's existing physical claims are not relabelled as that primitive.

### CB-07 · P1 · Design defect: reset/restore can clobber shared runtime state

**Original location:** D §3.1. **Evidence:** `IInferenceRunner` reset taxonomy and graph-builder reset/model-runtime prefix archive hooks. See S6.

**Failure:** C's prefix snapshot or B's retirement can restore/reset placement or histograms used by A.

**Disposition — revised; G5 open:** D §9.3 makes slot/transaction/domain classification a prerequisite, not an implementation-time comment. Mutable domain placement/maintenance is excluded from slot snapshots; immutable epoch compatibility remains. Request-local observations can feed the domain authority without acquiring reset authority over it. Existing reset/harvest interfaces require real changes and an exhaustive field inventory.

### CB-08 · P1 · Design defect: shared publication lease and framing credits

**Original location:** D §4.1 and §8. **Evidence:** whole-window `DeviceGenerationTerminalResult` materialization and `sink.write` inside the admitted inference job. See S1/S6.

**Failure:** one slow client pins staging required by the next batch; tool/reasoning/UTF-8 buffers escape output credit accounting.

**Disposition — accepted with precision; G1/G6 open:** D §10 separates GPU staging, host batch staging and detached request queues. GPU staging retires on its actual transfer/source-consumer event; host staging still waits for CPU demux readers. Neither waits on a socket. Credit covers the maximum complete-window commitment plus every framer/encoding/retained-copy obligation. Independent writers never run on the submission owner.

“Reclaim on completion” alone is insufficient if another producer/CPU reader still borrows the storage; the revised wording requires the actual last consumer.

### CB-09 · P1 · Design defect: cancellation requires model progress; terminal race

**Original location:** D §4.2. **Failure:** all-credit-blocked slots prevent another model transaction, so cancellation/retirement never runs; EOS and later cancel double-publish outcomes/usage; draining a never-reading client can hang shutdown.

**Positive seam:** `IKVCache::SequenceRetirement` exists, but covers only part of the required retirement. See S7.

**Disposition — revised; G6 open:** D §10 requires a lifecycle-only drain and separately reserved terminal/control receipt capacity independent of normal output credit. First committed execution terminal in transaction/lifecycle order wins; later valid cancellation is `AlreadyTerminal`. Execution and delivery outcomes are separate exactly-once values. Host I/O deadlines resolve transport retention without becoming inference deadlines. The complete composite lifecycle still needs implementation and adversarial tests.

### CB-10 · P1 · Blocking gate: ragged live collectives

**Original location:** D §3.2 and §9. **Evidence:** native prefix-row geometry excludes arbitrary ragged requests; `NativeAllreduceRequestRows` lowers each bank separately. See S8.

**Failure:** treating a ragged request-major bank as one live prefix sends padding or omits data. Per-bank lowering preserves extents but grows the collective count and may be uneconomical.

**Disposition — revised; G9 open:** D §12 requires device compaction/scatter or a proved ragged native protocol, and comparison against per-bank lowering. It measures actual bytes, launches, latency and arithmetic order. The reviewer's suggested request-count crossover is treated as a hypothesis, not an observed universal threshold. Compact messages can also change native reduction selection, so byte parity needs proof.

### CB-11 · P1 · Topology/scope gate

**Original location:** D §9. **Evidence:** single-request MPI command vocabulary and explicit batched-prefill exclusions for PP/global/multi-rank execution. See S2.

**Failure:** one globally in-flight transaction does not establish PP utilization. Pipelined requests need membership consensus, slot-qualified commands and exclusive arenas.

**Disposition — revised; G9 open:** the first engineering envelope is one endpoint per backend. PP, cross-rank, rank-local TP and multi-domain/heterogeneous extension paths are explicitly unqualified until their full contracts pass. D §12 requires a real PP microbatch schedule and at most one in-flight transaction per request. No narrow experiment authorizes removing previously supported application topology behavior at cutover.

### CB-12 · P1 · Design defect: second orchestration stack

**Original location:** D §2. **Evidence:** existing architecture assigns request lifecycle, sampling, MTP and MPI followers to `OrchestrationRunner`; application modes already call it. See S2/S10.

**Failure:** an engine above the runner duplicates authority; leaving scalar benchmark/frontend loops behind means production and measurements use different paths.

**Disposition — document corrected; G7 implementation gate open:** the competing `ContinuousBatchEngine` owner is withdrawn. D §3 retains `OrchestrationRunner` and gives transport/preparation adapters no model lifecycle authority. All modes must use the new request seam before retiring superseded scalar machinery. The Mermaid flow shows control and data ownership end to end.

### CB-13 · P2 · Economy gate: transaction pipeline

**Original concern:** completion/demux/ingress/submission can create a per-window bubble; deeper pipelining creates stale-command and arena hazards.

**Disposition — accepted with precision; G8 open:** D §8 chooses one in-flight model transaction per domain/request, no speculative next model submission, and two publication banks. Slot reuse depends on retirement receipts/events, not a blanket number of windows. A two-window lag may be relevant to a particular future pipelined protocol, but is not adopted as a general safety proof. Bubble fraction and actual cancellation/admission latency remain unmeasured.

### CB-14 · P1 · Delivery defect: scalar retirement without parity/accounting

**Original location:** D §2, §6 and §10. **Evidence:** the frontend admits model penalties, DRY according to backend capability, thinking budgets, tool choice, runtime summaries, token IDs and stream usage; summaries are runner-global and verifier counters already have per-request fields. See S1/S4/S6.

**Failure:** replacing scalar serving can reject existing supported traffic or disguise it as exclusive cohorts. Shared work has undefined per-request attribution.

**Disposition — revised; G7 open:** D §13 adds feature × CPU/CUDA/ROCm × ordinary/MTP obligations, expanded over topology/format/control-policy axes. Every target cell is still unqualified. Existing response-form restrictions remain explicit. Revision 2 elects to close GPU DRY history for the complete target rather than falsely claim parity from a count table. Shared wall time/physical bytes cannot be duplicated into every request and summed. Actual frontend inventory and production tests are still required.

### CB-15 · P2 · Preparation clarification

**Original location:** D §2.1. **Evidence:** current `setupInference` combines rendering/tokenization/default merging with runner mutation and MTP admission. See S1.

**Disposition — revised; G7 open:** D §3.2 uses immutable policy snapshots, a proven read-safe tokenizer or dedicated bounded preparation owner, and validation before execution mutation/SSE success commitment. Preparation cannot call global sampler/stop/reset operations. Queue acceptance is not execution admission. No tokenizer concurrency safety is assumed from this wording.

### CB-16 · P2 · Prefix gate

**Original location:** D §11. **Evidence:** prefix harvest occurs at reusable prefill checkpoints; recurrent state cannot be rewound; `harvestPrefix` lacks slot identity; immutable fingerprints reject stale placement harvests. See S2/S6.

**Failure:** arbitrary chunks miss snapshots, harvest copies occur in capture, queued readers pin the cache indefinitely, or movement increases discards without being observed.

**Disposition — revised; G3/G5 open:** D §7/§9 align canonical grids, require slot-generation-qualified lifecycle harvest/restore outside model capture, pin only at execution admission, and forbid relabelling stale state under a new epoch. Concurrent hit/discard/checkpoint rates are economy evidence, not an assumption. This does not manufacture absent snapshots or disable movement to pass.

### CB-17 · P2 · Binding across domains

**Original location:** D §2.1.

**Disposition — document clarified; G2/G9 open:** one serving-instance/slot/generation identity is shared by participants; each stage/shard/tier derives its physical mapping. Expert-only tiers do not acquire their own sampler/controller. Instance identity additionally rejects handles/tickets from a replaced runtime. Multi-device command consensus remains unimplemented.

### CB-18 · P2 · HIP retained catalog

**Original location:** D §5.1. **Evidence:** the current fixed-size ticket selects one next depth and excludes raw token/position/controller state, with a narrow committed-token cadence field. See S9.

**Failure:** fused chunk/depth/maintenance combinations can multiply graph inventory and storage; per-row depths are not established by the current ticket.

**Disposition — revised; G2/G8 open:** D §8.1 requires an actual branch inventory and PMA graph/workspace BOM. Prompt-derived canonical chunk preparation is host-side input processing; live post-admission decisions remain on device. Alternation reduces one source of combinations but proves neither catalog size nor economical heterogeneous depth. No live-state array is added to the host ticket by implication.

### CB-19 · P3 · Revision and policy provenance

**Original location:** R §1.2/register and D provenance/replay requirements.

**Verified facts:** the inspected commit is an unmerged descendant 100 commits ahead, not an older “historical” develop revision. Fourteen of 25 literal linked paths differ; resolving the unchanged root AGENTS symlink adds the changed policy target, for 15 effective contents. The standalone policy header is descendant-only, but the enum already exists on develop in `IInferenceRunner.h`. The descendant adds replay/dashboard wording absent from develop.

**Disposition — document corrected:** R now pins all L-series source URLs to the inspected fork commit, whose GitHub commit metadata was checked. The policy target is linked directly. D explicitly adopts the extra replay/document-classification rules as design requirements rather than claiming they landed on develop. Source pinning is not runtime qualification. No source code was copied between worktrees.

### CB-20 · P3 · Research omissions and helper overstatement

**Original location:** R §2.2 and recommended test/owner extensions.

**Disposition — document corrected:** R adds the cohort ABI, captured scalar sampler/default penalty gap, synchronous SSE writes, fixed-slot/private-copy KV, single-request commands and positive `SequenceRetirement` seam. It distinguishes independent seeded rows from independent law admission. `MTPSpecRequestBatchOwner` is a per-call host helper/oracle, not a GPU serving controller; superseded machinery must be retired rather than extended into a second authority. External engine/operator observations were not re-certified by this local source review.

## 4. Pinned source register for the review

These links use the review's **develop baseline** unless expressly marked as descendant. Line references in the original message are navigational; symbols and exact commit identities own provenance.

| ID | Sources |
|---|---|
| S1 | [ChatCompletionHandler.cpp](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/app/modes/ChatCompletionHandler.cpp), [ServerMode.cpp](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/app/modes/ServerMode.cpp), [ChatCompletionHandler.h](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/app/modes/ChatCompletionHandler.h) |
| S2 | [OrchestrationRunner.cpp](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/execution/runner/OrchestrationRunner.cpp), [OrchestrationRunner.h](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/execution/runner/OrchestrationRunner.h) |
| S3 | [DeviceGenerationContract.h](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/execution/mtp/DeviceGenerationContract.h), [DeviceGenerationInitialization.h](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/kernels/common/DeviceGenerationInitialization.h) |
| S4 | [DeviceGraphOrchestratorGeneration.cpp](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestratorGeneration.cpp), [ModelGenerationPolicy.cpp](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/models/ModelGenerationPolicy.cpp), [MTPRequestSamplingPolicy.h](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/execution/mtp/MTPRequestSamplingPolicy.h) |
| S5 | [PhysicalMemoryAuthority.h](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/planning/PhysicalMemoryAuthority.h), [CPURingKVCache.cpp](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/kernels/cpu/CPURingKVCache.cpp), [CUDARingKVCacheTQ.cu](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/kernels/cuda/kvcache/CUDARingKVCacheTQ.cu), [ROCmRingKVCache.cpp](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/kernels/rocm/kvcache/ROCmRingKVCache.cpp) |
| S6 | [IInferenceRunner.h](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/execution/local_execution/orchestrators/IInferenceRunner.h), [DeviceGraphOrchestrator.h](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h), [DeviceGraphOrchestrator.cpp](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp) |
| S7 | [IKVCache.h](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/kernels/IKVCache.h) |
| S8 | [NativeCollectiveRows.h](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/collective/NativeCollectiveRows.h), [NativeAllreduceRequestRows.h](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/collective/NativeAllreduceRequestRows.h) |
| S9 | [SamplingMath.h](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/kernels/common/SamplingMath.h) |
| S10 | [Develop architecture](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/.github/instructions/llaminar-v2-architecture.instructions.md), [develop policy](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/.agents/AGENTS.md), [inspected-descendant policy](https://github.com/fank/llaminar/blob/5e2a881aa99a9a4b428110a64b8ff64694c8b6b8/.agents/AGENTS.md) |
| S11 | [OrchestrationConfigParser.cpp](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/config/OrchestrationConfigParser.cpp), [RankMemoryPlanInputs.cpp](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/planning/RankMemoryPlanInputs.cpp), [ExecutionPlanBuilder.cpp](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/execution/mpi_orchestration/ExecutionPlanBuilder.cpp), [BenchmarkRunner.cpp](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/utils/BenchmarkRunner.cpp), [SerializedInferenceExecutor.h](https://github.com/Llaminar/llaminar/blob/ef1f54b0962d63531a9b33e06ca676e5124d05cc/src/v2/app/modes/SerializedInferenceExecutor.h) |

## 5. Second-pass brief and proof boundary

The focused second review was asked to check the normative boundary, state machines, per-row ABI, numerical/placement/LLEP contract, fixed physical mode, output/lifecycle liveness, one-owner decision, feature matrix and branch/pipeline inventory obligations. It identified the residuals in §6, not runtime qualification.

The reviewer's smallest subsequent proof is preserved in D §14: first recorded controller replay with A continuing, B terminating and C replacing B with a new generation; then a narrowly scoped CUDA/ROCm captured experiment. Penalty-free/prefix-off experiments remain diagnostic subsets and cannot certify the default serving feature set. No such tests were run in this review response.

All G1–G9 gates remain open. The current action is document review, not runtime implementation, source admission, E2E certification, benchmarking, commit, or push.

## 6. Second independent pass: residual decisions and new findings

**Received:** 2026-11-05, message `c96587c0-a442-4a9e-8bcc-66315ad995a7`, reviewer `01a1ad84`. The reviewer verified the two second-reviewed hashes above and review-record hash `7d507cb38eaf8cb33c2b50fb8a6f98bc2636ae3ba3dd1301f01717e44b182d09` unchanged at both start and send time. The owner independently rechecked those hashes before editing this record. Source is develop `ef1f54b…` unless stated otherwise. No builds/tests or runtime/performance certification.

**Verdict:** direction coherent, still not implementation-ready. Documentary resolution with no residual: **CB-04, CB-05, CB-07, CB-10, CB-11, CB-15, CB-17, CB-19**. The remaining original IDs have residuals or follow-ups below: CB-02 maps to CB-24; CB-06 to CB-23; CB-13 to CB-18R/CB-25; CB-16 to CB-25/CB-26. All native G1–G9 gates remain open even for documentarily resolved findings.

### CB-01R · P1 · Owner-policy decision and two contract defects

**D §5/§8/§11.** Recurring cancellation/credit/grant ingress and non-streaming per-window terminal D2H are not clearly inside maxims 8/9's closed exception list; per-transaction host completion waits also touch maxim 11. Existing non-streaming code normally requests the remaining generation budget, not the streaming publication quantum (S1). This needs an **explicit owner-approved interpretation or proposed amendment**, not merely a native test claiming compatibility.

The reviewer proposes defining bounded serving transactions as complete generation units for both response forms, a closed typed ingress class, and one terminal wait per transaction, guarded by source sanitizers. Current project rules are not amended by this review or by a design assertion. Owner direction remains required.

Two additional fixes are needed: use device transaction counts for scheduling age rather than an undeclared host clock/priority channel; and unify reasoning phase semantics. The current text splitter recognizes template delimiters, including split tokenizations (S1). A device matcher needs a canonical token-level rule and must publish authoritative phase-close/injection positions for presentation. Forced positions override target sampling, reject drafts at those positions, and join G1/G2's MTP proof.

### CB-03R · P1 · Numerical trace and intrinsic windows

**D §7/§11.** Enumerate placement switches by logical position, prefix-restore boundary, canonical chunks, per-request window/transaction boundaries and depth decisions, forced-sequence positions and seed. Identify which fields are intrinsic to the request. `SamplingMath.h` skips depth learning for budget-limited transactions (S9), so workload-dependent window truncation changes the learner and seeded output.

Required correction: a fixed per-request window budget counted from that request's commits; scheduling may defer but never resize it. Use the same rule in streaming/non-streaming and solo/batch. State the no-movement single-endpoint token-byte guarantee for the same prefix boundary. Explicitly decide whether cold/hit checkpoint equivalence is proved or the prefix boundary remains part of the comparison trace. This is not resolved by assuming an arbitrary recorded depth trace after scheduling has changed it.

### CB-08R · P2 · Response-lifetime sizing and composite host state

**D §4/§9.1.** Detached outputs outlive execution slots, so response queues, framers and non-streaming accumulation are sized per **response lease**, not per resident slot. Host lifecycle is the product of response lease, execution lease and outstanding publication state: execution can be Released while the response is still Delivering.

Existing PMA owner reservations plus `claimAllocation` already cover physical host backing allocations (S5). Only logical grants inside materialized storage need a new PMA primitive. Their existence must not be confused with complete HTTP-buffer integration.

### CB-09R · P2 · Typed lifecycle-only operation

**D §6/§10.2.** The current generation commit treats zero newly emitted tokens as `DeviceGenerationError::EmptyTransaction` (S9). Lifecycle-only work therefore needs a distinct operation type that never calls generation commit with zero output. Vacant, prefilling and output-blocked rows are Inactive for generation; they are not fake empty generations. Host idle/submission decisions must use pending intents and outstanding grants, not a device-progress mirror (CB-21).

### CB-12R · P1 · Migration sequence

**D §1/§3.1.** Narrow initial topology, no lost existing support, and no retained scalar path cannot all be satisfied by a direct narrow cutover. Required sequence: **Phase A** moves every mode/topology to the per-row seam at capacity one, preserving behavior and passing existing gates plus parity. **Phase B** increases capacity only for explicitly qualified topology capabilities. Capacity one is the same implementation, not an error fallback; an explicit request for higher unqualified capacity is rejected, never clamped. Version any separately changed numerical/window policy rather than hiding it in the behavior-preserving refactor.

### CB-14R · P2 · Cutover parity versus new target features

**D §13.** Both current GPU-MTP and ordinary resident sampling reject active DRY (S3/S4). GPU DRY cells marked Required are new work, not merely preservation of current support. Split the cutover matrix from the complete-target matrix and identify which gate blocks cutover. Known existing restrictions stay explicit; this does not allow a newly rejected supported option.

### CB-18R · P2 · HIP selection timing

**D §8.** Select after ingress using a selector transaction and ticket wait before the selected model transaction, or select at t's tail and explicitly account for later ingress/grants taking effect one step later. The first option costs two host completions per selection/model cycle; the second creates branch, grant and cancellation-lag obligations. Choose and include the cost in the retained catalog and bubble budget. Neither is already approved by the unresolved CB-01R policy decision.

### CB-20R · P3 · Research wording drift

R's “slot generation wrap policy”, “Shared read-only prefix; private tails” and “mixed prefill/decode policy” need correction. The selected initial design fails closed on generation exhaustion, keeps private live restored prefix copies, and alternates complete transactions. Shared live-prefix/paged and fused-mode statements must be separately labelled future paths, not properties of the fixed-slot initial design.

### CB-21 · P1 · Output eligibility has no single authority

**D §4/§5/§10.** Device credit counters would receive host writes to scheduling state that the boundary forbids; host selection would need an undeclared eligibility channel. Define **one host-owned execution grant per slot-generation**, authorizing at most one complete request window and backed by reserved destination storage. Carry it through ordered typed ingress. Device `OutputBlocked` is derived from absence of a usable grant, never directly assigned by the host. Credit return itself is a host buffer-lifecycle event, not a write to model state.

### CB-22 · P1 · Prefill transaction composition

**D §7.1/§8/§11.** Choose one request chunk or an aggregate of independent chunks per prefill transaction. Aggregation extends the numerical invariance burden to combined prefill M. One-request prefill retains solo prefill M but has burst-TTFT/throughput costs. Scope decode/verifier invariance to all admitted slots × verifier-row geometries; do not pretend the selected composition proves an unimplemented alternative.

### CB-23 · P1 · Serving-capacity authority and obsolete benchmark path

**D §3.1/§9.1/§13.** `--batch-size`, `--mtp-max-request-batch`, the HTTP capacity constant, terminal-head policy and fixed-cohort benchmark paths expose disconnected capacity meanings (S11). Define a typed serving-capacity input resolved by planning/PMA, sealed into startup identity and consumed by dependent policies. Separate resident slots from queue/response leases without adding independent authorities. Retire or rename overlapping CLI/config fields, fixed-cohort APIs and benchmark plumbing together. Specify that the replacement benchmark measures the installed serving seam rather than a special batch API.

### CB-24 · P1 · Host cannot authenticate device-selected participation

**D §6.** Host validation of the “participating slot-generation set” needs device selection knowledge and becomes a mirror. Bind host-known serving instance, input/window/catalog/workspace generations and the digest of the admitted resident binding snapshot owned by host leases. A device-selected participation digest is for participant agreement only, following the intent of `hasSameDispatchDecision` (S9), never host inference or validation of the selected set. Device checks selected rows against actual generation/state.

### CB-25 · P2 · Prefix/lifecycle I/O concurrency and ownership

**D §8/§9.3/§10.2.** Sequential multi-GB restore/harvest stalls all requests; concurrent I/O needs dedicated exact copy streams, exclusive staging, and events against slot writers/readers. Choose the concurrency contract and measure bandwidth contention and stalls. “One in-flight model transaction” does not itself settle copy concurrency or justify ignoring transfer ownership.

### CB-26 · P1 · Missing prefill-to-decode handoff state

**D §6/§7.1/§9.3.** Other requests execute between a last prefill chunk and first decode. Current `prefill_logits_ready_`, `ready_mtp_condition_`, `pending_mtp_condition_*` and `prelaunched_mtp_first_sidecar_*` are runner-global; decode consumes prefill logits directly (S2).

Terminal logits/hidden state and all pending/ready/sidecar handoff ownership must persist per slot across intervening work. Include at least `4 × vocabulary_size` bytes for FP32 logits per slot plus actual hidden/sidecar storage and consumer events. Prove no overwrite, duplicate sampling/emission/history update or stale-generation consumption. A request-specific sampled handoff is not automatically reusable cross-request prefix state.

### Second-pass proof boundary

Before code, settle CB-01R(a), CB-03R, CB-12R and CB-21/22/23/24/26. Before their native stages, settle CB-08R/09R/14R/18R and CB-25. The smallest composed recorded replay in D §14 additionally needs intrinsic windows, slot-local handoff state and grant-derived output eligibility. The reviewer did not authorize or execute that proof.

The owner independently confirmed the budget-limited learner/empty-generation checks, global handoff fields, differing streaming/non-streaming budgets, PMA reservation primitive, terminal-head capacity dependency, split capacity inputs and benchmark-only calls. These source reads support the correction; they do not settle policy approval or native feasibility.

## 7. Revision-3 author response

### 7.1 User direction and unchanged authority boundary

After the second review, the user selected **“Document narrow proposal”** from a structured choice. Its stated scope was to retain the bounded-window design and document typed lifecycle/grant ingress and terminal completion waits, including the extra HIP selector completion, **while leaving current project rules unchanged and runtime work blocked pending formal policy adoption**.

This is authorization to prepare a policy-amendment proposal, not an adopted interpretation of maxims 8/9/11. No change to `AGENTS.md`, runtime code, tests, services or deployment follows from that choice. D §5 now separates the formal adoption decision from native proof and requires narrowly scoped source-sanitizer enforcement if implemented.

### 7.2 Concrete proposed dispositions

These are author-proposed contract decisions, not a third review's approval. The initial no-residual findings remain recorded in §6; every native G1–G9 gate remains open.

| Finding | Revision-3 response | Remaining qualification |
|---|---|---|
| CB-01R(a) | D §5 records the user-requested narrow policy proposal, enumerated ingress and terminal/ticket completion boundary. Non-streaming window results are explicitly part of the proposed change. | **Formal adoption pending**, then G1 native/sanitizer proof; no implicit exemption. |
| CB-01R(b,c) | Device-counted saturating age; canonical marker matching over actual token pieces, authoritative output-coordinate metadata, device-owned forcing and forced-boundary learner skips. Host retains only bounded presentation fragments. | Tokenizer/marker, MTP, partial-fragment and frontend semantic parity at G1/G2/G7. Historical disagreements must be versioned/resolved, not called a pure refactor. |
| CB-03R | D §7 enumerates intrinsic versus external trace fields and fixed request-local window boundaries. No-move single-endpoint tokens must match solo under the same plan/request/prefix boundary/control trace. Depth decisions are checked, not assumed after the fact. | G3 proof; different prefix boundaries are not covered by a new cross-boundary byte claim. Changing the old non-streaming budget policy is separately qualified/versioned. |
| CB-08R | D §4 models response × execution × publications/grant ownership. D §9.1/§10 sizes detached output, framing and accumulation per response lease and uses existing PMA owner-reservation/backing claims. | G4/G6 integration, capacity and event/demux ownership proof. |
| CB-09R | `LifecycleSelect` never invokes empty generation; non-decode rows are Inactive. Idle uses outstanding grants/intents or actual storage-ownership wake sources; eligible granted work cannot disappear into Idle. | G6 native liveness and rejection of `EmptyTransaction` misuse. |
| CB-12R | Phase A migrates every current mode/topology at capacity one, preserving its current policy. A separate capacity-one policy-qualification step precedes Phase B capacity increases. One implementation remains; higher explicit unqualified capacity fails, not clamps. | G7 all-mode/topology parity, then G1/G3 shared-policy qualification and per-capacity native/economy gates. |
| CB-14R | D §13 separates cutover preservation from additional target features. Current GPU DRY rejection is explicit; native GPU DRY is new complete-target work, not a claimed existing parity feature. | G7 blocks cutover on newly lost support; G2/history and feature gates block complete-target claims. |
| CB-18R | HIP uses selector-first after sealed ingress. One selector plus one model graph has two terminal observations; further native MTP branch tickets are also counted. | G1 policy adoption and G8 actual catalog/receipt-count/bubble proof; no performance forecast. |
| CB-20R | R now names no-wrap exhaustion, private live restored prefix copies and alternating transactions. Shared live-prefix/paged and fused alternatives are not attributed to the initial mode. | Documentary correction only; corresponding runtime modes remain unimplemented. |
| CB-21 | One host-backed single-use grant per slot-generation authorizes one intrinsic output window. Credit drains stay host-owned; device eligibility derives from the grant and readiness. A grant may span that request's prefill chunks without authorizing additional output windows. | G2/G6 replay, destination reservation, stale/duplicate grant, no-credit cancellation and retirement proof. |
| CB-22 | One canonical request chunk per prefill transaction. Prefill M remains solo M; decode/verifier invariance covers the full admitted slots × verifier-row/expert-M range. Burst TTFT/prefill economy is explicitly at risk. | G3 numerical and G8 economy proof; aggregated/fused prefill remains a distinct unqualified optimization. |
| CB-23 | D §9.4 defines one typed capacity request/resolution for resident S, response R and queue Q with PMA BOM/identity consumers. Overlapping CLI/config/benchmark paths retire together; proposed new flag names are clearly unimplemented. | G4/G7 actual field inventory, schema migration, dependent head-policy binding, physical admission and serving benchmark proof. |
| CB-24 | Host validates a named immutable host-owned binding snapshot and known epochs/catalog/workspace. A device participation digest is for participant agreement only, never host selected-set validation. | G2/G9 native encoding/authentication/consensus and stale snapshot rejection. |
| CB-25 | D §9.6 chooses disjoint-slot I/O overlap with persistent copy lanes/staging. Source freeze ends on exact snapshot-copy retirement; slower tier I/O borrows detached state. An unready C cannot stall unrelated A through a whole-domain wait. | G5 selective event-aware readiness, retirement and full BOM; G8 copy contention/stall evidence. Serialized copies are not an automatic substitute. |
| CB-26 | D §6/§9.3/§9.5 makes terminal logits/hidden state and ready/pending/sidecar handoffs persistent slot state, with native events and `4 × vocabulary` FP32-logit bytes plus real hidden/sidecar BOM. Sample-dependent state is excluded from reusable prefix snapshots. | G2/G4/G5 interleaved overwrite/duplicate-commit/stale-generation and storage proof. |

Two precision corrections accompany these decisions. The current HTTP constant 32 counts total accepted responses including active and queued, not 32 extra queued requests. And matching an original scalar token stream cannot be assumed after changing its window/phase policy: the ownership-only refactor preserves the old policy; shared-policy adoption gets a distinct version and reviewed evidence at capacity one. Neither is an excuse to rebaseline failing tokens automatically.

### 7.3 Revision-3 artifacts

| Document | SHA-256 |
|---|---|
| `design_continuous_batching_20261105.md` | `68923ff36acb23f0cb7ccb75b1342e53f38b1c8a5eec5218bf59e90613debcfb` |
| `research_continuous_batching_20261105.md` | `1393cbede14734e981806046a61b1f2377a54b81470262206a74c34e18ac19fe` |

Research adds an explicitly develop-pinned controller/capacity source group rather than presenting its changed planner as the inspected descendant's policy. It now has 41 evidence groups; no new external engine measurement or re-certification is claimed.

**Revision-3 pre-publication handoff:** the proposals were written, but policy adoption, ABI/catalog details, native correctness, feature parity and economy remained unqualified. No third independent review, runtime changes, builds/tests, benchmark execution, commit or push was part of that editing/review phase.

## 8. Documentation validation

Before publication, revision-3 structural/provenance checks passed:

- Both current artifact hashes match §7.3; the earlier reviewed hashes remain historical inputs, not current-file assertions.
- 62 exact local Git source objects behind pinned repository URLs exist; nine relative links/explicit anchors resolve. This is not a fresh network audit of every external URL.
- Markdown tables, fenced-block balance and whitespace checks pass; Mermaid was not renderer-validated.
- The register contains 41 unique evidence IDs; the record preserves CB-01 through CB-26 and the eight second-pass residual IDs. All nine G1–G9 gates remain explicitly open.
- At that pre-publication validation, the isolated branch had only the three intended untracked Markdown files, no tracked diff from its develop base, and no new commit. No batching documents remained in the ARM checkout.

These checks validate the records, not the proposed controller, capture policy, numerical contract or performance. External engine/operator measurements were not rerun.
