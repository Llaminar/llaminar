# ARM performance / memory review — idea index

Created: 2026-11-07 (review carried out by read-only session `01a1b0ec`).
Reviewed against: `codex/qfn-ngram-continuation-arm64` @ `be6658c38`
(rebase onto `feat/qfn-ngram-continuation` @ `2f01249c9` complete; 28 ARM
commits replayed).

**Current anchors (2026-11-09):** arm branch PR #43 head = **`5d4e3716a`**
(126 commits @ base `46c35f15b`; rename `NativeVNNI`->`NativeQuantised` in
`a0fd16d04`; R1 fast-forwarded in: 425002104 -> 5d4e3716a; no rebase,
reviewed tip exact; PR body updated). R2 + CUDA spill commits stack on
`5d4e3716a`. Ideas-7+25 = `b199ccfa0` (on 425002104) — rebasing onto
`5d4e3716a` in progress.

Scope: performance and memory improvement ideas gathered during the review of
PR #32 (`feat(cpu): add native ARM64 inference kernels`) and PR #35
(`feat(cpu): native ARM64 NEON execution for MXFP4`), plus the DGX Spark
(GB10) context in the Linux ARM handoff. One markdown file per idea.

This is a review artifact, not an implementation ledger. The ARM implementation
session remains the sole writer of production source; the tester session runs
all builds/measurements. Nothing here is a measured economy claim. File:line
references are as of the head above and may drift.

## Status legend

- **Open** — not implemented at the reviewed head.
- **Partial** — a related change landed; the specific idea, or a remaining part
  of it, is still open.
- **Addressed** — implemented at the reviewed head; commit(s) cited. The entry
  is kept for context and any residual verification.

## Index

| # | Idea | File | Status |
|---|---|---|---|
| 1 | Local kernel-benchmark A/B harness | `01-local-benchmark-harness.md` | Partial (Perf__ suite exists for projection; gaps below) |
| 2 | Shared CPU/GPU (UMA) memory admission | `02-uma-shared-memory-admission.md` | Partial (joint pool implemented `80985c08e`; live GB10 page-cache admission gap open) |
| 3 | Move projection metadata validation out of the K loop | `03-projection-metadata-validation-hoist.md` | Open |
| 4 | Wider decode column panel + activation reuse | `04-decode-column-panel-activation-reuse.md` | Open |
| 5 | Heterogeneous (X925/A725) core scheduling | `05-heterogeneous-core-scheduling.md` | Partial (dynamic tile claiming; task-count-dependent placement evidence; typed policy pending) |
| 6 | I8MM (SMMLA) row-pair dot-product | `06-i8mm-row-pair-dot.md` | Implemented + measured (kernel 1.01–1.19, engine +1.3–1.7%, M=1 guard holds, no spills; Q6_K ~0 gain; tip cd55f6b27) |
| 7 | Hot-path allocation / cache cleanup (RoPE, RMSNorm, gamma TLS) | `07-hot-path-allocation-cleanup.md` | Open |
| 8 | ARM prepared-weight layout (drop dead compensation bytes, compact expansions) | `08-arm-prepared-weight-layout.md` | Open (ABI change; last) |
| 9 | Non-volatile rounding barrier evidence | `09-rounding-barrier-evidence.md` | Open (evidence required first) |
| 10 | FP32 RMSNorm scalar sum order + sanitizer warnings (issue #36) | `10-rmsnorm-fp32-sumsq-order.md` | Addressed (`3086894b5`, `ac3289bd9`); residuals listed |
| 11 | Attention validation hoist + integer-dot range proof | `11-attention-validation-and-integer-dot-range.md` | Open |
| 12 | Cache-topology-derived tiling / prefetch on GB10 | `12-cache-topology-tiling.md` | Addressed (overlay table retired `264970e14` after head A/B confirmed regression; methodology in idea 20) |
| 13 | Fewer OpenMP regions per decode step | `13-openmp-region-count-and-wakeup.md` | Partial (spin `27585692d` + verified bundle `408ff90e5`; remaining regions open) |
| 14 | Dense-model MTP verifier economics (27B) | `14-dense-mtp-verifier-economics.md` | Base-dependent (new base: MTP 5.45–5.55 vs no-MTP 5.28 → MTP wins; old base: MTP lost) |
| 15 | NUMA-less and cache-unpublished Linux ARM hosts | `15-numa-less-hosts.md` | Partial (memory attribution `a572c3c9c`) |
| 16 | MoE routed experts, GDN projection and LM head in CPU decode | `16-moe-expert-and-lm-head-decode.md` | Open (bundle part closed/verified; GDN and LM head open) |
| 17 | GB10 CPU decode bandwidth gap (post-merge follow-up) | `17-gb10-decode-bandwidth-gap.md` | Open (full cross-engine matrix: llama.cpp CUDA 1063/28.8 vs Llaminar CUDA 319/22.8; see idea 24) |
| 18 | GDN grouped verifier + publication economics | `18-gdn-grouped-verifier-economics.md` | Confirmed (GB10 grouped-vs-serial < 1.0 for all M; M=4 0.447 @20T) |
| 19 | Prefill vs decode gap on GB10 (short-prompt effect) | `19-prefill-vs-decode-gap.md` | Open (prompt-length sweep planned) |
| 20 | Overlay collection methodology (microbench wins that lose in-model) | `20-overlay-collection-methodology.md` | Open (new rule: in-model collection or per-format E2E validation) |
| 21 | CUDA GB10 (sm_120/sm_121) kernel register spills under the zero-spill guard | `21-sm121-cuda-kernel-spills.md` | Open (sm_120 worse: 45 instantiations; blocks GB10 GPU + UMA validation) |
| 22 | FA2 tile-tournament policy tail on GB10 | `22-fa2-tile-tournament-policy-tail.md` | Open (p95 regret 8.926% > 5%; tq4_tq4 17.6%) |
| 23 | qwen4exp HyperConnection auto-planning classification gap | `23-qwen4exp-hc-auto-planning.md` | Open (auto `plan` rejects HC models; authored `--expert-tier` works) |
| 24 | GB10 GPU prefill gap vs llama.cpp (3.3×) | `24-gb10-gpu-prefill-gap.md` | Open (dig: ~80% is bucket-pipeline overhead, not kernels; see idea 25) |
| 25 | Prefill capture/bucket-pipeline overhead reduction | `25-prefill-bucket-pipeline-overhead.md` | Open (blocked on tooling: nsys missing in uma3 image; evidence-based split filed; 15-min rerun once nsys lands) |
| 26 | Single-rank OpenMP binding on GB10 is A725-first | `26-single-rank-openmp-binding.md` | Addressed (a7f719dfb binding-only SHA; proofs pass; pending Main cherry-pick onto PR #43) |
| 27 | Fused NativeVNNI evidence cells fail on ARM (Pairwise/WideRows) | `27-fused-nativevnni-evidence-cells.md` | Addressed (fixed on PR #43; cells pass) |
| 28 | IQ4_XS M=1 multi-thread regression after the metadata seal | `28-iq4xs-m1-multithread-regression.md` | Open (investigation; window b; fix-in-R1 or cleared+tracked) |

Measured numbers for all ideas: `00-measured-baseline.md` (GB10 serving,
stage attribution, spin sweep, thread scaling, overlay statistics).

## Cross-cutting notes

- **Exactness gate:** every native-path change must preserve byte-identical
  serial/MTP output. Kernel-level byte authentication with independent oracles
  is the standard, not tolerance-based comparison.
- **MXFP4 coordination:** any shared change to the NEON projection tile or
  `NibbleMap` decode must be coordinated with the PR #35 lane so there is one
  change certifying all `NibbleMap`/scale representations.
- **MTP/admission test lanes:** per the project owner (2026-11-07), the MTP/
  admission test lanes are mostly for exhaustive debugging. Use them **later,
  on demand, as a perf-debug helper** — e.g. for verifier-depth and acceptance
  measurement when idea 14 (dense MTP verifier economics) or idea 6 (I8MM) is
  implemented. They are **not** part of the current work, and **not** intended
  to become mandatory CI/CD gates. Unit + model-free `ProductionTestPreflight`
  remain the pass gate.
- **Mac vs Spark:** Mac (M1 Max, 8P+2E, no libnuma, no production engine
  configure) is the fast correctness loop; DGX Spark (GB10, 10x X925 + 10x
  A725, one NUMA node, CPU/GPU UMA) is where scheduling, memory and bandwidth
  claims must be measured. Worker sweeps must cover 1..20 physical cores with
  X925-only / A725-only / mixed placements measured separately.
- **Later this cycle:** one natural checkpoint for these ideas is after the
  complete engine Unit/preflight gate on ARM; kernel-only wins then need
  full-inference confirmation.

## Remote/home note

The tracker and all lane branches are now mirrored on **Llaminar/llaminar**
(upstream): `codex/arm-perf-review`, `codex/i8mm-typed-placement`, plus the
frozen `codex/qfn-ngram-continuation-arm64`. The personal fork that hosted
them is being deleted by the operator — use upstream only; SHAs remain the
source of truth.

## Startable now (2026-11-08; independent of the i8mm merge)

Measurement/qualification on GB10 (current engine + hosts):
- 17 bandwidth sweep (1..20, per core class) · 19 prompt-length sweep (128/512/2048) · 2 live UMA joint-pool + page-cache admission validation (Qwen3.8; spill fixes landed) · 18 GDN kernel-vs-capture cost split · 8 memory sizing (prepared vs GGUF, dead bytes, peak RSS).
Device-free code (no conflicts):
- 23 qwen4exp HC auto-planning classification · 7 allocation cleanup (RoPE/RMSNorm/gamma TLS incl. stale-risk fix) · 11 attention validation hoist + integer-dot range proof · 14 depth-controller analysis (window_not_ready, MTPDepthDefaults).
CUDA / GB10 GPU (part of the ARM story):
- 21 remainder (BK256 non-ordered; tensor_prefill split — spec signed off) · 25 (PerfStats bucket counters + nsys-in-uma3 first; single-bucket/async path after) · 22 FA2 per-codec tournament prep.
After i8mm merge (same tile / post-merge):
- 4 wider panel + activation reuse (MXFP4-coordinated) · 5 rework prototype (id-gated claiming; lane-held) · 3 metadata hoist (lands on the tile) · 6 Q6_K prefill rerun under production binding.

## Live lanes

- **addon** (successor to i8mm; same session 01a1bbe6, branch codex/arm-addon
  on c24ce9b4e): Lane 2 = ideas 03 (metadata hoist + publication proof),
  04 (wider panel + activation reuse, MXFP4-coordinated), 11 (attention
  validation hoist + integer-dot proof). Plan-first; independent until fully
  merge-ready, then handover to Main.

- **ideas 25+7** (arm-qualification session 01a1bd53, branch
  codex/hotpath-cleanup-prefill on ea5402a54): idea 07 implementation (RoPE
  allocations, inv-freq lock, RMSNorm heap fallback, gamma TLS stale fix) +
  idea 25 startable parts (PerfStats bucket counters, nsys-in-uma3 image
  proposal, single-bucket/async design). Plan-first; handover to Main when
  fully merge-ready.

## #43 head 425002104 final aarch64 gate (2026-11-09, Main, Mac container)

- Preflight 2242/2317: 75 failures all known classes (NUMA/MPI env + base
  defects classified earlier); only new name CPUShardFusedFinalAddPreweight_
  neon = ARM cell of the base NaN-corpus vs fail_on_nan (its _scalar cell
  fails identically on the pure base). Unit 939/969: five load timeouts
  (Mac-VM-slow; pass standalone 2-8 s) + NativeQuantisedCommonDispatchPolicy =
  base defect (registry 10 vs 9; base added OrderedThreeRows; identical on
  pure base 46c35f15b). NO new ARM failures.

## Arm tip update (2026-11-09, Main)

- #43 rebased onto moved base **07d4f2085** → tip **7d1a94479** (147
  commits, MERGEABLE/CLEAN; replaces d730777bd). Content unchanged apart
  from conflict resolutions (contract-header comment; upstream packer doc
  line; V2CPUExecutionExactPolicyTests.cmake early-return on NEON envelopes
  reusing Main's intent; registration test new ARM case passes). CUDA stack
  byte-identical. OPERATOR DIRECTION: routine base rebases don't trigger
  rebuilds/regates → d730777bd-armed GB10 gates valid; run at 7d1a94479 if
  not started. Active anchors: base 07d4f2085; tip 7d1a94479.

## DGX station CPU test result (2026-11-09) — new hardware datapoint

- Host: NVIDIA GB300 station, **Grace / Neoverse-V2**, 72 cores 1 socket
  1T/core, PAGESIZE=65536 (64K), NUMA 9 nodes (node0 = all CPUs + 506GB;
  nodes 1-8 memory-only). Features: sve sve2 svei8mm i8mm bf16 svebf16 fp16
  sha3 sha512 sm3 sm4 paca pacg bti → I8MM YES, SVE2 YES, SMMLA NO (no SME).
- Run: 39c24d8b4 (content-equivalent to tip 7d1a94479 for CPU-only; CUDA
  stack excluded; operator no-regate direction applies), DEVcontainer gcc-14
  + cmake + ninja + ccache, Integration CPU-only SHARED_CORE=ON, .venv (no
  model fetch), build EXIT=0 (3177 targets, ~6 min).
- UNIT 952/981 pass/29 fail (151s): base 9 + env 15 (64K-page assumption +
  CPU-less NUMA nodes: mmap 'page-aligned and nonzero', first-touch
  placement, capacity-0/NULL alloc, estimator byte equality) + fixture-
  missing 5 (ModelLoaderRowSlice [needs qwen2.5-0.5b q4_0 GGUF; NOT on
  station] + ROCm trainer CSV ×4). 40 initial false failures = missing
  python deps → cleared via requirements into .venv.
- PREFLIGHT 2243/2344 (685s): 45 = 64K-page mmap (×581), 16 = MPI hwloc
  bind (NUMA), 14 = first-touch, 26 = owner base + env arena geometry. NO
  arithmetic/NaN regression evidence. No source regressions.
- Envelope built = NEON default (LLAMINAR_COMPILED_WITH_NEON_I8MM=0);
  NEON_I8MM explicit supported + hardware capable → FOLLOW-UP OFFERED:
  -DLLAMINAR_CPU_ISA=NEON_I8MM gate (real SMMLA datapoint on new CPU).
- VOCAB-only GGUFs only → smoke skipped (no fetch allowed).

- Arm tip moved again (Main): **a27c7ed4f** (replaces 7d1a94479; routine base
  rebase; no-regate rule). Bundle 1 (measurement lane 01a1bd53) executing:
  15 (doc) → 20 → 09 → 01; answers: raw WSL logs pulled
  (/tmp/armqual-b1/raw/wsl/); DGX extracts incoming; deliverables = docs→
  tracker, 01 code = codex/bundle1-perf-harness @ a27c7ed4f → reviewer →
  FF to Main; 01 scope = 01-1 + 01-3; isolation = own container/private
  ccache (the (C) lesson).

## No-regate rule nuance (2026-11-09, Bundle-1 lane → Main)

- Routine base REBAsES (no content change) → no regate, correct. It does NOT
  cover a base commit that ADDS new x86-only test code to shared files:
  cc502b95a (base 07d4f2085) added the unguarded __m256 AddWitness to
  CPUNativeQuantisedUnsignedWeightDotLifecycle.inc; #43's aarch64 rebase
  adapted the .inc but missed that block → ARM preflight gate (and R2
  r2gate2) cannot build at a27c7ed4f. LESSON: a cheap COMPILE-ONLY aarch64
  check of the gate targets (or the R1/develop gates) on the merged tip
  would catch new-x86-code-in-shared-files instantly; recommended as a
  routine step on any base pickup (candidates: compile-only ARM target
  build in preflight or CI image gate). Mac record (lane) continues with
  ninja -k 0 (affected = BUILD-BREAK, not env); ETA ~1h.

| 30 | ARMv9 ISA bundle (SVE2 + FP16/fp16fml) | `30-armv9-sve2-fp16fml-bundle.md` | Open (proposal; operator decision) |
| 31 | S=4 for remaining expanded encodings (Q5_K+nibble spillfix) | `31-panel-s4-expanded-spillfix-followup.md` | Open (via R2 C4 census; 5–40% Q5_K, 2–11% nibble) |
