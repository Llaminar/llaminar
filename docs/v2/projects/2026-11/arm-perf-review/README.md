# ARM performance / memory review — idea index

Created: 2026-11-07 (review carried out by read-only session `01a1b0ec`).
Reviewed against: `codex/qfn-ngram-continuation-arm64` @ `be6658c38`
(rebase onto `feat/qfn-ngram-continuation` @ `2f01249c9` complete; 28 ARM
commits replayed).

**Current anchors (2026-11-10):** arm branch PR #43 head = **`8fcc0fe0b`** (169 commits; R2 MERGED via FF — 29351a57e -> 8fcc0fe0b; reviewed tip exact)
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

## ROLE DIRECTIVE (2026-11-09, operator) — DGX station = perf implementation lane

- The DGX-station session (subagent-chat-01a1c290-ac3d-771f@dgx-station) is the
  PERFORMANCE-IMPLEMENTATION lane: anything that gives speed or lower memory.
  RULE: only implementations that benefit SPARKS (GB10/X925). Verify first
  that the mechanism/feature EXISTS on Spark (e.g., probe .80), then
  implement on the station. Grace-only = ignored.
- Sequence per item: verify-on-Spark -> plan-first (to arm-perf) -> implement
  on the station (own branch off the ARM tip; push to origin) -> self-gate +
  reviewer checkpoint -> .80/WSL gates (arm-sparks) -> handover to Main (FF).
- Queue (Spark-first): (a) X925 corrected VL probe verdict -> SVE2-256 impl
  OR deprioritize; (b) idea 31 S=4 spillfixes (GNU/Spark side: NibbleLUT
  spills + Q5_K minimum epilogue); (c) FP16 fmlalq emission check; (d) future
  perf ideas route here (13/16/22 etc. as the operator directs).
| 32 | NEON serial/grouped NaN-precedence witness | `32-neon-nan-precedence-witness.md` | Open (Main follow-up; not a blocker) |

## Tip f69446e3c (2026-11-09, Main) — base fixes in

- (a) qwen4exp_device_primitives CUDA branch guarded on TARGET (CUDA Release
  + tests ON configure OK @sm_121); (b) UnsignedWeightDotLifecycle AVX2/AVX512
  witness __AVX2__-only + the x86 planner assertions (plan.projections /
  nativeQuantisedUsesKPartitions) x86-only. Full aarch64 Integration tree
  builds on Mac (exit 0); fused-completion + unsigned-dot NEON cells pass.
  LESSON (recorded): routine base rebase with NEW test code in shared files
  needs a FULL ARM COMPILE; configure+policy tests insufficient (matches the
  no-regate nuance entry). research/gdn_leaf_elimination __m256 = not
  CMake-built; left (Main). Arm tip = f69446e3c (151 commits @ 07d4f2085).

- OPERATOR (2026-11-09): the DGX-station session keeps the implementation role for
  performance work (it is a fast implementer). Routing rule: perf implementation
  (idea-31 S=4 slices, FP16/FMLAL, follow-ups) -> DGX perf lane; Spark-first
  verification; gates/reviewer/evidence unchanged. Keep the pipeline fed so the
  fast implementer can move immediately when a checkpoint releases.

## Host inventory (operator reminder 2026-11-10)

- TWO DGX Spark GB10 hosts (arm-sparks owns both): **.79** = GPU/CUDA side
  (GB10 CUDA runtime gates, sm_121 spill compile, Qwen3.8-UD trace) and
  **.80** = CPU/tree side (R2 sequence, censuses, disasm, Release/economy,
  B-fix no-4KiB-regression runs). RUN IN PARALLEL where independent — do not
  serialize unrelated work across the two.
- WSL (x86): x86 CPU-only gates/i8mm control/N3. Mac (M1): AppleClang/portability
  + Bundle-1 measurement. DGX station (Grace/GB300, 64K): perf-implementation
  lane + the only 64K host (B-fix validation window).

## R2 MERGED (2026-11-10, Main) — #43 FF 29351a57e -> 8fcc0fe0b (169 commits)
- Review tip exact (FF, no rebase); PR body updated. R2 = C2c admission +
  C3 Ops seam + C4 per-encoding widths + tournament harness (the full fix-back
  chain #151/#153/#155/#157 + owner gates: .80 43/43 first-FA-exec, engine
  byte-identical + decode +7.9%/+2.3%, WSL x86 14/14 + admission 3/3+6/6).
- Lanes re-anchor to 8fcc0fe0b: idea-31 (972c0081e) + policy selector
  (6ebe1aefb) rebase onto 8fcc0fe0b after the .80 GNU census green; B9 = addon
  tooling commit; GB10 CUDA stream investigation continues on the new tip.
- YTD merged chain: base -> R1 -> ideas-7+25 -> CUDA stack -> B-series (B1-B7
  + B10) -> R2. Remaining handovers: idea-31 (post-census rebase) + Bundle-1
  (FF-time rebase target 8fcc0fe0b).

## .80 disk-full incident + build-tree rule (2026-11-10, addon lane)

- `/` hit 100% (6.3 MB free); the idea-31 census attempt at 3027d7bb8 died
  with `ld: final link failed: No space left on device` (i31-run.log RC=1,
  i31-build.log). **No idea-31 GNU census receipt exists yet.**
- Root cause: `build/int-3027d7bb8` = **473 GB** — configured with the
  CPU-only default `LLAMINAR_SHARED_CORE=OFF` → full core statically linked
  into all ~380 test binaries (~1.2 GB each). Gate trees use
  `-DLLAMINAR_SHARED_CORE=ON` → ~30 GB each. (Same lesson as the WSL
  400G+ trees.)
- RULE (runbook): **every new Integration int dir on .80/.79/WSL/station is
  configured with `-DLLAMINAR_SHARED_CORE=ON`** unless a static-core build
  is explicitly required and its size budgeted.
- Cleanup: addon lane deleted its own superseded gate dirs (int-f9f190059,
  int-777f7a52a, int-8c90cd879; logs kept) → 91 GB free (98%). arm-sparks
  to delete its own int-3027d7bb8 from inside the container (root-owned).
- Receipt recipe (no full Integration build needed): `census-matrix.sh`
  builds only the witness probes for the 9 GNU-14 cells (~25 min), then
  `r2cmp.py` per cell; if Unit/preflight TileRegister tests are wanted,
  reconfigure the int tree with SHARED_CORE=ON (r2gate2.sh pattern).
- Side observation (non-blocking, census tooling): `g-*.census.log` shows
  `Debug * CONFIGURE_FAILED` lines before the 9 results — a first-attempt
  Debug configure that was retried; addon lane to investigate.

## Evidence correction — census-gnu-obj reuse defect (2026-11-10, addon lane)

- DEFECT: `source119-80.sh` wrote the GNU 9-cell census to the FIXED dir
  `build/census-gnu-obj`, first configured for trees/46c9ffc58. Any other
  TREE → CMake cache mismatch → ALL 9 configures failed; the log grep only
  surfaced Debug lines; `census-matrix.sh` didn't clean old outputs or exit
  nonzero; the summary step then read STALE JSONs (11-09 21:36-21:38, from
  the f3b28a120 run).
- CORRECTED CLAIMS: the "GNU 9-cell Rows1=0 / budgets byte-equal" parts of
  g-f9f190059, g-777f7a52a and **g-8fcc0fe0b are STALE (f3b28a120 values)**;
  the R2-handover row claiming that matrix at 8fcc0fe0b is wrong as stated.
- STILL VALID: TileRegister ctest 12/12 at int-8fcc0fe0b (incl. the
  canonical build-time witness V2_Integration_CPUNativeNEONTileRegisters,
  GNU-14 Integration NEON_I8MM, at 8fcc0fe0b); the FRESH per-encoding
  census `census-r2gnu` at 063fc292a (green; `git diff 063fc292a 8fcc0fe0b`
  empty over gemm/primitives/witness/budget/portability/cmake); ALL Mac
  AppleClang censuses (fresh dirs).
- TOOLING FIX (deployed to .80 build/, Mac-verified): census-matrix.sh now
  removes per-cell build dirs + old JSON/err before configuring, stamps the
  tree SHA into every JSON, and EXITS NONZERO if any cell fails (forced
  failure → rc 1, zero JSONs left). source119-80.sh writes per-tree dirs.
- CORRECTIVE RECEIPTS queued on .80 (stamped, fresh dirs, ~25 min each):
  **3027d7bb8 → 8fcc0fe0b → c4536cf3c**; success = rc 0, all 9 lines
  `exit=0 tree=<sha>`, Rows1=0, budgets 1029/1176/1274.

## Base moved again (2026-11-10): #43 replayed onto 2948b1e49 — FREEZE until publish

- Operator-requested rebase. Main replayed #43 (171 commits) onto base
  2948b1e49; local candidate **a710eba02, NOT pushed**. Full ARM Integration
  compile underway; NO full regate / 6-arch rebuild (routine-rebase rule).
- Resolutions preserve: new-base caller-NUMA ownership (all allowed CPUs,
  /proc/thread-self, unknown=-1), portable OpenMP placement + wait policy,
  both preflight registration blocks.
- FREEZE: all lanes hold at c007a59a2 until Main publishes the rebased SHA.
  Pinned gates finish unchanged (CUDA rerun at c007; censuses at
  3027d7bb8/8fcc0fe0b/c4536cf3c = content receipts, unaffected). Idea-14 A +
  reviewer checkpoint stay anchored at c007 (routine-rebase rule).
- FF-time consequence: the idea-31 stack (95d462930/3027d7bb8/e5248cba8/
  042a9e9da) and Bundle-1's 897727484 sit on 8fcc0fe0b → both redo their
  mechanical rebases onto the published tip before push/FF (Bundle-1
  pre-committed to redo-on-new-tip; idea-31 replay + reviewer re-pin per
  protocol). Bundle-1's PUSH is on hold until then.

### Rebase overlap detail (Main, publish pending)

- Resolutions touched: tests/v2/CMakeLists.txt (both late base registrations
  + backend-omission accounting preserved), Test__MPIBootstrap.cpp (both
  independent tests), MPIBootstrap.cpp (portable placement + bounded wait),
  NodeTopology/NUMATopology headers/bodies (all-CPU exact caller ownership;
  unknown stays "unknown"). NO gemm/tournament production conflicts →
  idea-31 stack replay = clean.
- New-base fix 28ddbe253: gates the x86-only OrderedNibbleSplitDot-
  BoundaryLifecycle body + AVX2/AVX512 preflight registrations off ARM
  (borrowed banks/scheduler) — sits near Bundle-1's CMake insertion point;
  its replay resolves semantically (three coexisting elements: new base
  ARM-gated registrations + DecodeBandwidth 39 lines + tournament anchor;
  added-line identity = the proof).
- B9: census receipt at c4536cf3c is content-bound → travels with an
  identical replay (range-diff 1/1 `=` + reviewer re-pin).

### PUBLISHED (2026-11-10, ~11:10Z): #43 tip = 28ddbe253 (172 commits onto 2948b1e49)

- Operator direction: always push, don't wait for compile → published with
  the full ARM compile STILL RUNNING (not certified); 14/14 target-selection
  checks pass. Freeze LIFTED for mechanical rebase/reviewer re-pin; pinned
  gates finish unchanged (B9 census proceeds at c4536cf3c as queued).
- Both stacks replay onto 28ddbe253 BEFORE FF handover: idea-31 stack
  (95d462930/3027d7bb8/e5248cba8/042a9e9da — clean replay expected, no
  gemm/tournament conflicts) and Bundle-1's 11 commits (CMake resolves
  semantically around 28ddbe253's own ARM-gated x86-only registrations;
  added-line identity = the proof). Bundle-1's push additionally waits for
  the .80 bound-run result.
- I8MM runtime byte cells (focused set: PreparedProjection Unit/Integration,
  PanelTailsMatchSerial, I8MMOperandGrouping, I8MMTailsMatchSDOT,
  PanelTournamentContract) run ONLY at the eventual rebased handover tip.

### Stacks rebased onto 28ddbe253 (2026-11-10, addon lane; all-mechanical, range-diff `=` every commit)

- **Idea-31 + selector** = `codex/arm-panel-tournament-policy` @ **df15aca96**:
  4950e6101 (=95d462930) → **870bb545e** (=3027d7bb8, install commit) →
  8e23691d8 (=e5248cba8) → **df15aca96** (=042a9e9da).
- **B9** = `codex/arm-witness-post-index` @ **f265d2950** (= c4536cf3c).
- Ancestry: 28ddbe253 does NOT contain 8fcc0fe0b — Main re-landed R2 as
  **cd0f95150** (all 11 commits present); neither stack carries old R2
  commits. Mac quick checks: PreparedProjection DotProd 15/15 @ df15aca96;
  tournament self-test 0 failed; witness shards OK @ f265d2950.
- .80 ready (refs i31final/b9final/base28d; trees seeded; **561 G free** —
  the 473 G deletion evidently done): focused I8MM byte-cell set at
  df15aca96 (GO) → B9 census at c4536cf3c (queued). Install receipt at
  870bb545e = content-bound to the GREEN 3027d7bb8 receipt; re-run only if
  the reviewer requires. Reviewer re-pins requested for both stacks.

### arm-sparks status report (2026-11-10, ~11:21Z)

- int-3027d7bb8 DELETED (in-container): 473 G freed; .80 = 84% used /
  **564 G free**. Nothing else touched.
- **Bundle-1 bound run (5d606a108) = CASUALTY, never started** — the disk
  filled during the int-3027d7bb8 static-core build BEFORE the bound run
  began (no results dir). Requeued FRESH at SHARED_CORE=ON per runbook.
- Census queue: 3027d7bb8 GREEN, 8fcc0fe0b GREEN (9/9 exit=0, Rows1=0,
  budgets 1029/1176/1274, 176 programs each; tree-stamp "unknown" — see
  note), **c4536cf3c GREEN with PROPER STAMP** (`tree=c4536cf3c` in all 9
  summary lines; B9 receipt complete).
- Tooling note (to addon): the first two runs' summary tree field read empty
  for tree dirnames that matched an existing worktree path but were not
  git-dir-parseable; the c4536 run (after `worktree add` of that exact
  dirname) stamps correctly; JSON `tree` field exists in all. Re-stamping
  3027/8fcc optional (trees still present) — provenance was independently
  verified via compile_commands.
- MTP trace-inventory sweep queued (read-only).

### Reviewer #164: idea-31+selector re-pin ACK at df15aca96 — census does NOT auto-transfer

- Bounded mechanical/source re-pin ACK: chain 28ddbe253 → 4950e6101 →
  870bb545e (install) → 8e23691d8 → df15aca96; range-diff 8fcc..042a vs
  28dd..df15 = 4/4 `=`; same 5-file +423/−35 delta; endpoint files
  byte-identical; PS-1 admission preserved.
- Ancestry independently confirmed: 8fcc NOT an ancestor of 28dd (none of
  its 11 R2 objects in ancestry); Main's re-land ends at **cd0f95150** (IS
  an ancestor; 29351..8fcc vs 96e19ccc3..cd0 = 11/11 `=`; Q16 endpoint
  matches) — preservation evidence, not fresh R2 qualification.
- **CENSUS QUALIFICATION (decisive):** `=` alone does NOT prove the 3027
  receipt qualifies 870bb545e — the canonical census binds dependency
  bytes/resolved paths, configured command+directory, compiler/assembler
  identities, object, checker/envelope/budget, disassembler. Main must
  authenticate complete binding against the new configured tree OR obtain
  FRESH new-pin evidence. DECISION: run the fresh census at 870bb545e
  (tree seeding by the addon; arm-sparks executes) — supersedes the
  earlier content-bound-receipt default.
- Changed base NOT globally identical (175 files; the GEMM family's
  differing Gemv patch = AVX512 pair-dot work, outside this ACK); witness/
  probe/budgets/checker/envelope/portability + rounding registrations +
  tournament contract entries unchanged at base.
- Source ACK ≠ FF/INSTALLABLE/integration/merge grant. B9 = next, separate
  review (#165); #159 did not cover its grammar.

### Tip moved again (2026-11-10, ~11:25Z): e5ee7f672 (173 commits) — gates RESEQUENCED

- Main fixed the .79 span-ctor blocker and pushed immediately: CUDA + ROCm
  rectangular fixtures use typed native_interleaved.unsealedGenerationBytes()
  on the resized destination; fixture-only (two .inc lines), no production
  change; publication-writer policy + diff checks pass.
- Consequence: the stacks (df15aca96 / f265d2950, on 28ddbe253) are NOT
  FF-able from the new head → they re-rebase onto the SETTLED tip (trivial;
  the .inc fixtures don't overlap the stacks' files).
- RESEQUENCED gates (avoid binding soon-dead SHAs per the reviewer's binding
  rule): 870bb545e censuses (GNU + AppleClang) HELD; focused set at
  df15aca96 finishes as content evidence; BOUND RUN next on .80; the .79
  4-cell rerun at e5ee7f672 = the tip-settling event; Main declares
  settlement → addon re-rebases → censuses + focused set run ONCE at the
  final install/handover tip → FF.

### Settlement protocol (Main, 2026-11-10)

Tip NOT settled for handovers. Hold stack re-rebases/censuses until Main's
explicit SETTLED declaration, after the .79 rerun + its GPU iteration/fix-
backs. Rerun expectations at e5ee7f672: the 20 stream-fixed cases GREEN +
the 2 known arena/transfer cases STILL FAILING (that is the slot's purpose;
hand the GPU over on that outcome, not a wholly green padding cell). Further
pushes expected (2 arena/transfer fixes, possibly tile/occupancy). Main will
publish ONE settled exact SHA + overlap list for the final
rebase/re-pin/bound gates. e5ee7f672 = current CUDA build/rerun target.

- (addendum) The addon's Mac AppleClang 9-cell at 870bb545e completed one
  minute before the hold arrived: GREEN (9× exit=0 tree=870bb545e, Rows1=0,
  budgets 1415/2003/2171 byte-equal, no foreign sources). Recorded as
  **superseded evidence, NOT a receipt** (binds a soon-dead SHA); JSONs
  archived at arm-addon-notes/evidence/census-870bb545e-appleclang/. Value =
  strong prior for the final census at the settled install SHA.

### Bundle-1 re-rebased onto 28ddbe253 (2026-11-10, lane; LOCAL ONLY)

- Candidate **a4ebff3dd** (11 commits on 28ddbe253; --onto from 8fcc0fe0b;
  NO conflicts). Proof: range-diff 11/11 `=` vs 897727484 (10/11 `=` vs
  5d606a108 — same CMake anchor `!`); delta-equality (39 identical added
  lines, 0 base removals; the expected three-way CMake interaction with
  28dd's x86 gating did NOT materialize — far offset in the file);
  reconfigure+build OK; ctest -N both sets (x86-only OrderedNibbleSplitDot
  registrations correctly ABSENT on aarch64 = 28dd gating confirmed);
  5/5 rerun. Files: /tmp/armqual-b1/01-ff-28ddbe253-*.
- Bound run stays at 5d606a108 (queued; content identity proven — no
  push-before-acceptance). After Main's SETTLED + publish: one more
  mechanical re-rebase onto the settled tip, then push a4ebff3dd' on the
  green bound result → Main FF.

### Receipt-composition constraint (reviewer #165 finding, relayed by Main)

idea-31/selector and B9 source deltas are disjoint, but their RECEIPTS
cannot compose: B9 changes `checker_sha256` (pre-B9 3027-class proof fails
under the new checker) while B9's c4536 census ran on PRE-S4 source.
POST-SETTLED therefore: the addon prepares ONE COMBINED candidate (both
stacks on the settled tip, either order) → reviewer re-pin → final receipt
set runs ONCE against that actual checker + S4 source + configured targets:
GNU census (.80) + AppleClang 9-cell (Mac) + focused byte-cell set (.80).
No two independent censuses inferred into a combined pass. Reclassified as
superseded evidence: B9's c4536cf3c census (pre-S4 source) and the
870bb545e AppleClang run. Bundle-1 = separate FF (own harness evidence;
re-rebases onto the post-FF tip). No new runs before SETTLED.

### Reviewer #165: B9 SOURCE ACK at f265d2950 (first grammar review)

- Mechanical: parent 28ddbe253; c4536cf3c→f265 = 1/1 `=`; endpoints
  byte-identical (+25/−1, exactly two tooling files); probe/envelope/
  budgets/registrations match the donor.
- Semantics SUPPORTED in CPUNativeTileRegisterEvidence.py: the two forms
  (FORMS:187; widths 8/2 at :817-824); post-index uses the OLD base for the
  transfer, reloads tracked whole-X pointer slots, advances the base after
  (:1218, :1243-1244, :1261-1267); narrow W loads cannot become whole-
  pointer reloads; partial reads of tracked pointer slots rejected
  (:1286-1292); transferred-GPR/writeback-base overlap rejected BEFORE
  transfer/state mutation with canonical X/W aliases (:1206-1210; covers
  ldr pre/post + either ldp destination); structural failures fatal before
  budget evaluation; no budget relaxation; the two named unsupported ldr
  forms stay rejected (ldp x,x,mem,imm already admitted — do not
  generalize). New test method :411-428 (5+5 cases) registered in Unit AND
  ProductionTestPreflight.
- NONBLOCKING strengthening recommendation: the new counters do not
  distinguish correct post-increment from no increment; no actual stored-
  stack-pointer reload through the new form; ldurh examples lack an
  adjacent live pointer slot (2 vs 4 bytes indistinguishable). Recommend
  offset-sensitive live-slot controls + an actual pointer-slot reload +
  adjacent/overlapping narrow-read positives/negatives. DECISION: fold the
  strengthening into the combined candidate as a SEPARATE 6th commit (B9
  itself stays byte-identical to the ACKed f265d2950; the reviewer reviews
  the 6th commit fresh in the combined re-pin — and since the test method
  lives in the checker file, folding it now avoids ANOTHER checker_sha256
  composition break later).
- Q6_K remains S2 for Rows1 SDOT (incl. I8MM lone-row leaf); ASIMD/
  shared-row/row-pair = S1; no installation changes; forced-S4 10/25 =
  owner diagnostics, NOT installability; program_key still rejects forced
  Strips!=0. Composition constraint reiterated: the final combined
  source+checker/config needs its own canonical proof. No grants; Main owns
  handover acceptance.

### Correction from the 64K validation owner (2026-11-10)

The b-fix-9490b66bc "4 FAILs = one env class (memory-only NUMA nodes)"
attribution was imprecise: **3 of the 4 fails (NUMATopologyCPUList,
Membership, DeviceRegistry) were B10 (use-after-free) showing through** —
fixed at ce9ddb326 and green in the 9/9 focused run; only
**ClusterInventoryBootstrap** is the true env-class residual. Main's
counter-evidence push was the right call. Full station runbook relayed by
the owner to the continuous-batching lane: devcontainer CPU-only image;
.venv from requirements.txt (no model fetch); Integration CPU-only flags;
bounded parallel build; SYS_NICE + seccomp=unconfined REQUIRED for
page-owner tests (evidence: EPERM log line + 16-flip unmasking proof +
unmask_cmd.txt); onednn = CMake AUTO-CLONE v3.11.3 with
build-neon*.llaminar.lock lock files (NOT a submodule), cnpy nonexistent,
corpora submodule must NOT be inited; own worktree or own external bind —
NEVER a shared writable external/. Campaigns COMPLETE; build_v2_dgx/ =
historical evidence, no active reservation.

- (addendum) FOCUSED I8MM BYTE-CELL SET GREEN at df15aca96 (Integration
  NEON_I8MM SHARED_CORE=ON; all 7 matched tests PASSED, 1.71s):
  PreparedProjection Unit 3886 + Integration 3887, PanelTailsMatchSerial
  3890, I8MMOperandGrouping 3892, I8MMTailsMatchSDOT 3893 (+2). = content
  evidence; the final focused set runs at the COMBINED tip post-SETTLED
  (composition constraint). arm-sparks queue now: bound run (.80) + .79
  CUDA rerun build at e5ee7f672 (parallel hosts).

- (addendum) Rebase VALIDATION COMPLETE at e5ee7f672: full local ARM NEON
  CPU-only Integration build EXIT 0 (same unrestricted/-k0 settings; prior
  parity-compiler OOM attempts retained); focused 14/14 target-selection,
  4/4 MPI/OpenMP/NUMA CTest cells + 7/7 NUMA/socket fixtures PASS; live
  registry case unselected on the NUMA-less VM. PR body updated;
  GitHub MERGEABLE/CLEAN. Still NOT CUDA qualification / NOT SETTLED for
  handovers (.79 rerun + GPU slot outstanding). The earlier
  "published-while-compiling" caveat is closed for the CPU-only side.

### 01 BOUND RUN COMPLETE — graded BOUND (2026-11-10, .80 @ 5d606a108)

- Results: build/results/{bound-5d606a108.json, bound-bw-result.json}.
  Kind=kernel-metrics; build identity: Release, NEON ISA envelope (default,
  pre-ISA-pin configure — acceptable: the bound run validates harness
  content), g++-14 14.2.0, -O3, SHARED_CORE=ON, CUDA/ROCm OFF;
  observations=1 (Ladder: Q4_K/Q5_K/Q6_K/IQ3_S × threads 20/10/1 × max
  64MiB; 2 tests PASSED).
- Provenance note: tree = /work/trees/b5d606a108 (the 5d name was claimed
  by a stale prunable entry; identical tree + clean .git).
- Bundle-1 acceptance input SATISFIED → push sequence: Main's SETTLED
  declaration → final re-rebase onto the settled tip → force-with-lease
  push (a4ebff3dd') → Main FF.

### Standing rule adopted (reviewer ruling #166, operator-delegated): C as a HARD installed-set gate

Committed VERBATIM to **register-gate-standing-rule.md**. Two corrections to
the coordinator's draft: budget improvements = explicit committed downward
ratchet (never auto-rewritten acceptance tests); an unqualified installed
state cannot ride along in an otherwise unrelated merge. Key operational
point: the standalone census subcommand is DIAGNOSTIC — the hard gate =
the canonical BOUND build/verify evidence per required cell; tree stamps /
GREEN summaries / patch equality alone cannot satisfy it. Combined
idea-31+B9 qualification still needs its own combined evidence (receipts do
not compose). No authentication/branch-protection/merge grants in the ruling.

### Addon adoption of the standing rule + receipt-path replacement (2026-11-10)

- RECLASSIFICATION: census-matrix.sh output (incl. the 3027/8fcc/c4536/
  870bb545e runs) = DIAGNOSTIC evidence, not qualifying receipts, per the
  standing rule. Qualifying evidence = the canonical bound proof from a
  configured gate build for EVERY required cell: the
  `v2_cpu_native_tile_register_proof` target (proof.json, budgets applied,
  binding verified) + the `V2_*_CPUNativeNEONTileRegisters` and
  `TileRegisterEvidence*` ctests.
- Combined-candidate composition (pending reviewer confirmation at re-pin):
  18 required cells = **9 configured builds per compiler** (3 build types ×
  NEON / NEON_DOTPROD / NEON_I8MM) × GNU-14 (.80) + AppleClang-21 (Mac),
  via the core-free portability project.
- The addon is writing a FAIL-CLOSED driver (fresh per-cell configure at the
  exact tip → proof target → TileRegister ctests → per-cell proof.json →
  nonzero exit on any failure/missing) to replace census-matrix.sh as the
  receipt path; census-matrix.sh stays diagnostic. Ratchet: any grouped-key
  lowering by the combined candidate = explicit downward-ratchet commit
  through review.
- PRECISION (reviewer, #166 follow-up): the five TileRegisterEvidence
  Analysis/Binding0-3 shards test the CHECKER (incl. stand-in tools) — they
  are NOT the bound native register witness. Per covered cell, the native
  owner is `v2_cpu_native_tile_register_proof` with verification aliases
  `V2_Unit_CPUNativeNEONTileRegisters` + `V2_Integration_CPUNativeNEONTileRegisters`;
  the required checker shards COMPLEMENT each cell's native build/verify
  evidence. A proper build directory alone changes nothing about this.

### Final-evidence composition specified (reviewer #167) — committed verbatim

**register-gate-final-composition.md**. Core: 18 required cells (GNU-14 +
AppleClang-21 × Debug/Release/Integration × NEON/NEON_DOTPROD/NEON_I8MM) via
the core-free portability project; ADDITION = the companion
compiler-envelope proof (`v2_cpu_codegen_envelope_proof`) alongside the
native proof target; per cell = config identity + compile DB + 14 CTest
aliases (native/envelope/checker, Unit+Integration); receipt = proof.json +
proof.o/dis/i + envelope proof.s/target.txt + compile DB + exit statuses +
inventories; final acceptance = the aliases re-running the SAME verifier
after final inputs fix (no third verifier); driver = completeness/binding
coordinator, NOT a new authority; native verification is host/toolchain-bound
(no cross-host JSON splicing); Debug = real config flags into the OPTIMIZED
witness (never omitted/loosened/copied); shared budget key → max across
configurations, never lowest/average. Old idea-31/B9 receipts still cannot
substitute. Spec only — no driver approval/execution/merge grant.

### Receipt driver v1 prepared (addon, 2026-11-10) — pre-#167 shape, adaptation directed

- `arm-addon-notes/register-gate.sh` (+ .80 build/): per compiler run —
  (a) native evidence per cell (3 build types × 3 ISA; fresh portability
  configure; proof target; both native aliases; proof.json retained);
  (b) checker shards per checker scope (10 TileRegisterEvidence tests, all
  listed + pass). Fail-closed per class (native=/checker=/gate= summary);
  nothing reused between runs; failures retained.
- Mac trial (AppleClang-21 @ c4536cf3c; TOOLING TRIAL, not a receipt):
  9/9 PASS, native/checker/gate PASS, rc 0; full binding in every
  proof.json; Rows1=0. NEGATIVE CONTROL: lowered one grouped NEON key by 1
  → the 3 NEON cells failed with PROOF_BUILD_FAILED citing the exact
  checker reason; others passed; rc 1. Archived:
  evidence/register-gate-trial-c4536cf3c-appleclang/.
- Adaptation to reviewer #167 directed (envelope proof + aliases; 14
  aliases/cell; full receipt artifact set; final-acceptance-via-aliases;
  coordinator-not-authority; host binding; Debug semantics; shared-key max).
  No gate runs before SETTLED.

### .79 disk-full incident (2026-11-10, fixed) + pruning hygiene rule

- .79 hit 100% during the e5ee7f672 Integration build (closing-depfile
  failure at v2_test_native_source_final_retirement; 48 earlier FAILED
  entries = the same starvation cascade — free-space artifact, NOT a tip
  defect). arm-sparks cleaned 2 superseded build dirs it owns
  (build-gb10i-c007 + build-gb10i-07d4; 70G freed; .79 = 69G/1.8T free) and
  relaunched the -k 0 build (~32G expected; monitored).
- HYGIENE RULE (both GB10 hosts): prune superseded gate build dirs once
  their gates seal — keep the last sealed tip's dir + all logs. Accumulation
  across tips starved .80 (473 G static-core) and .79 (70 G) the same day.

### Receipt driver v2 — #167-conformant (addon, 2026-11-10)

`register-gate.sh <tree> <out> <expected-identity> [cmake args]` (also on
.80 build/); completeness/binding coordinator only, verdicts from the
existing build/verify authorities. Per cell (Debug/Release/Integration ×
3 ISA): persistent per-cell portability configure (regenerated on input
change; failed configure fails the cell; Integration empty-flag guard
enforced by v2_validate_portability_integration_flags) → build BOTH proof
targets → inventory = exactly 14 aliases (native/envelope/checker,
Unit+Integration), run with --no-tests=error, all must Pass → per-cell
receipt/<cell>/ with the full #167 artifact set (register
proof.json/.o/.dis/.i + deps; envelope proof.json/.s/target.txt; compile
DBs; commands/exit codes/logs; inventory) → reconcile (identity/ISA/build-
type/budget+checker hashes vs tree). Receipt dir RECREATED each run (no old-
file harvesting); failure = failure report, never a receipt. Checker shards
conservatively PER CELL pending the reuse-clause answer. Mac trials
(AppleClang-21 @ c4536cf3c; tooling trials): fresh 9/9 + reuse 9/9 (~9 min);
negatives = identity mismatch (reconcile), unregistered alias (inventory
13/14), lowered budget (cell-attributed PROOF_BUILD_FAILED) — all rc 1.
Archived: evidence/register-gate-trial2-c4536cf3c-appleclang/ (125 MB).
Post-SETTLED usage: .80 GNU-14 + Mac AppleClang-21 at the combined tip.
Holding until SETTLED.

### .79 4-cell rerun COMPLETE at e5ee7f672 (2026-11-10, ~14:07Z) — handover condition MET

- CUDAResidualAddParity PASSED (9/9 focused set incl. TailExtents +
  GroupedVerifierRows_CUDA_ResidualAdd); CUDAFlashAttention_DecodeKVLen
  PASSED (incl. SmallMFP16KVRowsMatchSerialReference);
  CUDAAttentionPaddingParity CausalMask+NonCausal PASSED, the 2 known
  arena/transfer cases still failing = Main's expected known set.
- NEW: CUDAOpsParity SEGFAULT (signal 11) in ResidualAdd_FP32_Small,
  reproducible 2/2; the 15 prior cases PASS (RMSNorm×3, SwiGLU×6, RoPE×3),
  then the FIRST ResidualAdd case inside CUDAOpsParity segfaults while
  CUDAResidualAddParity's identical-class cases pass → fixture-level
  (CUDAOpsParity ResidualAdd setup path), NOT the null-stream class (no
  timer exception). Fix-back item for Main's slot.
- 20+ stream-fixed cases GREEN + the 2 known fails = handover condition
  met; .79 GPU FREE; Main's ~2h slot agenda: (0) CUDAOpsParity segfault,
  (a) 2 arena/transfer cases, (b) sm_121 tile inventory, (c) idea-21b
  grouped-IMMA probe. SETTLED after the slot + focused rerun.

### GPU slot slice 1 GREEN (2026-11-10, ~14:32Z) — tip 18b455d62

Main pushed **18b455d62** (first slot slice): compiled ops+padding owners
with the native GB10 toolchain; full CUDAOpsParity + full
CUDAAttentionPaddingParity + both new focused ProductionTestPreflight
invocation aliases PASS (4 native CTest cells + model fixture). ROOT CRASH
CAUSE (reproduced before fix): the local test's five-arg C redeclaration
omitted the actual sixth stream argument → production declaration included
+ explicit same-stream I/O. **LMHead workspace + KV input admission fixes
ALSO PASS** → the 2 known arena/transfer cases appear fixed. Continuing:
sm121 tile inventory + idea21b probe. NOT SETTLED yet.

### GPU slot slice 2 (2026-11-10, ~14:40Z) — tip 371f5ab35 (175 commits)

- TILE INVENTORY (b) FIXED AT SOURCE: the missing dense prefill tile was
  exactly tile5 — NOT sm_121 resource ineligibility; the fixed inventory
  geometries hit short-K/marginal-wave heuristic demotions at GB10's SM
  count. Fix: an all-format long-K geometry scaled to >=1.5 waves, all tile
  assertions + captured byte oracles retained. Native GB10 focused bucket-M
  prefill PASS (16s); both invocation regressions repeatedly passing.
- IDEA-21b (c) REPRODUCED EXACTLY: codebook4 C32 = 62 regs / 8 CTAs;
  codebook13 C32 = 56/9; NO SPILLS. Main investigating 12.x occupancy
  WITHOUT budget loosening (real kernel improvement first — consistent with
  the standing rule's escape-hatch discipline). Finishing durable summary.
NOT SETTLED yet.

### GPU slot: profiler-attachment safety finding (2026-11-10, ~14:48Z)

- During idea-21b baseline profiling on .79: NCU produced its report and the
  test passed, but the owned driver window FAILED on repeated new NVRM
  refcntRequestReference_IMPL warnings (state1/current0/status0x56). No
  Xid/reset seen; Main STOPPED further NCU/NSYS attachment for this slot
  (safety-first). Evidence: /home/factory/idea25/slot-profile-baseline.log +
  slot-driver2-report.json + slot-profiles/grouped-cb4-baseline.ncu-rep;
  first failed attempt (session-root path admission) retained separately.
- ROUTE CHANGE: non-attached CUDA tests + compiler/SASS inspection only;
  clean non-profiled driver window check before more runtime work.
- Compile-only occupancy candidate status: CB4 48 regs / CB13 40 regs, zero
  spills; baseline full6 compile green; no kernel change published yet —
  the budget targets reachable WITHOUT loosening (standing-rule compliant).
  NOT SETTLED.

## SETTLED FOR HANDOVERS (2026-11-10, ~15:07Z): e8866ec044d69e4d5821a7a8ed43fa45c53b2b64

#43 = 176 commits on 2948b1e49. .79 GPU returned to arm-sparks. Final
exact-source selected CUDA gate GREEN: 10 native cells + fixture (JUnit
11/11, zero skips/errors), both new invocation contracts repeated 20/20;
final non-attached driver window clean. Resource contract remains
<=48regs/>=10CTAs, now includes source-scale IQ4 on 12.x; CB4/13/24
registers 62/56/56 → 48/40/45, ZERO SPILLS. Changed TU guarded
121,120;121; full6 green; shipped SASS encodings unchanged (only nvcc
source-digest symbol salts normalized). Matched Integration event delta
~0.6% within noise; NO Release/E2E speedup claim. Profiler safety HOLD
remains (no NCU/NSYS until the NVRM attachment issue resolves).
Evidence: .79 slot-probes/final2-runtime.log, final/results.xml,
final/test-inventory.txt, final2-driver-report.json; source clean.

Overlap list since 28dd: tests/v2/CMakeLists.txt adds 2 invocation aliases
at tail; ops/padding/GemmParity test bodies, CUDA/ROCm rectangular
fixtures, CUDAMoEGroupedPrefillKernels.cu, Perf__MoEVerifierPrefill.cpp,
ARM plan/evidence docs. NO CPU gemm/tournament/checker/budget
modifications.

CASCADE DIRECTIVES: combined idea31+B9 rebase/re-pin now → ONE final
18-cell canonical proof set (per #166/#167); Bundle-1 follows SEPARATELY on
the post-FF tip; bounded source review of the 3 CUDA commits requested from
the reviewer. The settle notice is not a merge or unrelated qualification
grant.

### COMBINED CANDIDATE READY (addon, 2026-11-10, ~15:23Z): a16591eed

`codex/arm-s4-nibble-witness` @ **a16591eed** — 6 linear commits on SETTLED
e8866ec04: 14c1a0483 (=95d462930) → 860cf70d8 (=870bb545e, install) →
0bd132dbc (=8e23691d8, selector) → a87447dc6 (=df15aca96, K-alignment) →
2646ffebe (=f265d2950, B9 grammar) → **a16591eed (NEW 6th)** =
`V2_Integration_CPUNativeNEONPanelTournamentInstalledWidth` — pins the
installed panel width via a ProductionTestPreflight cell (production batch
path at Strips=0 vs the Strips=1 program; byte-identity + dirty/guard/
immutable checks). Range-diff: 5/5 `=` + the 6th addition; rebase clean;
overlap confirmed (CMakeLists CUDA-tail only).

**Mac AppleClang-21 register-gate receipt at a16591eed — QUALIFYING per
#167:** register-gate.sh 9/9 cells PASS, 14 aliases/cell,
native/checker/gate PASS rc 0; tree stamp = a16591eed; every proof binds
this tree's budget/checker hashes; Rows1=0 everywhere; grouped
1415/2003/2171 = the committed ceilings; **ratchet: no grouped key lowered**
(budget JSON diff vs e8866ec04 empty). Receipt archived on the Mac.
Reuse ruling applied: per-cell checker shards kept; 5/5 witness shards OK
at the combined tip. .80: i31final = a16591eed, trees seeded, 524G free.

Pending: reviewer #169 re-pin (incl. fresh a16591eed review; NOTE — the 6th
commit is the InstalledWidth cell, not the #165 B9 checker-strengthening
controls; routing that question to the reviewer) → arm-sparks GNU driver +
focused set (+InstalledWidth) → Main FF.

### Reviewer ruling on the 6th commit (2026-11-10): B9 strengthening DEFERRED

- #165's offset-sensitive/pointer-slot/narrow-footprint strengthening was
  explicitly NONBLOCKING and remains so → acceptable to DEFER for this
  delivery. **Deferred follow-up on the books** (B9 test strengthening:
  offset-sensitive live-slot controls, actual pointer-slot reload,
  adjacent/overlapping narrow-read positives/negatives — to ride with a
  later B9-adjacent change or a dedicated commit).
- The InstalledWidth cell covers a DIFFERENT idea-31 contract — it is NOT a
  substitute for the B9 controls.
- #169 queues the exact a16591eed-over-e8866ec04 re-pin behind #168 (Main's
  3-CUDA-commit slice, in progress).
- WORDING DISCIPLINE (reviewer correction): the Mac 9/9 ×14 and the GNU
  launch are OWNER reports, not reviewer-authenticated "qualifying"
  evidence; #166/#167 govern complete current binding and the final set; a
  source re-pin is not receipt authentication or FF permission — Main owns
  acceptance/grants.

### Reviewer #168 SOURCE ACK — Main's CUDA slice 28dd..e8866ec04 (no blocking finding)

Verified the four linear commits (e5ee7f672 → 18b455d62 → 371f5ab35 →
e8866ec04), static code/registration only:
- Fixtures obtain the owned freshly-resized unsealed-generation span (no
  seal bypass, no production-byte change); residual tests use the
  production six-arg declarations + the fixture's CUDA ordinal + an
  explicit nonblocking stream, ordered upload→launch→download→completion;
  LMHead binds the declared persistent workspace; KV prep via TransferEngine
  on the actual append stream.
- The tile5 long-K geometry (K=2304, N=128*max(14,ceil(3*SM/8))+1,
  M511/513 buckets) = the intended full-wave AUTO tile5 opportunity; no
  forced tile, no eligibility weakening, no removed obligations/comparisons;
  capture poisoning + double launch retained.
- PRODUCTION CHANGE = ONLY the __launch_bounds__ minimum on
  __CUDA_ARCH__>=1200, Columns32, codebooks 4/13/24; resource tests retain
  <=48regs/>=10CTAs for 4/13 and extend to 24 on major>=12 (query addresses
  the actual specialization).
- Both new invocation aliases explicitly in ProductionTestPreflight.
- Limits: source correspondence only; owner runtime reports stay owner
  evidence; no FF/merge grant. #169 (combined re-pin) now unblocked.

### Reviewer #169 HOLD on the 6th commit (a16591eed) — two P2s; 5-patch re-pin ACK stands

- **I31-C1 [P2]:** the new cell sets LLAMINAR_PANEL_STRIPS=0
  (tests/v2/CMakeLists.txt:19183-19188) but the tournament's admit():177
  parses [1,4] excluding 0 → main():557/597-600 returns 2 ("must be an
  integer in [1,4]: '0'") BEFORE allocation/reference/timing; and
  runStrips():248-256 has only 1/2/4 — widening admission alone would
  abort, not execute the installed path. FIX: correct installed-mode
  admission AND dispatch, positive coverage for the real entry, update the
  advertised grammar. Forced-4 substitution does NOT satisfy the
  installed-selection contract.
- **I31-C2 [P2]:** the invocation inherits formats Q4_K/Q6_K/IQ3_S/IQ4_XS/
  Q8_0 (tournament:194), OMITTING Q4_0 and IQ4_NL — the actual no-minimum
  NibbleLUT S4 branch (PreparedProjection.h:195-203); IQ4_XS is
  NibbleLUTSourceScale and stays S2. FIX: explicitly include both intended
  centered/nonlinear sources; retain the metamorphic production-batch-0-vs-1
  byte/dirty/guard/immutable checks.
- Confirmed: 6 commits over e8866ec04; 4/4 `=` + 1/1 `=`; all endpoint files
  byte-identical to reviewed predecessors; #165 deferral not a cause.
- IMPLICATION: the Mac 9/14 core-free receipt does NOT execute this new
  full-project CMake registration (portability ≠ tests/v2/CMakeLists.txt) —
  the native proofs stand but are NOT execution of the new cell; #166/#167
  binding/final-set rules unchanged. Corrected successor requested.

### #169 fix-back (addon, 2026-11-10): combined tip = ee7bf0596

Stack: 5 ACKed patches unchanged → a16591eed → **ee7bf0596** (test-only
fix-back; origin FF; .80 i31final updated; trees seeded, HEAD verified
clean). Addresses both P2s:
- I31-C1: admit() parses STRIPS in [0,4] (3 still rejected; advertised
  grammar updated); runStrips() gains `case 0: return
  run<Instruction,0>(batch)` — the real production selection
  (execute<Instruction,0> → per-encoding panelStrips); positive self-test
  control "installed width 0 admitted" (strips 0, asimd, Q4_0/IQ4_NL); the
  registered cell IS the positive real-entry run; the I8MM M>=2 rejection
  now reads "strips > 1" (0 always admissible).
- I31-C2: default formats + the cell explicitly include Q4_0 and IQ4_NL
  (the no-minimum centered/nonlinear NibbleLUT branch); IQ4_XS stays at its
  S2 source-scale width; the cell exercises every S4-installed source.
- Checks unchanged: Strips=0-vs-1 byte identity, per-invocation dirty,
  guard words, immutable inputs, retained team with observed width;
  metamorphic wording stated in the cell docs. Core-free scope honesty
  confirmed. Mac: self-test 0 failed; real STRIPS=0 run over all 7 formats
  passes byte checks + CSV strips=0; STRIPS=3 still rejected (updated
  message). No FF until re-review + GNU receipt + focused set at ee7bf0596.

### Reviewer #169 SOURCE ACK — I31-C1/C2 closed at ee7bf0596 (fix-back +32/−6, two files)

- C1 verified: tournament:178 admits canonical 0..4 (3 explicitly
  rejected); runStrips():258 forwards 0 → run<Instruction,0> →
  batch.execute → tile.executeAdmitted → unchanged run() selecting
  panelStrips at Strips==0 — the REAL installed table, not forced4 or a
  substituted path. Positive self-test (admitted0/asimd/Q4_0+IQ4_NL) fails
  on exception; the full-project alias supplies the real-entry-0 run. The
  I8MM M>=2 strips>1 predicate was already present — 0 admitted without
  weakening it.
- C2 verified: CMake:19181 pins
  Q4_0,IQ4_NL,Q4_K,Q6_K,IQ3_S,IQ4_XS,Q8_0; defaults include both missing
  nibble sources → the M1 DotProd path reaches both idea-31 S4 no-minimum
  programs; IQ4_XS stays the distinct S2 source-scale program; ASIMD
  installed choice remains S1 (CSV strips=0 = table selection, not a
  measured width witness).
- Unchanged: finite Strips1 reference, retained-team/observed-width,
  per-invocation dirt, byte comparisons, guards, immutable inputs;
  metamorphic production-path coverage. Prior five re-pins preserved; no
  production/checker/budget/probe/envelope bytes change.
- Nonblocking prose follow-up: tournament file-header default-format list
  (line 41) still lists the old five — update at the next touch.
- Reviewer note: its local branch/origin-tracking refs still named
  a16591eed (it reviewed the ee7bf0596 OBJECT directly; no remote-state
  check performed) — addon to confirm the origin ref state.
- Scope: static source ACK; not compilation/execution/receipt
  authentication/FF permission. #169 done → the FF gate = GNU driver
  receipt + focused set at ee7bf0596 (in flight) → Main.
