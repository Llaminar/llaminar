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
| 3 | Move projection metadata validation out of the K loop | `03-projection-metadata-validation-hoist.md` | Addressed (R1 `5d4e3716a` + P2 hardening; evidence complete) |
| 4 | Wider decode column panel + activation reuse | `04-decode-column-panel-activation-reuse.md` | Addressed (R2 C4 `8fcc0fe0b` + tournament); residual open: M=1 activation-reload headroom (Lane-2 candidate) |
| 5 | Heterogeneous (X925/A725) core scheduling | `05-heterogeneous-core-scheduling.md` | Partial (dynamic tile claiming; task-count-dependent placement evidence; typed policy pending) |
| 6 | I8MM (SMMLA) row-pair dot-product | `06-i8mm-row-pair-dot.md` | Implemented + measured (kernel 1.01–1.19, engine +1.3–1.7%, M=1 guard holds, no spills; Q6_K ~0 gain; tip cd55f6b27) |
| 7 | Hot-path allocation / cache cleanup (RoPE, RMSNorm, gamma TLS) | `07-hot-path-allocation-cleanup.md` | Bundle-1 proven (graded BOUND on 28ddbe253; awaiting combined FF) |
| 8 | ARM prepared-weight layout (drop dead compensation bytes, compact expansions) | `08-arm-prepared-weight-layout.md` | Open (Phase A plan approved + GO; window queued post-FF) |
| 9 | Non-volatile rounding barrier evidence | `09-rounding-barrier-evidence.md` | Open (evidence required first) |
| 10 | FP32 RMSNorm scalar sum order + sanitizer warnings (issue #36) | `10-rmsnorm-fp32-sumsq-order.md` | Addressed (`3086894b5`, `ac3289bd9`); residuals listed |
| 11 | Attention validation hoist + integer-dot range proof | `11-attention-validation-and-integer-dot-range.md` | Addressed (R2 C2c + Q16 bound; evidence complete) |
| 12 | Cache-topology-derived tiling / prefetch on GB10 | `12-cache-topology-tiling.md` | Addressed (overlay table retired `264970e14` after head A/B confirmed regression; methodology in idea 20) |
| 13 | Fewer OpenMP regions per decode step | `13-openmp-region-count-and-wakeup.md` | Partial (spin `27585692d` + verified bundle `408ff90e5`; remaining regions open) |
| 14 | Dense-model MTP verifier economics (27B) | `14-dense-mtp-verifier-economics.md` | Base-dependent (new base: MTP wins 5.45–5.55 vs 5.28; B-wave depth tuning booked: first post-FF window) |
| 15 | NUMA-less and cache-unpublished Linux ARM hosts | `15-numa-less-hosts.md` | Partial (memory attribution `a572c3c9c`) |
| 16 | MoE routed experts, GDN projection and LM head in CPU decode | `16-moe-expert-and-lm-head-decode.md` | Open (bundle part closed/verified; GDN + LM head open; Phase A plan approved, window queued post-FF) |
| 17 | GB10 CPU decode bandwidth gap (post-merge follow-up) | `17-gb10-decode-bandwidth-gap.md` | Open (full cross-engine matrix: llama.cpp CUDA 1063/28.8 vs Llaminar CUDA 319/22.8; see idea 24) |
| 18 | GDN grouped verifier + publication economics | `18-gdn-grouped-verifier-economics.md` | Confirmed (GB10 grouped-vs-serial < 1.0 for all M; M=4 0.447 @20T) |
| 19 | Prefill vs decode gap on GB10 (short-prompt effect) | `19-prefill-vs-decode-gap.md` | Open (prompt-length sweep planned) |
| 20 | Overlay collection methodology (microbench wins that lose in-model) | `20-overlay-collection-methodology.md` | Open (new rule: in-model collection or per-format E2E validation) |
| 21 | CUDA GB10 (sm_120/sm_121) kernel register spills under the zero-spill guard | `21-sm121-cuda-kernel-spills.md` | Addressed in-tip (SETTLED `e8866ec04`: 10 native cells 11/11, CB4/13/24 zero spills; sm_120/sm_121 clean) |
| 22 | FA2 tile-tournament policy tail on GB10 | `22-fa2-tile-tournament-policy-tail.md` | Open (p95 regret 8.926% > 5%; tq4_tq4 17.6%) |
| 23 | qwen4exp HyperConnection auto-planning classification gap | `23-qwen4exp-hc-auto-planning.md` | Open (auto `plan` rejects HC models; authored `--expert-tier` works) |
| 24 | GB10 GPU prefill gap vs llama.cpp (3.3×) | `24-gb10-gpu-prefill-gap.md` | Open (dig: ~80% is bucket-pipeline overhead, not kernels; see idea 25) |
| 25 | Prefill capture/bucket-pipeline overhead reduction | `25-prefill-bucket-pipeline-overhead.md` | Trace-ready (nsys present in uma4; smoke = owner-reported candidate, window UNQUALIFIED — canonical driver diagnostics + operator scope pending; 15-min rerun queued post-FF) |
| 26 | Single-rank OpenMP binding on GB10 is A725-first | `26-single-rank-openmp-binding.md` | Addressed (fastest-first binding in the released base 880429059) |
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

### Combined tip advanced (addon correction, 2026-11-10): 717b2bfe6

- CORRECTION accepted: the addon's earlier "origin FF" claim was wrong (the
  fix-back push had not happened; the reviewer's stale-ref observation was
  right). origin/codex/arm-s4-nibble-witness NOW carries **717b2bfe6**;
  before this push it carried a16591eed.
- Prose folded NOW (nonblocking, comment-only): the tournament file header's
  default-format line lists all seven sources, matching line 196.
- NEW combined tip: **717b2bfe6** (7 commits: 5 ACKed + a16591eed +
  ee7bf0596 + 717b2bfe6 prose-only). .80: i31final updated;
  trees/717b2bfe6 seeded.
- Receipt validity: a register-gate receipt at ee7bf0596 remains fully
  valid (the tournament file is NOT a register-gate input; probe/budget/
  checker identical); if arm-sparks hadn't started, retarget to
  trees/717b2bfe6. Awaiting arm-sparks' answer on which SHA its run is
  against. Reviewer light-confirmation of the comment-only delta requested.
- Reviewer confirmation: ee7bf0596 → 717b2bfe6 = direct single-line
  comment-only correction (seven-format default list); the #169 SOURCE ACK
  EXTENDS to this exact pin; the prose follow-up is CLOSED. Source-only —
  not receipt authentication or FF permission. Review side COMPLETE for the
  combined candidate; remaining FF gate = GNU driver receipt + focused set
  at 717b2bfe6 (arm-sparks, in flight) → Main.

### Half-perf cell checkpoint (2026-11-10, ~16:0xZ): adb1c98e9 — test-only, review + Linux validation pending

New lane arm-halfperf (cpu-tuning agent, gpt-6.1-sol:low; operator-directed
parallel implementation) completed its first checkpoint:
**codex/arm-half-perf-cell @ adb1c98e9** ("test(cpu): add independently
owned half operand projection cell"; worktree /Users/floriankinder/repo/
llaminar-halfperf). Branch off settled e8866ec04. TEST-ONLY per plan v2.1
(#150): Perf__CPUHalfOperand* registration; oracle test-owned;
C0-per-invocation-v1. Mac commit = hook-disabled per program pattern
(AppleClang/OpenMP structural limitation), Linux validation OUTSTANDING and
merge-gating: full Integration build + Unit/preflight on .80 + the cell's
economy receipts (authentic GDN geometry/recorded fixture). NOT merge-ready.
Pending: reviewer source review (routed) → .80 window (booked after the
combined-candidate receipts) → Main.

### Reviewer #171 IN PROGRESS (half-perf cell @ adb1c98e9) — early blocking finding

- Confirmed: direct e886 child; exactly 3 test/docs/CMake paths; the
  supplied v2.1 body (pre-synchronized-status) exactly matches the released
  69ad object.
- EARLY BLOCKING DEFECT: Fixture::check():123-124 compares
  FP16Tensor::data() (the DEQUANTIZED FP32 cache,
  src/v2/tensors/FP16Tensor.cpp:112+) against uint16_t half-word arrays —
  not native-byte immutability; the default-order fixture's first half
  0x6c00 widens to FP32 0x45800000, so the arrays differ even with an
  untouched correct kernel. FIX: native typed_data() with exact native
  extents. More findings pending — ONE comprehensive fix-back after the
  full #171 verdict.
- Registration precision: the new correctness alias is
  Integration/ProductionTestPreflight only; smoke is Performance (the
  RELEASED plan says smoke not Unit) — the released plan governs over the
  draft's registration line. No tests run by the reviewer.

### GNU-14 register-gate receipt GREEN at 717b2bfe6 (2026-11-10, ~16:23Z)

**gate=PASS, 9/9 cells PASS** (GNU-14: Integration/Release/Debug ×
NEON/NEON_DOTPROD/NEON_I8MM; programs 64/128/176, sites 1029/1176/1274;
per-cell object SHAs recorded). Receipt: .80
/home/factory/llaminar-i8mm/build/rg-717b2bfe6/{build,receipt}/ +
rg-717b.log. Owner evidence pending reviewer authentication at the FF
request. FOCUSED SET: correctly NOT runnable in the gate build (the
register-gate configure builds only the proof targets — portability
project, no test binaries; first attempt showed 5×Not Run = expected
behavior, not a defect); running NOW from a proper Integration build at
717b2bfe6 (SHARED_CORE=ON, df15 pattern; the 6 cells incl. the NEW
InstalledWidth). FF gate after: focused set green → Main takes the
combined branch.

### Reviewer #171 COMPLETE — HOLD at adb1c98e9, five P2s, one comprehensive fix-back requested

Scope confirmed (3 paths/+288; no production edits; v2.1 body = released
69ad). Findings:
- **HP-1:** Fixture::check() compares FP16Tensor::data() (dequantized FP32
  cache) vs uint16_t arrays → use typed_data() + exact native extents
  (positive/mutation controls).
- **HP-2:** completion-discriminating fixtures all use alpha=.5/beta=.25
  (exact products; fusion/splitting indistinguishable) → add public-entry
  test-owned literal controls distinguishing rounding boundaries + absent-
  bias vs explicit +0 signed-zero (examples: e=2^-23, dot=1-e, alpha=1+e,
  bias=-1; beta=1+e, C0=1-e, value=-1); retain K257; exercise the ACTUAL
  warmup/sample/postcheck interval path with nonzero beta (no second
  timing/reset implementation).
- **HP-3:** identity — hash the SELECTED native storage (not always
  halfWords-as-physical_A_hash); record declared profiling mode (not
  hardcoded none); scope fegetround() to caller rounding + record remaining
  external/unobserved fields; Git stamp ≠ compilation authentication.
- **HP-4:** enforce timing-only selection/fresh-process mode (generic
  gtest_main runs ContractReplay's 13 configs before timing → mixed
  invocation could publish an accepted single-case sample).
- **HP-5:** whole-case admission — one admitted case owns ALL
  geometry/extents/FP-policy validation BEFORE the first fixture allocates
  (N=9999/K=10000 second-derivation and N=65536→65537 examples;
  ContractReplay positives bypass validate) + boundary rejection controls.
  Harness-admission findings, not a demonstrated production overrun.
- RETAIN: borrowed ctor/API order, borrowed-mode-per-plan, public-entry
  usage, test-owned walk/tree/completion, K257, C0-per-interval restoration,
  off-clock validation/stats. Registration settled: Performance smoke +
  Integration/preflight correctness (no Unit timing alias). Doxygen
  standard for the new source. Generated diagnostics ≠ recorded GDN
  capture; Linux validation stays outstanding.
- No builds/tests/auth performed. Await ONE exact comprehensive successor.

### Half-perf fix-back committed (2026-11-10, ~16:4xZ): 1ea0821184cb090bb2fa9bc961927df2f0a2ff7b

One comprehensive successor on codex/arm-half-perf-cell (parent adb1c98e9)
addressing #171's complete five-finding set (HP-1 native typed_data + exact
extents; HP-2 rounding-boundary discriminators + nonzero-beta interval-path
control; HP-3 selected-native-storage hash + declared-profiler + FP-scope;
HP-4 enforced timing-only fresh-process selection; HP-5 whole-case admission
before first allocation) + Doxygen standard. Hook-disabled Mac commit; Linux
validation remains merge-gating. Routed to the reviewer for #172 re-review;
.80 window follows the combined-candidate receipts.
- ADDON OWNER-VERIFICATION of the GNU receipt at 717b2bfe6 (16:58Z):
  QUALIFIES — binding verified (tree stamp = origin ref; expected_identity
  GNU-14; budget/checker hashes 6a34b344…/f180104a… IDENTICAL across hosts
  to the Mac proofs; 9/9 × 14/14 aliases; Rows1=0; grouped sites at the
  committed ceilings; provenance clean — compile_commands reference only
  /work/trees/717b2bfe6; full binding in proof JSONs). The failed rg-ee7bf
  attempt (started before the tree was fully seeded) superseded by the run
  at the final tip; failure report retained per the rule. → GNU side of the
  FF gate GREEN (owner-verified). The reviewer's comment-only ack for
  717b2bfe6 was already given (15:48Z). Remaining FF item: the focused set
  (status requested from arm-sparks — .80 seen idle at 16:52Z).

### FOCUSED SET 8/8 GREEN at 717b2bfe6 (2026-11-10, ~16:58Z) — COMBINED-CANDIDATE EVIDENCE COMPLETE

Integration SHARED_CORE=ON NEON_I8MM, 0 failed:
**V2_Integration_CPUNativeNEONPanelTournamentInstalledWidth PASSED (0.05s —
the reviewer-held cell now executes for real)** + PanelTournamentContract
Unit+Integration, PreparedProjection Unit+Integration,
PanelTailsMatchSerial, I8MMOperandGrouping, I8MMTailsMatchSDOT.

FF-GATE STATUS: COMPLETE —
1. Review chain: #161 (idea-31 rebase ACK) → #162 (PS-1 closed) → #164
   (stack re-pin ACK) → #169 (fix-back ACK, C1/C2 closed) → the 717b2bfe6
   comment-only extension CONFIRMED (#169 ACK extends to the exact tip).
2. GNU-14 register-gate receipt 9/9 (owner-verified by the addon: binding,
   hashes cross-host-identical to the Mac proofs, provenance clean).
3. Mac AppleClang-21 register-gate receipt 9/9 (owner report, #167-class).
4. Focused set 8/8 incl. InstalledWidth.
5. Ratchet clean (no grouped key lowered; budget JSON diff vs e8866ec04
   empty); composition #167-conformant; B9 strengthening deferred by ruling.
→ Main FF request sent: codex/arm-s4-nibble-witness @ 717b2bfe6 → #43
   (7 commits). Queued next: half-perf Linux validation at the FIX-BACK tip
   1ea08211 (not adb1c98e9 — the fix-back is what would merge) + idea-16
   Phase A window.

### Reviewer #172 SOURCE ACK — half-perf fix-back 1ea08211 (HP-1..HP-5 closed at source)

Two-file delta (+102/−21) verified; direct adb1c98e9 parent; the adb HOLD
retained. HP-1 native typed_data() comparisons with
positive→mutate→reject→restore→positive controls; HP-2 four public-entry
controls genuinely discriminate rounded alpha-product/bias completion, fused
beta completion (0xa8800000) and absent-bias −0 vs +0 (real half operands
incl. 0x1000=2^-11); the nonzero-beta control calls the SAME Replay::intervals()
as timing (warmup2/sample3/postcheck; resets outside timed calls);
HP-3 physical_A_hash selects actual storage, rounded-half digest separately
named, profiler declaration/unknown + caller-only rounding explicit, Git
value = snapshot not provenance; HP-4 exact SingleConfiguredCell filter +
repeat1 + count1 checked before construction/publication, mixed selection
rejected; HP-5 c(admit(x)) finishes validation before first fixture init,
N+1/extents/policy covered, boundary rejects + valid admissions present.
Retained set confirmed. Production/CMake unchanged. Registration: smoke +
Integration/preflight (no Unit timing alias).
Nonblocking #173: method Doxygen for Fixture/Replay ctors + nontrivial
helpers — scoped follow-up, does not reopen the five blockers.
Static source ACK only; Linux build/replay/preflight + GDN geometry/trace +
final bound evidence + economy = owner/Main obligations. Source-review gate
CLOSED for the half-perf cell.
- #173 Doxygen follow-up committed: **b512e25a3f4514d4f1b134d67484d2a9219a3683**
  (comments-only, scoped #173; on codex/arm-half-perf-cell after 1ea08211).
  → HANDOVER PIN FINAL = b512e25a; Linux validation (.80) targets this tip
  (comment-only delta over the #172-ACKed 1ea08211). Reviewer light
  confirmation requested per the 717b precedent. Lane pending: Linux
  validation → merge-ready → Main.

### Main's FF admission UNDERWAY (2026-11-10, ~17:20Z) — HOLD pending artifacts + one driver repair

- Main independently confirmed: exact 717b2 linear descendant; the source
  ACK covers the exact pin; production scan clean; actual range = EIGHT
  commits. HOLD pending ACTUAL artifact paths/log inspection (requested
  directly from the addon — summaries no longer suffice for the 18×14
  acceptance).
- NEW FINDING (aggregate-driver error path): register-gate.sh does not check
  the Python reconciliation subprocess exit → a malformed proof/cache
  exception can leave identity EMPTY while status stays PASS. Directive:
  fail-closed DRIVER-ONLY repair + a focused forced-failure check. Valid
  native artifacts are not automatically invalidated; the repair need not
  move 717b2/source or recompile unchanged bindings.
- Still owed to Main: actual Mac + GNU receipt/phase-log/build artifact
  PATHS for the 18×14 acceptance inspection.

### B9 strengthening READY (arm-addon3, 2026-11-10): codex/arm-b9-strengthening @ 0185f69b3 (3 test-only commits, NOT landed)

Branch off e8866ec04; each commit touches ONLY
tests/v2/unit/cmake/test_cpu_native_tile_register_evidence.py (+79 total;
checker/budget/CMake untouched → checker_sha256 UNMOVED — the B9 commit in
the combined stack is what changes it). Implements #165's three controls:
(a) 3ed993968 offset-sensitive live-slot (advance must land exactly on a
live ABI save slot; negatives: one-past, wrong-slot-with-pending, #0,
two-save per-register binding; catches no-increment AND fixed-step-8);
(b) f3faa540c actual stored-stack-pointer RELOAD through the new post-index
form (str x9,[sp] → ldr x11,[x10],#400 → str q0,[x11] = 1 spill; restore
through the reloaded pointer = 0; post-advance whole-X reload; negative
misaligned ldur); (c) 0185f69b3 narrow-read width pinned both ways against
slot [400,408) (a h→4 mutation fails ONLY this control — the exact #165
gap). MUTATION SELF-TEST (scratch copies): M1 no-increment → (a)+(b); M1b
fixed-step-8 → same; M2 h→4 → (c) alone; M3 missing-reload → (b); all 62
pre-existing analysis cases pass under M1. VALIDATION on the B9 composition
(a16591eed): analysis 64/64, binding 4/4; rebase rehearsal onto a16591eed =
one trivial conflict, resolved byte-identical, shard green. LANDING DEFERRED
post-combined-FF (checker-epoch consequence documented: full 18-cell
re-campaign at landing). Worktrees left for inspection:
/tmp/b9-strengthen-branch, /tmp/b9-strengthen-wt.

ITEM 2 (census Debug look): root-caused to the two documented same-day
mechanisms (pre-fix fixed-out-dir reuse [grep artifact] + the .80 disk-full
[Debug×3 = last cells, first victims under active fill]); no in-script
retry existed; existing fixes close both; RECOMMENDATION = document benign
(reordering Debug first is cosmetic). Pending: pin the specific mechanism
from the retained g-*.census.log files on .80 (arm-sparks to cat).
- SHA CORRECTION (addon): control (c) was AMENDED (comment-only — the
  [x10,#4] case reads [404,408), interior to the slot [400,408)) → final
  (c) SHA = **825fe8deb** (not 0185f69b3). Final strengthening SHAs:
  3ed993968 (a) / f3faa540c (b) / 825fe8deb (c). Content byte-identical to
  the validated composition; analysis shard 64/64 green after the edit.
- MECHANISM PINNED (arm-sparks verbatim evidence, 17:34Z): the census Debug
  CONFIGURE_FAILED lines = **cache-mismatch, NOT ENOSPC** —
  census-gnu-obj/b-Debug-NEON.cfg.log (+8 siblings): "The source
  /work/trees/8fcc0fe0b/... does not match the source /work/trees/46c9ffc58/
  ... used to generate cache" (the FIXED out-dir was first configured for
  46c9ffc58; every later tree failed). ZERO ENOSPC in any retained
  .err/.cfg.log (Debug-*.err = 0 bytes; CMake errored to the .cfg.log); the
  disk-full era affected the int-* builds, NOT the census cells. Item 2
  closes: single mechanism (silent fixed-out-dir cache mismatch), fixed by
  the per-cell fresh dirs + stamps; reordering Debug first = cosmetic, no
  change recommended.

### arm-addon3 Item 2 final note (verbatim, 2026-11-10)

ITEM 2 (census Debug configure, CLOSED): root cause = the pre-fix FIXED
out-dir reuse defect, mechanism CONFIRMED from retained .80 evidence
(cache-mismatch, NOT ENOSPC). `census-gnu-obj` was first configured for
trees/46c9ffc58; every later tree's `cmake -B` into that dir failed with
"The source /work/trees/8fcc0fe0b/... does not match the source
/work/trees/46c9ffc58/... used to generate cache" (verbatim in
census-gnu-obj/b-Debug-NEON.cfg.log and all 8 siblings). The
`Debug * CONFIGURE_FAILED` lines in g-*.census.log are exactly these 9
failures propagated through a summary that read stale f3b28a120 JSONs;
Debug-only visibility was a log-grep artifact, not a Debug-specific
failure. Zero ENOSPC in any retained .err/.cfg.log (Debug-*.err are 0
bytes — CMake errored to the .cfg.log before any census step); the
disk-full era affected the int-* builds, not the census cells. No
in-script retry existed; "retried" = a second invocation. FIX MAPPING
(already deployed): per-cell fresh dirs + `rm -rf` pre-clean kills the
cross-tree cache reuse; nonzero exit + per-tree out-dirs + tree stamps
kill the silent stale-JSON path. NO ORDER CHANGE RECOMMENDED: the loop
order (Integration→Release→Debug) is cosmetic — cells are independent and
rc=1 if any fails; reordering would only have moved the victim cell.
Portability CMakeLists has no Debug-specific configure work, so there is
nothing Debug-fragile to fix. Benign; closed.

FINAL SHA STATE (arm-addon3): codex/arm-b9-strengthening off e8866ec04,
tip 825fe8deb (3ed993968 a / f3faa540c b / 825fe8deb c), not landed,
byte-identical to the validated B9 composition (analysis 64/64, binding
4/4). Both arm-addon3 items CLOSED; standing by for the fold-time
re-rebase or new assignment.

### STATE CORRECTION (Main, 2026-11-10): idea-21b is PUBLISHED + natively evaluated IN the settled tip

My "compile-only candidate, not published" framing was STALE. Corrections:
- idea-21b is ALREADY PUBLISHED and natively evaluated at
  **e8866ec04** (the settled push; head = "retain narrow grouped IMMA
  occupancy on sm12"). Baseline 371f5ab35; source 12.x C32 CB4/13/24 launch
  bound 10: registers 62/56/56 → **48/40/45, zero local bytes**; resource
  tests pass unchanged ≤48/≥10 thresholds (source24 included on 12.x);
  changed TU guarded 121,120;121 full6; shipped 80/86/89/90 instruction
  bytes unchanged (only source-digest symbol salts normalized).
- Non-attached protocol ALREADY EXECUTED: cudaFuncGetAttributes +
  cudaOccupancyMaxActiveBlocksPerMultiprocessor (compiler resources +
  POSSIBLE CTAs — NOT achieved residency); existing preflight
  compiler-resource/all-codebook + paired all-format captured serial-row
  byte gates + staged-metadata captured gate; production-shaped native
  event-timed trainer (h2048/r512/e256/top8, M512 power_law, IQ2_S gate/up,
  IQ4_XS down; 24 candidates + raw samples + full-byte certificate,
  baseline vs candidate); clean owned driver windows. No NCU/NSYS/CUPTI.
- RESULT: resource+byte gates GREEN; candidate down 45 regs/10 possible
  CTAs vs 56/9; 15-sample median 3336.51 → 3315.53 µs (~0.6%, within
  noise) — NOT a Release/E2E speedup claim. Final 10-native-cell
  selection+fixture 11/11 no skips; invocation regressions 20/20; clean
  final driver window.
- EPISTEMIC BOUNDARY: PerfStats proves executed routes/work; events prove
  elapsed intervals; NEITHER yields achieved occupancy or measured
  resident-CTA counters — "do not invent such a metric"; a residency
  requirement reports UNAVAILABLE non-attached, not a failed ≥10
  potential-CTA contract.
- Durable summary: ARM_CPU_KERNEL_IMPLEMENTATION_EVIDENCE.md 'GB10 CUDA
  maintenance evidence — 2026-11-10' + the kernel-plan maintenance
  frontier. Tile5 = coverage geometry (fixed 371f5ab35). Raw evidence:
  .79 /home/factory/idea25/slot-probes/. No second candidate publication
  needed. arm-sparks' state corrected: its .79 window = B11 attribution
  (NOW unblocked — the CUDA chain is settled) + the Dir-B trace (blocked on
  Main engine item #71).

### Driver repair delivered (addon, 2026-11-10, ~17:56Z): combined tip = 252da4a53

- **Repair SHA `252da4a53`** = origin/codex/arm-s4-nibble-witness
  (force-with-lease from 717b2bfe6; ONE added file:
  tests/v2/cmake/CPUNativeTileRegisterGate.sh — the driver is now a
  CHECKED-IN TOOL). Tooling-only: reconciler exit status now checked —
  malformed proof/cache/exception fails the cell as `reconcile-crashed`,
  never PASS-with-empty-identity. 717b2's valid native artifacts untouched,
  no recompile. Forced-failure check: corrupted proof + real cache →
  `FAIL (reconcile-crashed rc=1)`; positive: full fresh 9-cell run with the
  repaired driver → 9/9 PASS.
- **Artifact paths sent to Main** (Mac archive:
  evidence/register-gate-717b2bfe6-appleclang/ = a FRESH final run of the
  repaired driver, replacing the older trial; .80:
  build/rg-717b2bfe6/{build,receipt}/, rg-717b.log, focused-717b2.log,
  f717-*.log).
- COUNT CORRECTION: e8866ec04..717b2bfe6 = **8 commits** (5 re-pins +
  InstalledWidth + functional fix-back + comment); with the driver commit
  the candidate = 9 commits to 252da4a53. Production diff scan: clean.
- OPEN for admission: the .80 GNU receipt PREDATES the driver repair (happy
  path identical; whether a re-run with the repaired driver is required =
  Main's call; a rerun is ~25 min if requested). Reviewer light note on the
  driver-script commit requested (717b-precedent class).

### ADMISSION BLOCKER (Main, 2026-11-10, ~17:56Z): the Mac "717b2bfe6" receipt is bound to c4536cf3c — WITHDRAWN

Main's artifact inspection found the Mac
'register-gate-717b2bfe6-appleclang' receipt is actually fresh-bound to
CLEAN **c4536cf3c** at /Users/floriankinder/repo/llaminar-addon (summary +
configure.command + compiler jobs): the actual PreparedProjection header
hash ad00ac6a… vs the combined 717/252's 991a4347… — i.e. PRE-S4 source,
not merely a stale Git label. REQUIRED: withdrawal + a CORRECT-SOURCE Mac
9×14 canonical qualification at the ACTUAL combined tip (clean worktree at
252da4a53; binding must prove the combined source; NO relabeling/copying).
GNU 717 artifacts: not automatically invalidated, under inspection. New
candidate 252da4a53 adds only the repaired coordinator (driver) — bounded
reviewer source ACK requested. FF HOLD remains. Count = 9 commits over
e8866ec04.

IMPLICATION: the driver's reconcile trusted the DECLARED identity and did
not catch the source-tree mismatch (Main caught it via header hashes) — a
driver-strengthening candidate (verify the configured source tree against
the expected identity, e.g. via compile_commands paths + header hash) is
now evident. The earlier owner "tree stamp = a16591eed" claim was wrong.

### Reviewer #174 COMPLETE — coordinator HOLD at 252da4a53 (three P2 defect classes; #175 tracks fixes)

The narrow defect (Python reconciler exit) IS source-fixed (lines 106–128;
nonzero reconciliation fatal). Broader coordinator defects prevent the
driver ACK:
- **RG-1 [P2] error policy:** only nounset at startup; errexit appears only
  AFTER the first reconciliation → the first cell ignores failed receipt
  retirement/copies; failed removal can leave an old artifact; failed
  overwrite/truncated copy can leave it present while the new
  proof/cache reconcile → cell stays PASS. run() in a negated conditional:
  set-e alone doesn't make its writes reliable; it returns the tool rc even
  if metadata publication failed; the inventory rc is unchecked. FIX:
  explicitly propagate cleanup/metadata/inventory/retention/publication
  errors; never publish success for merely existing artifacts; ONE stable
  error policy (no first-cell-dependent set-e transition).
- **RG-2 [P2] argument detach:** trailing unrestricted CMake args can
  detach configuration from the certified cell (an appended -B configures
  elsewhere while the driver builds/tests its original B → old cells all
  pass; an overriding -S selects a different source tree; reconciliation
  checks only compiler identity/ISA/type + tool/policy hashes, NOT the
  declared source/build root). FIX: protect coordinator-owned
  source/build/generator/cell settings; validate the ACTUAL generated
  configuration before building.
- **RG-3 [P2] no final coherent set/retention binding:** at final gate=PASS
  only the initial budget/checker hashes are reconciled; nothing proves an
  earlier cell still binds the final candidate/tool/configuration or that
  the archived bytes are those its verifier accepted (a dependency/SDK tool
  changing between cells → observations from incompatible epochs). #167:
  protected authenticated accepted artifacts/current common inputs OR final
  canonical revalidation. ARCHIVE CAVEAT: prefixed copies are not directly
  runnable receipts (the register verifier needs the original object_file
  basename + executed paths; the envelope copy renames proof.s/
  target.txt) — keep original bindings intact; authenticate any archive
  mapping; do NOT rewrite proof JSON to make relocation pass.
- ALSO: add a checked-in fast/device-free regression around the ACTUAL
  coordinator's failure/publication paths with explicit
  ProductionTestPreflight coverage (a forced exception in only the embedded
  Python block does not cover the shell/epoch paths); align ordinary
  invocation with the repository's active-Ninja pinning + unrestricted
  build/test parallelism (separate from register policy).
- Main seq115 = attributed context only. #175 tracks the fixes. Narrow ACK:
  the preserved 717 machinery + the Python-exit closure ONLY.

### Admission-blocker responses delivered (addon, 2026-11-10, ~20:21Z)

1. **Formal withdrawal + binding-failure acknowledgment:** the Mac receipt
   claimed as 717b2bfe6-appleclang was bound to c4536cf3c (pre-S4 header
   ad00ac6a…) — the driver ran against the addon worktree without verifying
   its HEAD, then was mislabeled; the earlier "tree stamp = a16591eed"
   claim was the same class. Withdrawn unchanged at
   arm-addon-notes/evidence/withdrawn/receipt-c4536cf3c-WITHDRAWN-17-19z/
   with a cause note. Nothing relabeled.
2. **Correct-source Mac 9×14 delivered:** fresh clean worktree at
   **252da4a53** (/tmp/mac-final) → 9/9 cells, 14/14 aliases/cell, Rows1=0,
   sites 1415/2003/2171, provenance verified PRE-delivery (configure.command
   + all compile_commands → /tmp/mac-final; header 991a4347…; tree-own
   budget/checker hashes). Receipt: arm-addon-notes/evidence/
   register-gate-252da4a53-appleclang/ + BUILDER_MAPPING.txt.
   Host/toolchain-bound — inspection on the Mac.
3. **#175 successor in progress** per #174: coordinator-owned override
   rejection (with spellings), configured-source validation (compile_commands
   root + production header hash vs candidate) BEFORE build, one-epoch
   input/manifest authentication at publication, explicit rc accounting
   (cleanup/metadata/inventory/retention), and a fast device-free regression
   driving the ACTUAL shell coordinator against a synthetic tree (negative
   tests: cleanup/retention/override/inventory/drift/publication) —
   ProductionTestPreflight-registered. Successor SHA + a new Mac rerun (if
   epoch semantics change) to follow.

### OpenMP wake-up/binding plan APPROVED (arm-addon3, 2026-11-10): A1 counter GO

Full plan approved with amendments: (1) A1 region-creation counter — always-
on summary (omp_region_creations_per_decode_step) + gated per-stage family;
fork sites hoisted through one inline helper in OpenMPUtils.h + the two FA
raw sites; prep-only sites count-exempt with a named allowlist; retained
teams don't count; Unit tests + a ProductionTestPreflight Perf__ cell
asserting the canonical decode graph's EXACT region inventory AND measuring
the counter's own cost; counter-off default check. (2) A2 = ONE combined
.80 rider set (per-step inventory 35B+27B, fork micro-cost attribution,
sched_getcpu idea-26 closure); x86 spin rider DEFERRED pending WSL
coordination. (3) Phase B evidence-gated: B1 = pool the serial SwiGLU batch
into the fused down launcher (19/20 cores idle-spin through it; byte-
neutral via the admitted serial fallback's exact per-expert arithmetic),
implementation gated on A2 evidence; B2 router-Q8 consolidation
evidence-conditional; B3 retained-team per layer DEFERRED; B4 spin re-tune
only with idle-cost evidence. No overlap with the register-gate/B9/idea-31
tooling set. Tracker idea-26 reframe updates via coordinator after the plan
lands.

### Half-perf cell HP-FINAL gate (arm-sparks, 2026-11-10, ~21:12Z): b512e25a GREEN

CFG=0 BUILD=0; **Unit 990/16 failed, Preflight 2419/23 failed**; **half-
operand cells 5/5 PASSED** (incl. the +249-line Perf__CPUHalfOperandProjection
rework from #173). Failure classification = the R1/R2-known base+env set,
SAME NAMES as the sealed runs at e8866ec04/7d1a94479 (cross-checked, 0 new
names). Count caveat for the reviewer: Unit 990 vs 979 earlier = BASE GREW
(failed-name SET identical). Logs: .80 build/hpb-{unit,pre,cells}.log +
hp-b512.status. Economy receipts: the hpb-cells outputs at b512e25a = the
plan-v2.1 evidence (no separate run unless the reviewer requires a specific
form). Then merge-ready → Main.

### Main corrections + tip movement (2026-11-10, ~21:13Z)

- Main CORRECTED itself: GroupedSealedPlan is NOT in 58dcdaff5 (that was
  #71 typed work classification only); #72 stub + #73 FusedFinalAdd triage
  remain QUEUED/unimplemented.
- **Latest tip: 0d6f09919** (#71 cont.): complete sparse-source + explicit
  host-service fixture; new focused preflight now GREEN; the affected
  Unit_MemoryPlanner + 20x cases = the #71 fix's own code-path reach (not
  env) — underway. The combined candidate's re-rebase target updates to the
  latest tip when #71 completes (always-push).
- idea-05 rider scope accepted; PR prose = per-member tail evidence wording.

### Half-perf FINAL BINDING RUN at b512e25a (arm-sparks, 2026-11-10, ~21:22Z) — + attribution

CFG=0 BUILD=0 (SHARED_CORE=ON, NEON_I8MM); Unit 990/16; Preflight 2419/23;
**HalfOperand cells 5/5 PASSED** (smoke+replay+NEON/NEON_I8MM). Failure
sets = the known base/env classes PLUS the #71-pending ngram/defaults class
(MTPHardwareDefaults, NativeSourceFinalRetirement,
NativeQuantisedGPUPrefillTrainerPolicy, CandidateNgram*/HostAdaptive*/
MTPTerminalHeadDefaults/QFNOrdinaryTP4HeadDefaults/PlanningHostService*/
PipelineHostService* — the stale-ngram-observation class from the
coordinator's 415/9-era focused description, exactly what Main's #71 fix
targets). COORDINATOR CORRECTION: my retarget note ("comments-only delta ⇒
near-fully reused") was WRONG — b512e25a vs adb1c98e9 includes the FULL
fix-back (+102/−21 functional rework); the early-signal run was pre-fix-
back and the final run is the real one. arm-sparks caught it. Attribution
ANSWER to arm-sparks's question: YES — the ngram/defaults pre-failures =
base-pending-#71 (Main's fix), NOT lane regressions (names known from the
415/9-era observations before the cell existed; zero other new names).
Merge-readiness package routes to the reviewer with that attribution.

### Tip + attribution updates (Main, 2026-11-10, ~21:23Z)

- Pushed tip = **a847db3286b60d7db9f59b58b23653453d944b7d** (179 above
  2948; MERGEABLE/CLEAN) — not 0d6f09919 (production code unchanged
  0d6→a847; a847 repairs the ngram FIXTURES' NUMA observations + adds PP
  terminal/lookup ownership tests).
- ATTRIBUTION REFINEMENT (Main, precise): the 415/9-era ngram failures =
  **INVALID TEST INPUTS** (CPU0/GPU3 or unknown-NUMA records, rejected
  BEFORE request-work classification) — **not environment failures and not
  code-path effects of the new classifier**. a847 FIXES the fixtures.
  (Second station-baseline-class correction: the half-perf b512 run's
  ngram/defaults names are this fixture-defect class — now FIXED
  upstream; the reviewer attributes them away from the lane.)
- Unit + expanded preflight + 20x rerun underway. **Real-model plan-only
  window DELEGATED to arm-sparks at exact 0d6** (canonical driver
  begin/finish; e886 preserved control) — the Qwen3.8-UD planning proof.
- Addon still held; re-pin onto the eventual final Main tip mandatory.
- LINEAGE CLARIFICATION (arm-sparks + coordinator): adb1c98e9..b512e25a =
  +255/−21 (2 files) = the #172-ACKed functional fix-back (+102/−21) PLUS
  the #173 comments-only Doxygen blocks (+153/0) — consistent; the
  early-signal run (adb) was pre-fix-back, so the b512 run = the real
  final-state validation. Merge-gating Linux validation COMPLETE at b512;
  package → reviewer.
- REVIEWER PACKAGE (half-perf merge-readiness): #172 source ACK at
  1ea08211 (+153-line comment-only residue to b512 — 717b-precedent class);
  Linux gate: Unit 990/16 + Preflight 2419/23 (failed-NAME sets = the
  established base/env families + the #71 ngram/defaults class; count
  deltas = base growth: 990 vs 979, 2419; the ngram names = FIXTURE
  DEFECTS FIXED at a847 per Main — attributed away from the lane);
  economy receipts = the hpb-cells 5/5 outputs per plan v2.1 (.80
  build/hp-b512.* + trees/b512e25a). Reviewer verdict → merge-ready → Main.
- Half-cell EXACT-PIN SOURCE ACK at b512e25a (reviewer): direct 1ea08211
  child, only Doxygen expansions; #173 closed. Count correction accepted:
  residue = +166/−13 (NET +153), not +153/−0; adb..b512 aggregate +255/−21.
  HP-1..HP-5 source closure preserved; NOT merge readiness: the Linux gate
  counts + failed-name attributions + a847 fixes = owner evidence, not
  green gates; the hpb 5/5 logs are uninspected owner evidence — smoke/
  correctness alone cannot certify plan-v2.1 throughput/actual workers/
  FP/profiler identity. #175 + Flash-Next #177 HOLDs independent.
- FRAMING CORRECTION (Main): the half-perf cell = **SOURCE-COMPLETE /
  ADMISSION-HELD** — NOT gate-complete. The Linux canonical gates are RED
  (Unit 16 + preflight 23 failures): red gates are not waived by name
  attribution or base growth; a847's fixture fixes do not retroactively
  qualify b512; 5/5 cells do not authenticate throughput/actual-workers/
  FP/profiler identity. Decision: **focused economy/identity evidence
  review BEFORE any acceptance**; no FF or .80 rerun booking yet; the
  review follows #71/#177 row-policy correction priority; no new lane
  execution implicitly approved. Receipts preserved.
- #71 progress (Main, ~21:34Z): tip **35a827a2a** (181 above base;
  MERGEABLE/CLEAN). Reviewer #176/#177 SOURCE ACK closes the HC row-policy
  issue: all HC/norm = full forward rows; only LM head = compact. The
  native exact-original BAD-count-expression control FAILED 2 CostModel
  tests while 3 passed (desired discrimination); corrected source restored;
  final Unit + 5-GTest preflight 20x underway. .79 owner (arm-sparks)
  preparing the corrected pin; the Dir-B trace launch held until green.
  Addon #175 HOLD unchanged; no FF.
- #71 native validation GREEN at 35a827a2a (Main, ~21:36Z): complete
  MemoryPlanner suite + 5-GTest preflight, then 20 consecutive preflight
  repetitions; the exact-old count-expression replacement control had the 2
  expected pricing failures; correct source restores green. Reviewer
  176/177 source-ACK exact 35. GREEN/READY → arm-sparks for the reserved
  .79 real-model 4096+8192 plan-only proof at 35 (canonical driver
  begin/finish). #71 completes after the independent model outcomes. Addon
  #175/bound-artifact HOLD unchanged.

### Main queue progress (2026-11-10, ~23:20Z): tip a0393b64a2 (183; CLEAN) — #72 DONE

#72 GroupedSealedPlan: the real registered script reproduced the 10
missing-field errors/12; an explicit empty physical_launch_evidence
fixture FIXED it; full registered native ARM script PASS. NO production
fallback/validator change. Docs now record the actual 35/5b proof +
handback; core production identical to 5b. #71 classification + #74
category DONE; 8192 = NotAdmitted by true capacity. #73 FusedFinalAdd
policy triage next. Addon FF admission + A1 #180 holds unchanged.
**DO NOT launch the final register campaign until Main's next SETTLED
notice** (the tip moves under it).
- #73 triage (Main, ~23:24Z): the preweight API has NO worker/degree gate;
  the byte-oracle corpus deliberately contains NaN/Inf while the default
  fail_on_nan=true correctly REJECTS nonfinite weighted rows; the fused
  low-level combine is pure arithmetic and the fused publication wrapper
  separately enforces the SAME fail_on_nan gate. => FIXTURE-POLICY
  MISMATCH, not a single-worker/column capability gap. Fix scope: the
  NONFINITE bit-oracle's diagnostic policy only, with RAII restoration +
  separate default-on rejection / finite-admission / empty-work tests. NO
  production validation weakening or optimization disable.

### #73 fix pushed (Main, 2026-11-10, ~23:28Z): e33bf1de6 (184; CLEAN)

Test-only: reproduced the exact workers1/cols1/degree2 false; confirmed the
CORRECT rejection of authored NaN/Inf under default fail_on_nan (NOT a
single-worker gap). The diagnostic byte oracle gets an explicit RAII bool
scope; separate default finite-accept / nonfinite-reject / empty /
source-bytes / policy-restore contract tests; existing preflight ISA
aliases auto-select them. NO production validation/dispatch change.
Native Unit + scalar/NEON aliases 20x + review running. Main thanks for
the DirB scope pause; no gate waiver / no implicit GPU qualification.

## MAIN SETTLED HANDOVER BASE: 1f01cb3e54ba31256239de4fba4bb8cf38604015 (2026-11-10, ~23:34Z)

#43 = 185 above 2948; MERGEABLE/CLEAN; doc-only after the test-only e33
(#73); production unchanged from 5b76a146c. #71/#74/#72/#73 DONE (bounded
scopes); #73 native Unit + scalar/NEON preflights 20x each GREEN + reviewer
#182 source ACK; correct NaN-policy fixture mismatch; no production
validation change; no metamorphic→independent-oracle/fused-publisher claim.
.79 handback acknowledged by arm-sparks; **Dir-B CPU/4096 scope still needs
the coordinator's explicit intent/prereq approval** (paused — the exact
spec from arm-sparks + Main's sign-off pending). Addon #175: owner
acknowledges the full 18×14/current-binding HOLD; the restored Mac contexts
are NOT admission. DIRECTIVES: the owner completes the comprehensive #175,
then MECHANICALLY RE-BASES + re-pins the final candidate onto EXACT
**1f01cb3e5** and obtains review — **NO FF yet** (Main decides after the
reviewed re-pin).

### Addon admission progress (2026-11-11, ~00:23Z): 2ff599ec0 pushed (FF from 252da4a53)

origin/codex/arm-s4-nibble-witness = **2ff599ec0** (completed coherent
checkpoint on the existing feature branch; no pin branch; no FF implied;
checkpoint rule honored). Content = the fail-closed reconciler repair +
RG-2 override rejection + RG-3 source-header binding/publication
authentication + rc accounting. The #175 successor (checked-in coordinator
regression + preflight registration) remains UNCOMMITTED until coherent →
one exact SHA for the reviewer. Final campaign HOLD until Main's B12-fix
SETTLED pin (post-B12 tip = the re-pin target; confirmed again). Source ACK
#184 (763) received/noted.
- (addon, ~00:34Z) codex/arm-s4-nibble-witness = **38ed21507** (FF over
  2ff599ec0; formatting-only reflow of the RG-2 refusal patterns — no
  behavior change; NO --parallel 1 anywhere; ordinary build/test
  parallelism unrestricted — confirmed). Worktree clean at HEAD → the
  committed script IS the current script (the pre-reflow digest a6861a71…
  applies to the working variant only; the committed script's own digest
  rides in each run's summary). #175 successor (checked-in coordinator
  regression + preflight registration) next; exact SHA when coherent.
  Reviewer re-check at the final pin (digest consistency between the
  reviewed pre-reflow script and the committed one must be noted).

### Bundle-1 lane RETIRED (operator, 2026-11-11 ~00:35Z) — 01 + idea-31 handover accepted

Lane 01a1bd53 retired; handover note archived:
evidence/bundle1-handover-01-and-i31.md + evidence/idea31-appleclang-secondary.md.
STATE + INHERITED RESPONSIBILITIES:
1. **LANE 01 — one mechanical step remains, coordinator executes after the
   combined FF relay**: remote codex/bundle1-perf-harness = 5d606a108;
   acceptance inputs COMPLETE (WSL final GREEN; .80 bound run graded BOUND
   @ 5d606a108); local candidate a4ebff3dd proven on 28ddbe253 (11/11 `=`
   vs 897727484; delta-equality; ctest -N; 5/5 rerun), NOT pushed. Procedure
   at execution: re-rebase --onto the then-current tip (bases were
   REWRITTEN: ce9ddb326 → 8fcc0fe0b → 28ddbe253 → settled), fresh range-diff
   + ctest -N + 5/5 rerun → proof → push --force-with-lease=
   codex/bundle1-perf-harness:5d606a108 → Main FFs. Runner container
   llaminar-b1-dev (private; cmake -S /src/src/v2 -B /build;
   CCACHE_DIR=/ccache). The only local copy of a4ebff3dd = worktree
   llaminar-bundle1-dev — clean up ONLY after the push lands.
2. **IDEA-31 AppleClang closure — REASSIGNED (queued to arm-addon post-#175)**:
   the lane's AppleClang-21 closure offer (NEON + DOTPROD + I8MM
   compile-only canonical-witness extension) does not survive the session;
   the accepted record = evidence/idea31-appleclang-secondary.md
   (0ad84ecca). The i31 worktrees + 4 builds are freed once the closure is
   reassigned (Mac session: addon lane).
3. **Cleanup after 01 push + FF confirm**: worktrees llaminar-bundle1(-dev),
   containers llaminar-b1/-dev, volumes, remaining /tmp evidence — per the
   standing rule (coordinator enforces).

## Branch-lifecycle rule + remote-branch matrix (operator, 2026-11-11)

RULE: when a session's work is fully handed over to Main (merged into
feat/arm64-native-kernels) and confirmed, the session's REMOTE BRANCH IS
REMOVED (coordinator executes the deletion after the FF/merge confirm; SHAs
remain in the tracker docs + sealed receipts). Keep refs ONLY while their
content is unmerged or carries a reviewer ACK pin still needed.

REMOTE BRANCHES (ARM perf program) + removal triggers:
- codex/arm-addon — DELETED (2026-11-11): content merged via Main's R2
  re-land cd0f95150 (11/11 `=` vs 29351..8fcc; cd0f is an ancestor of the
  current tip — verified).
- codex/arm-s4-nibble-witness @ 38ed21507 — combined candidate; REMOVE
  after Main's FF of the combined stack (its commits then live in #43).
- codex/arm-panel-tournament-policy @ e5248cba8 + codex/arm-witness-post-index
  @ f265d2950 — reviewer-ACK pins (#161-#165, #169); REMOVE after the
  combined FF + Main confirmation (ACK history preserved in the tracker).
- codex/bundle1-perf-harness @ 5d606a108 — REMOVE after Bundle-1's FF.
- codex/arm-b9-strengthening @ 825fe8deb — REMOVE after the fold lands
  (post-FF; content then in #43).
- codex/arm-half-perf-cell @ b512e25a — REMOVE after the half-perf
  handover/merge.
- codex/arm-idea26-openmp-wakeup @ 16d633c15 — REMOVE after the A1/OpenMP
  item's handover.
- feat/neon-s4-nibble-lifetimes @ 972c0081e — frozen idea-31 evidence pin;
  REMOVE after the combined FF (content re-derived in the stack).
- codex/arm-perf-review — PERMANENT (the tracker; never removed).
NON-ARM branches (feat/continuous-batching*, feat/mxfp4*, feat/qfn*,
feat/qwen*, feat/clover*, feat/native-*, claude/*) belong to their own
programs — not removed by the ARM program; their owners follow the same
rule at their own completions (informed).

## Qualification-scope rule (OPERATOR DIRECTIVE, 2026-11-11 ~01:00Z)

The per-step qualification burden was too heavy. STANDING RULE, effective
immediately for all ARM-program lanes:
1. **Lanes test ONLY the related changes** — focused tests on the touched
   surface + the direct controls/regressions for the change. NO full-suite
   portability/preflight/census runs as routine lane evidence.
2. **FULL qualification belongs to MAIN, once per COLLECTION** — after a
   batch of changes is merged into feat/arm64-native-kernels (e.g., the
   combined FF), Main runs the complete qualification ON ITS TIP: portability
   matrix, preflight, census/register campaign, cross-config receipts. That
   single post-collection campaign is the actual gate — not per-lane
   full-suite runs.
3. **Reviewer verdicts judge focused evidence** — sufficient to reason about
   the changed surface; the reviewer does NOT demand whole-suite execution
   from lanes. Unexecuted whole-suite items are marked unverified-at-lane
   and covered by Main's post-collection campaign.
4. Consistent with the established principle: receipts don't compose across
   checker epochs → ONE collection + ONE campaign + ONE receipt set.
Rationale: per-step full qualification dominated wall-clock and disk (the
.79/.80 disk incidents); correctness at the tip is what ships.

### Main ADOPTS the qualification-scope rule (2026-11-11 ~00:56Z)

Adopted verbatim: lanes supply focused/related evidence + an explicit
unverified-whole-suite list; Main owns ONE full qualification campaign on
the exact final-COLLECTION tip before qualified/SETTLED/final-FF
declarations. No per-lane full-matrix/preflight/census/cross-config
campaigns; Main retracts prior requests for lanes to run the final full
campaign themselves. #166/#167 CONTENT UNCHANGED: the final Main-owned
combined gate is still the full 18 cells × 14 canonical aliases / current
bound programs + applicable full gates; no source-equality or
receipt-splicing substitutes. The current B12 exact-5 + new-contract-20x
run on 763 = FOCUSED related-source-fix evidence, NOT full certification.

### Host-ownership settlement (Main → arm-sparks, ~04:41Z)

Main explicitly handed BACK .79 queue ownership to arm-sparks under its
own bookings (Main's read-only inspection complete; B12 closed; no Main
jobs active). NOT new trace/model/GPU authorization; all contexts/raw
bytes preserved; no more epoch-mixed packets — Main's project record is
authoritative (core a9cb…, module c615…, 8 actual tests, 400-each overrun
recorded). Program state: SOURCE BASE 880429059; #191 still blocks the
collection. arm-sparks' next queue: nsys install + attach smoke test
(operator-authorized, idle-time), then idea-16 Phase A / idea-14 B wave /
Dir-B rerun / Qwen3.8 trace — all gated on the combined FF.

### CB C1 → C2 replacement handoff (2026-11-11 ~09:11Z)

The DGX-station worker (01a1c1d0) accepted the C1 implementation/
proof-repair replacement with all constraints binding: capacity verified
free; no displacement (llaminar-cb-c1 @ db13d5726, refcap @ 28ddbe253,
GLM serving, other lanes' worktrees untouched); / at 90% (177G free —
ASK BEFORE PRUNING ANYTHING, incl. historical build trees + the 22GB
policy-rejected model); isolated worktree llaminar-cb-c2 @ db13d5726 from
the shared .git (NO rebase/cherry-pick); established CPU-only recipe
(devcontainer + .venv, Integration, SHARED_CORE=ON, bounded parallelism
coordinated with arm-perf); canonical FULL Unit + preflight; gate
discipline verbatim (genuine exit 0 before commit/push; no parity-as-
authorization; no hook tricks; no refusal-to-fix; escalate contradictions;
no PR/merge/GPU/GLM/unauthorized downloads). Execution GO issued;
detailed surviving-findings handoff follows from the C1 arc. WSL
replacement plan = CB coordinator's call, not the worker's.

### OPERATOR THROUGHPUT DIRECTIVE (2026-11-11 ~09:45Z) — no idle-while-queued

The operator re-stated the operating model and called out the throughput
failure: queued tasks sat behind the FF while sessions idled. BINDING
CORRECTIONS:
1. Operating model: task → session works it autonomously end-to-end →
   hand fully-merge-ready to Main → Main applies to its branch → done.
   Validation on the Sparks hosts where needed. Qualification per the
   standing scope rule (lanes = focused/related evidence ONLY; Main = ONE
   full campaign per collection on its tip).
2. The FF is an integration formality, NOT a content gate: the B12 fix is
   verified IN base 880429059 (8/8 on the real host), so measurement and
   implementation on the current tip are valid; the anchor-refresh rule
   covers tip movement. NO task waits for the FF to START.
3. Assignments executed immediately: reviewer → half-perf economy/identity
   review (queued since source-complete); addon2 → idea-16 Phase A on .80
   NOW, then idea-08 Phase A; arm-sparks → host queue (.80 Phase A, .79
   Dir-B trace rerun per its ACKed scope, nsys bounded-fact read, idea-25
   runbook prep); addon → Lane-2 scope (ideas 03/04/11) on 880429059
   immediately after #191 ACKs; Main → integrate as work arrives + ONE
   campaign when the collection completes → FF → close #58.
4. Coordinator duty: keep every session holding a current task; escalate
   stalls instead of gating them.

### Post-merge campaign slot plan (arm-sparks, ~11:18Z)

Counter collection MERGED (bb893cd79 on feat/arm64-native-kernels, PR #43
204 commits). Main's ONE post-merge campaign being staged: .80 frees
≈11:50-12:00Z (Phase A matrix + K14-I16A seal + A2 riders); .79 frees
≈11:55-12:05Z (bb rebuild + Dir-B trace). Sparks recommends .80 first
(warmer ccache, 429G free vs .79's ~66G — the campaign build ≈25-35G at
SHARED_CORE=ON fits .80 comfortably; .79 keeps only the trace build).
Inventory: g++-14/GNU 14.2.0, image llaminar-arm-engine-sdk-py:e0d6d34e7
(.80) / SDK-cu13 (.79), NEON_I8MM, fresh build-campaign-bb dir, ccache
warm. LAUNCH = first host free AND Main's campaign definition (exact
CTest invocations/labels/parallelism) — whichever is later. Mac
portability receipt needs a Mac-side executor assignment by Main.

### MAIN CAMPAIGN DEFINITION published (~11:23Z) — post-merge, model-free, on bb

Main's ONE post-merge ARM CPU campaign on exact bb893cd79 (MERGED tip):
- .80 locked at next free slot (~11:50-12:00Z); A2/idea16 bookings preserved.
- Fresh checkout with usable in-container git metadata; no corpora/LFS.
- Fresh Integration build; e0/763/35/5b contexts PRESERVED. GNU14.2,
  NEON_I8MM, SHARED_CORE=ON (actual option name), compile DB export,
  HAVE_CUDA=OFF/HAVE_ROCM=OFF (omitted backends recorded NOT QUALIFIED).
  Active Ninja resolved once via CMAKE_MAKE_PROGRAM:FILEPATH; cmake
  --build only; no ad-hoc OMP/NUMA/capacity workarounds.
- PRIMARY UNMODIFIED AUTHORITY: scripts/ci/run_production_prerequisites.py
  --build-dir --output (complete ^V2_Unit_ namespace + ProductionTest
  Preflight inventory; full Unit then canonical preflight resource lanes;
  receipt/log/JUnit retained; Unit parallelism = full real affinity 20;
  no focused-panel flag/regex subset/prior receipt/no parallel caps/no
  repeats/no second suite).
- Pre-build: cmake --build --parallel --target llaminar2 v2_unit_gate
  v2_production_test_preflight_gate -- -k 0 (real exit preserved).
- Coverage assertions from ACTUAL discovered/JUnit inventory:
  V2_Unit_MPIBootstrap (incl. cost body), V2_Unit_OpenMPTeamForkStage
  Evidence, V2_Integration_OpenMPTeamFork{CounterContract,CounterPubli
  cation}, V2_Integration_OpenMPForkEvidenceEnginePath. Owners:
  v2_unit_mpi_bootstrap, v2_test_forward_execution_engine,
  v2_unit_openmp_team_fork_evidence + shared workshare.
- Save all identities + CTest JSON + canonical prerequisites JSON/log/JUnit.
  Zero selected/skips/missing ≠ pass; failures reported honestly, FIXED
  FORWARD (no weakening/skipping/rewriting).
- Release context preserved; app built only for booked measurements — NOT
  a second campaign. No additional models/profiler work.
- MAC portability: Main runs locally as the SAME campaign leg (count from
  CTest, not hardcoded 295) — not sparks'; .80 not held for the Mac.
- Executor: arm-sparks under the coordinator booking; launch when .80
  frees; ACK paths + planned start first.

### MAC CAMPAIGN LEG LAUNCHED (Main, ~11:31Z)

Main launched the Mac portability leg NOW via its owned monitor: isolated
clean bb893cd79 detached checkout (/tmp/llaminar-main-bb-source; full
Integration NEON AppleClang-21/libomp/GTest/active Ninja), unrestricted
full build (-k 0) → complete portability CTest inventory / all tests with
logs/JUnit, no subset/count hardcode. Results:
/tmp/llaminar-main-bb-mac-results. ONE campaign leg — NOT lane
qualification / 18-gate. .80 definitional already delivered (canonical
full Unit/preflight); awaiting the planned slot ACK. Old scheduled prompt's
#58-OPEN state ignored: #58 CLOSED NOGO — no apparatus work.

### MAC CAMPAIGN LEG COMPLETE + VERIFIED (Main, ~11:34Z)

Exact MERGED bb clean isolated source; AppleClang-21/libomp Integration
NEON; full build -k 0 unrestricted PASS; **complete portability CTest
295/295 PASS in 22.31s**. Main independently parsed inventory + JUnit:
EXACT 295 names, zero failures/errors/skips/embedded zero-or-skip markers,
raw 295 Passed records; exit phase=completed 0. Source checkout still
CLEAN. Retained: results /tmp/llaminar-main-bb-mac-results/, build
/tmp/llaminar-main-bb-portability/, source /tmp/llaminar-main-bb-source/.
SCOPE: Main post-merge campaign MAC LEG — NOT Mac engine/full-backend/
economy qualification. .80 canonical full Unit/preflight Engine leg
PENDING the booked slot. PR status updating hash-free.

### OPERATOR DECISIONS (2026-11-11 ~13:10Z) — recorded

1. **nsys wrapped smoke: APPROVED.** One booked window:
   gpu_driver_diagnostics begin → nsys profile on the trivial vectorAdd
   (no model/inference) → finish, with REAL begin-before/after-finish per
   the custody rule. Booked by coordinator/arm-sparks at a clean slot.
   NCU stays separate/unlifted.
2. **B4 LM-head shortlist contract: routed to the REVIEWER for expert
   input** (operator: "ask the reviewer, it has more knowledge about it")
   — reviewer advises on the contract shape (greedy exactness vs
   stochastic semantics boundary) before any unfence/design.
3. **idea-14 27B rider: APPROVED after the 35B result** — one matching
   rider (same controlled-reference protocol) once the 35B greedy rider
   settles the sampling-mode question.

### .80 CAMPAIGN LEASE TRANSFERRED TO MAIN (coordinator, ~14:35Z)

Binary question expired without a compliant answer → **Main holds the
exclusive .80 campaign lease** (conditional idle verified by direct read
14:30Z: dockerps empty). FACT: the sparks 480 focused attempt #3 PASSED
4/4 at 14:23:30 (attempts 1/2 = wrong cmake --build target syntax; the
17-step build + 4 passes valid — logs copied by Main). Main is preparing
the canonical FULL 480 continuation on the existing fresh 480 contexts
(BB preserved), new non-existing output. ALL .80 ACTORS: no launch/write/
build until Main hands the lease back in writing; sparks quiesce ACK
requested; queued items (greedy rider, M=1 window, B1 focused checks,
Lane-2 window) locked behind the campaign. No new experiment/qual
model.

### .80 LEASE EXECUTION UNBLOCKED (coordinator, ~14:36Z)

Sparks' definitive state (third message): **quiesced — nothing launched/
written/built/touched on .80 since the lease transfer**; awaiting Main
handback. The earlier "launching the full wrapper NOW" language did not
materialize into a running process. FACTS STANDING: focused epoch-480
phase completed BEFORE the lease transfer (attempt #3: BUILD3=0, 4/4
PASS at 14:23:30 — Main's copied logs; readelf standalone = no core-DSO;
BB evidence preserved untouched incl. prerequisites.json 13:37:27Z +
preserve-bb-failed/ snapshot). Available to Main: sparks' staged
campaign480-full.sh wrapper + fresh 480 contexts (build-campaign-480,
campaign-src-480 @ 480de4344, ccache-warm) — use-or-replace is Main's
call. Main proceeds with lease execution; own read-only dockerps at
start per its standard practice. Sparks quiesced until lease returned
in writing; lockbox (greedy rider, M=1 window, B1 checks, Dir-B, nsys
smoke) queued behind the campaign.
ONE-BIT CONFIRMED by sparks: the full campaign wrapper was NEVER launched
("resuming NOW" = plan statement superseded by the lease transfer; message-
sequencing error, admitted). No wrapper process/container; dockerps empty.
.80 CLEAR for Main; relinquish stands. Lease execution unblocked.

### **MAIN CAMPAIGN LAUNCHED (Main, ~14:54Z) — epoch-480 canonical, exclusive lease**

**Main480 continuation ACTUALLY LAUNCHED 14:52:19.599810845Z** on .80 under
the exclusive lease: CID 5bab3cb5cce1a7bcb3d427a08aa289f70794d1d3b9d75d9
386c44db01a3fc451, name main-campaign-480, host PID 2207511, image
c0ca1dc9…; source 480 CLEAN verified; OUT campaign-480-results ABSENT
before launch (no pre-create). Phase BUILD now. Main's thin phase-gated
literal runner /work/build/main-campaign480.sh (actual file transferred +
bash -n), logs /work/main-campaign480-{launch,build,driver}.log + actual
phase/exit. Canonical UNMODIFIED driver after the full target build; no
duplicate. Runtime qualification profile EXPLICIT UID 1000:1000, private
IPC/default private PID, no network, SDK deps RO, SYS_NICE + seccomp
unconfined (the repository NUMA/page-owner test-harness setup) — NOT a
claim that old BB flags match or C1-restricted-profile qualification
(old BB perms not reconstructed). No host driver/OMP changes or waived
tests. .79 lease NOT needed by Main — its booking continues through
arm-sparks under prior permissions.

### .79 QUEUE LOCKED (sparks, ~15:02Z)

Dir-B trace runs NOW on .79 (build-bb Release @ bb893cd79 BIN-OK; two
authoring fixes in; ~35 min; seal + per-bucket table on completion).
Then: half-perf artifact bundle (read-only; includes the FMLAL
native-leaf-existence answer from the tree grep) → nsys wrapped smoke
(5-min, operator-approved, custody rules: REAL begin before attach,
finish after, full identity capture — clean capture = nsys attachment
PROVEN → idea-25 trace bookable) → B1 focused check on .79 if disk fits
after pruning superseded dirs (Release ≈10G + smoke ≈1G within 66G).
.80 = Main's lease, untouched. Post-campaign .80 order locked: greedy
rider → idea-08 BOM census + tournament → M=1 economy → Lane-2.

### OPERATOR DECISION QUOTED (Main's request, ~20:47Z): campaign scope

The operator's campaign-scope decision was given via the coordinator's
structured question (~20:15Z), answer verbatim: **"Accept qualified-
relative (Recommended)"** — selected for the question "The campaign at
0924175d: 25 failures remain — ALL inherited/environment (0 new, 10
collection-introduced fixed). Main holds the .80 lease and asks your
scope decision..." That is the literal operator instruction record;
Main's execution of it (lease return, inherited-dispatch) follows it.

### LITERAL .80 SNAPSHOT 20:46:53Z (Main SSH read-only) — recorded

(1) docker ps EMPTY — no greedy rider running yet. (2) **A2 EXPORTS
LANDED:** host /home/factory/llaminar-i8mm/a2-bb893cd79/ (container
/work/a2-bb893cd79/): a2-35b-mtp.{json,out}, a2-35b-nomtp.{json,out}
(13:17-13:18); a2-27b-mtp.{json,out}, a2-27b-nomtp.{json,out} (13:50-
13:53); plan35/plan27 {json,log}; a2-r3-serve.log + a2-r3-ready.txt
(14:32); a2-cfg.log + a2-build.log; runner build/a2run27.sh. NOTE: the
A2 riders ran at the bb893cd79 tree (dir name), not 880429059. Main has
not authenticated these artifacts. (3) Campaign evidence retained:
12675c956 (build-campaign-1267, campaign-1267-results, receipt + unit/
integration-{host,exclusive,cuda-socket,rocm-socket} xml/log, cfg/build/
driver/launch logs + exit.txt, source campaign-src-shared2 @ 0924175d);
480 (campaign-480-results, build-campaign-480, main-campaign480-*);
focused/diagnostic dirs (build-focused-ae76 [failed first candidate],
build-focused-fix2, build-diag3, build-a1fix, build-a1fix2, build-shared1
+ their ctest logs/xml); mpi480-baseline.{log,xml}. Local copies:
/tmp/llaminar-main-bb-linux-results/.

### STATE-AUDIT VERIFIED SNAPSHOT (sparks literal, 20:50Z) — reconciled clean

.80: **greedy rider RUNNING** (container 15b95d465be2, launched 20:41Z;
phase = MTP-dynamic n=256 benchmark after PLAN=0; plan-rider.json/
rider.out → /work/results/i16rider/). A2 exports complete at
/work/a2-bb893cd79/ (checker identity 176/27/54 in the run log) —
delivered to addon3. .79: Dir-B sealed (K14-IDEA25-Q38-TRACE-CPU4096-
20261111); B1 focused check = NOT a .79 item (addon2's W2 at edcce0183
is booked on the .80 post-campaign queue — W2 after W1; the earlier
.79-offer superseded by the coordinator's locked order); .79 otherwise
idle. RECONCILIATION: matches the locked order + lease state; no drift.
Verified snapshot timestamp: 20:50Z.

### STANDING DIRECTIVE (operator): keep the PR-branch pace

The main PR (feat/arm64-native-kernels) must not sit static for hours
while lanes churn. Whenever a lane item reaches a clean reviewer verdict
+ passing validation, IMMEDIATELY trigger the handback → Main
application path; Main states application intent as soon as a candidate
goes clean (don't wait to be asked). Coordinator duties: maintain the
collection-candidate inventory, route clean items to Main proactively,
keep at least one item in the handback pipeline at all times.
Validation cycles that catch real defects justify slow PR movement —
a CLEAN item idling in a lane branch is coordinator neglect.
Current candidates (2026-11-11 late): FMLAL leaf (one K=16 residue),
B9 fold (verdict requested), B4 diagnostic bench (round-5 verdict
pending → W3 JUnit).

### PR #43 REBASE COMPLETE on Main's side (~23:55Z); campaign #2 booked

207 commits replayed onto base d1fd0a077, ~17 conflicts resolved;
force-push pending validation; new tip to be announced. Campaign #2
(the post-rebase qualification) GRANTED on .80 immediately after the
B1 attribution export (seal K14-B1ATTR-12675C956); W3 (B4 diagnostic)
RE-QUEUED behind it. **Reviewer rule now explicit: source verdicts do
NOT transfer across the rebase — FMLAL/B4/idea16 lane tips rebase onto
the new base at handover and their verdicts re-pin.** FMLAL K=16
repair tip should land ON the new base if the announcement precedes
its push.

### REBASED TIP = dd8f2bd96 (Main, ~23:56Z); campaign #2 STARTED on .80

Main announced the rebased tip dd8f2bd96 (207 commits onto d1fd0a077)
and GRANTED-started campaign #2 immediately (the B1 attribution
export's exit was UNCONFIRMED at grant time — sparks overdue on the
audit answer; the export re-queues post-campaign if it didn't
complete). Force-push of dd8f2bd96 = the rebase-anchoring base for
FMLAL/B4/idea16 at handover; verdicts re-pin.

### CORRECTION (Main, ~00:22Z): dd8f2bd96 NOT pushed — rebase only on the force-push announcement

dd8f2bd96 exists only as Main's local replay; Main pushes after the
campaign validates it. Lanes must NOT rebase onto it until the
force-push announcement. FMLAL sequencing revised: hold tip 67e8856c9 →
on the announcement, rebase → push → validation #5 → verdict (one
cycle). Campaign attempt 1 failed at configure: the fresh clone lacked
the ignored external/onednn + network-none → clone failure (environment
class, same as the station's solved blocker); preserved in
preserve-dd8-attempt1/; external/onednn + external/cnpy copied from
campaign-src-shared2 (tree still clean); attempt 2 = container
main-campaign-dd8b started 00:21:32Z.

### REBASE CANDIDATE SUPERSEDED (Main, ~00:49Z): dd8f2bd96 → ebd66f0cb

Main's attempt-2 configure exposed a resolution defect in the dd8
replay: a tests/v2/CMakeLists.txt conflict shared a trailing endif()
between the base's new block and ours → one if() unclosed across 63
commits. Fixed with an autosquashed fixup into the originating commit
(every commit's CMakeLists balances; tip diff vs dd8 = +endif() only).
NEW CANDIDATE: **ebd66f0cb** (tree 735712de3, still 207 commits on
d1fd0a077). Campaign attempt 3 = container main-campaign-ebd (started
00:48:23Z); attempts 1-2 preserved. **LANES: rebase onto ebd66f0cb (not
dd8) only after Main announces the push.** Relayed to addon (FMLAL had
rebased locally onto dd8 — redo onto the announced SHA).

### CAMPAIGN #2 RESULT (~01:38Z, ebd66f0cb): exit 8, 3535 tests, 17 failed / 0 skipped

- **20 inherited failures FIXED by the base** (the base absorbed most of
  the qualified-relative inherited set).
- 12 NEW failures, classified by Main:
  (a) Rebase interaction — FIXED in **74797687f**: 3 new base GTest
  bodies price GPU-native objects and fail CPU-only (5 aliases +
  MemoryPlanner + PhysicalMemoryAuthority) → guarded HAVE_CUDA||HAVE_ROCM
  + REQUIRES_BACKEND GPU; base routes --mpi-procs ranks into cluster
  thread config needing original-entry native admission (MPIBootstrap ×2)
  → applied only when admission exists; duplicated kwarg from auto-merge
  (GroupedSealedPlan).
  (b) Base topology (inherited, not fixed): **ShmemSpinAbsorbingAbortReplay**
  — the base now rejects two CPU-expert ranks on one NUMA node (.80 has
  one) = same class as CPURootedGatherParallelCopyReplay. WATCH: semantic
  interaction with the branch's merged ShmemSpin shared-partition fix —
  if #3 still shows it, route the analysis.
  (c) Suspected flake: KVStateLifecyclePolicySelfTest (exit 255; same
  files passed at 1267) — recheck in #3.
- Unchanged inherited: 4 ROCm trainer fixtures + the 2-socket cell.
- **Campaign #3** on the final tip 74797687f (tree 9879e16dc), container
  main-campaign-747, started 01:37:54Z — fix-forward continuation within
  Main's window; push only if no branch-introduced failures.

### CAMPAIGN #3 RESULT (~02:19Z): 0 branch-introduced failures — FORCE-PUSH LANDED

Campaign #3 (74797687f): 3532 tests, **6 failed, 0 branch-introduced**:
- 4 ROCm trainer fixtures (base deleted them) — inherited;
- CPURootedGatherParallelCopyReplay — the 2-socket cell (hardware-bound);
- ShmemSpinAbsorbingAbortReplay — **RESOLVED as base topology validation
  vs a single-NUMA host**: the branch has ZERO diff vs base in
  NativeCPUCollectiveProgram.cpp + Perf__CPURootedPublicationTransport.cpp;
  the test builds one member per inventory rank, .80 has one NUMA node,
  base d1fd0a077's new check rejects before any ShmemSpin code runs —
  independent of the 12675c956 pinning fix; needs a two-NUMA host (same
  class as the 2-socket cell).
- KVStateLifecyclePolicySelfTest PASSED in #3 → one-off flake confirmed.
- GroupedSealedPlan + MPIBootstrap ×2 now pass.
**PR #43 FORCE-PUSHED: feat/arm64-native-kernels = 74797687f (tree
9879e16dc), lease from 12675c956; 207 rebased + 1 reconciliation commit
on d1fd0a077; GitHub MERGEABLE/CLEAN. Lanes rebase onto 74797687f.**
Raw: .80 campaign-747-results/, /tmp/llaminar-main-bb-linux-results/
campaign747/. PR body updated.
NEXT: FMLAL landing cycle (addon rebases onto 74797687f → push → #5
first-priority on .80 after the B1 export → handback → Main applies).

### SECOND UPSTREAM REBASE IN FLIGHT (~09:20Z): operator-requested, base ca9902d3f

Main rebased PR #43 onto the new upstream base ca9902d3f (6 commits by
David): candidate **ae51ce3f0** (233 replayed commits + 1
reconciliation). One semantic conflict: upstream's caller-mask restore
in configureClusterRankThreads — the branch requires native core
admission there and keeps the bound master's permanent place; the
upstream test adapted to assert unbound requests are rejected.
**Main's validation running (main-rb2, started 09:20:33Z, ~60 min,
Integration NEON_I8MM); the M=1 window waits; the M=1 re-anchor follows
Main's new-tip announcement (not 7a4790c77 — the base changed).**
Lanes rebase onto the announced tip after validation.

### BRANCH CLEANUP EXECUTED (~09:55Z): three merged-and-confirmed branches deleted

Per the branch-lifecycle rule (removed once content merged + confirmed):
- codex/arm-fmlal-half-leaf DELETED (content = PR #43's 0ad09600b, FF'd);
- codex/arm-b4-vocab-budget DELETED (content = PR #43's 7a4790c77, FF'd);
- codex/arm-idea26-openmp-wakeup DELETED (merged in the counter
  collection; deletion queued since the campaign).
KEPT: codex/arm-perf-review (PERMANENT tracker); codex/arm-b1-fused-
fork-pooling (9f96536a6 — the B1 pooling WIP, in Main's structural
pass); codex/arm-idea16-moe-decode (5c2cef47a — the W2/B1 bench chain,
window material); codex/arm-b9-strengthening (108bce2e — closed/
benched, the preserved artifact per the operator); codex/arm-s4-
nibble-witness (7501904e8 — frozen unmerged tooling per Main); codex/
bundle1-perf-harness (5d606a108 — the Bundle-1 final hop pending);
codex/arm-half-perf-cell (b512e25a — frozen, superseded by the landed
FMLAL leaf; deletion candidate awaiting an operator call since the
content is unmerged-obsolete, not merged).

### PR #43 REBASED AND PUSHED (~09:56Z): new tip ae51ce3f0 (lease from 7a4790c77)

The operator-requested upstream rebase (base ca9902d3f): 233 commits
replayed, 3 conflicts (2 mechanical: the CMake preflight list + two
independent ForwardExecutionEngine tests; 1 semantic: upstream's
ScopedCallerThreadAffinity in configureClusterRankThreads dropped —
the branch requires native core admission + keeps the bound master's
permanent place; the upstream MPIBootstrapCallerAffinity test adapted
to assert unbound requests are rejected + caller/child/workers keep
the rank mask). **Validation (main-rb2): build OK; focused affinity
cells 8/8; canonical 3568 tests (34 new from the base), 0 skipped, 6
failed = the same inherited set (0 new). MERGEABLE/CLEAN.**
build-rb2 pruned. **ALL LANES rebase onto ae51ce3f0, incl. the B1 WIP
(codex/arm-b1-fused-fork-pooling; the "B1 authorize" quote accepted;
Main takes the runner-placement pass on a rebased copy).** .80 free →
the M=1 window (re-anchored to ae51ce3f0).

### OPERATOR DIRECTIVE (~10:00Z): the base-branch owner will merge to develop — stabilize + complete-open only

The base-branch owner: "I will finish running the Opencode harness tests
on the topologies and then merge to develop." **Program priority:
(1) COMPLETE what is open; (2) address ONLY critical items needed to
stabilize the ARM branch; (3) DON'T START new work.**
CLASSIFIED:
- CRITICAL: main-rb2 (the upstream-rebase validation — running) → the
  new-tip announcement → the lane-rebase wave.
- OPEN (complete, don't escalate): the W2 collector-fix re-run → the B1
  seal → the idea-08 inputs (the tournament + census = the booked W2
  payload); the M=1 window re-run (re-anchored to ae51ce3f0); the B1
  pooling cycle (authorized; Main's structural pass + validation);
  idea-23 stage-1 fix (dispatched, small); the model download
  (passive); B1ATTR2's analysis.
- HOLD (don't start): the 27B rider; the Bundle-1 final hop (parked —
  not stabilization-critical); the idea-04 implementation decision
  (diagnostic evidence only until post-merge); any new windows/ideas;
  MXFP4 (gated on the merge — post-merge phase).

### SESSION RETIREMENT: arm-addon2 (~10:07Z) — retirement-ready after cleanup

Everything LANDED/PUSHED/handed: the B4 diagnostic (PR #43 at 7a4790c77
+ the record corrections fcf2dde9b); codex/arm-idea16-moe-decode @
5c2cef47a (the complete bench line: the B1 STT bench + the counter fix
+ the collector enablement + the compact tournament + the admission/
backend-init + the GCC-14 fixes — the W2 re-run's entire payload origin-
visible); the tag b4-diagnostic-landed @ 7a4790c77 preserves the pinned
SHA. PENDING (owned by live executors): the W2 re-run (sparks, booked);
the B1 attribution (delivered — the 4-placement completion rides the
seal); the idea-08 GO/NO-GO recommendation (inputs only — the successor
session writes it from the seals); the B4 spec of record 10137aa3 (v4).
CLEANUP APPROVED: /private/tmp/idea16-wt + /private/tmp/b4-wt +
build_stt removed (nothing local-only). **arm-addon2 retires. The
idea-08 GO/NO-GO recommendation (inputs-only) transfers to the
successor session or a re-wakened addon2 when the census + tournament
data land.**

### arm-addon3 RETIRED (~10:09Z): worktree + build dirs removed (~500MB freed), signed off

The successor session picks up from the handover note (the B1 runner
placement via Main's structural pass; the named-attribution export
pass with the corrected recipe; the B1 ranking finalization; the
idea-23 model-level validation post-merge; the depth sweep + 27B rider
post-campaign). Two addon sessions retired tonight (addon2, addon3) —
both fully accounted.

### LANE STRUCTURE CHANGE (~10:12Z): the addon lanes on this host are GONE (operator-confirmed)

arm-addon, arm-addon2, arm-addon3 are all gone (addon2/addon3 retired
cleanly with full handovers; arm-addon gone per the operator). The
remaining structure: arm-perf (coordinator), Main, arm-sparks (the SOLE
host-execution lane for .79/.80), arm-reviewer, the station session
(c290), + the CB program. **Artifact survival: the addon-notes dir
(/Users/floriankinder/repo/arm-addon-notes/) SURVIVES on disk —
window-m1-activation.sh, w-m1act-summary.py, the idea-04 M=1 kit, and
the handover docs are intact; sparks executes the M=1 window from that
script + their own corrected m1.sh fixes.** Work routing: implementation
→ Main; host execution → sparks; verdicts → the reviewer; station ops →
the station; analysis packages → the coordinator or a future re-spawn.

### OPERATOR CLARIFICATION (~10:27Z): performance changes are ALSO critical — the bar is same-or-better

"performances changes are also critical. we need stablility, once this
is ok, then we need same or better performance, anything worse will get
rejected!" — REFRAMES the stabilize-only directive: (1) **B1 pooling
proceeds NOW** (the "B1 authorize" grant + this clarification; the
landing timing follows the evidence — pre-merge if the validation
completes with same-or-better evidence, else immediately post-merge);
(2) **the acceptance bar for everything: same-or-better performance,
anything worse REJECTED** — the M=1 signal, the B1 non-regression, all
future changes; (3) the stability-first ordering stands (correctness
gates → performance evidence).

### ARM BRANCH SURFACE CLEANUP (~10:40Z, operator: "take care of all those old branches created for arm")

FULL ENUMERATION + CLASSIFICATION of every ARM-related branch on origin:

DELETED (1):
- codex/arm-half-perf-cell @ b512e25a — SUPERSEDED by the landed FMLAL
  leaf (the half-product path is implemented + landed with the full
  evidence chain; the half-perf-cell draft is obsolete unmerged work).

DELETED EARLIER TONIGHT (3): codex/arm-fmlal-half-leaf (landed
0ad09600b); codex/arm-b4-vocab-budget (landed 7a4790c77); codex/
arm-idea26-openmp-wakeup (merged, deletion verified clean — the
exemptions cleanup is in the collected lineage).

KEPT (each with its reason):
- feat/arm64-native-kernels @ ae51ce3f0 — THE PR (MERGEABLE/CLEAN,
  awaiting the base owner's develop merge).
- codex/arm-perf-review @ 22633bd05 — the PERMANENT tracker.
- codex/arm-idea16-moe-decode @ 5c2cef47a — the W2/B1 bench chain (the
  open W2 payload: the B1 seal re-run + tournament + census executes
  from it; deletable after the idea-08 inputs complete).
- codex/arm-b1-fused-fork-pooling @ 9f96536a6 — the B1 WIP (superseded
  by Main's corrected-target implementation d5e04fdce, but the tile
  extraction + test file are references; deletable after Main's B1
  lands).
- codex/arm-idea23-hc-planning @ 640d625cb — parked per stabilize-only
  (applies post-merge if the model-level run confirms a gap).
- codex/bundle1-perf-harness @ 5d606a108 — the parked Bundle-1 final
  hop.
- codex/arm-b9-strengthening @ 108bce2e — the benched B9 artifact
  (closed topic, preserved per the operator's "leave it benched").
- codex/arm-witness-post-index @ f265d2950 — the addon's B9 composition
  (the reviewer's range-diff reference; carries the overlap invariant +
  alias controls the strengthening lacks — the better B9 artifact if
  the topic is ever revived).
- codex/arm-s4-nibble-witness @ 7501904e8 — the register-gate arc's
  preserved artifact (Main's #58 disposition).
- feat/neon-s4-nibble-lifetimes @ 972c0081e — the register-gate arc's
  sibling artifact (same preservation class).
- codex/arm-panel-tournament-policy @ df15aca96 — idea-22 is OPEN (the
  FA2 policy-tail finding); the branch carries the open topic's work.
- feat/mxfp4-arm64 @ 6c6993dae — the MXFP4 work (OPEN, post-merge).

NOT ARM'S (left untouched): feat/opencode-hybrid-decode (the base
owner's harness), feat/qwen38-w4a4, feat/qwen-flash-next, feat/clover-
moe, feat/continuous-batching-c1-ownership, claude/deepseek-*,
feat/qfn-ngram-continuation (the base branch), master.

### ARM BRANCH SURFACE — FINAL STATE (~10:45Z, operator rule: "we don't need artifacts — part of the arm branch if important, otherwise delete")

SIX MORE DELETED per the artifact rule (closed/descoped topics whose
content will never join the arm branch; the topics' records live in the
TRACKER, not in dead branches):
- codex/arm-b9-strengthening (B9 benched/closed),
- codex/arm-witness-post-index (the B9 composition — the range-diff
  reference; the topic is closed),
- codex/arm-s4-nibble-witness (the register-gate arc's artifact —
  descoped),
- feat/neon-s4-nibble-lifetimes (the register-gate sibling),
- codex/arm-panel-tournament-policy (idea-22's experimental branch —
  held, not started; the tracker preserves the p95-regret finding),
- codex/arm-b1-fused-fork-pooling (addon3's WIP — superseded by Main's
  corrected-target implementation d5e04fdce on the PR candidate).

**THE REMAINING ARM SURFACE (6 branches):**
- feat/arm64-native-kernels @ ae51ce3f0 — THE PR (MERGEABLE/CLEAN).
- codex/arm-perf-review @ 8e7b95dd6 — the tracker (permanent).
- codex/arm-idea16-moe-decode @ 5c2cef47a — the ACTIVE W2 payload
  branch (the B1 seal re-run + tournament + census execute from it;
  deletable after the idea-08 inputs complete — the benches land via
  Main or go per this rule).
- codex/arm-idea23-hc-planning @ 640d625cb — pending-decision (the
  model-level run decides land-or-delete post-merge).
- codex/bundle1-perf-harness @ 5d606a108 — the parked Bundle-1 hop's
  subject (the hop decision determines land-or-delete).
- feat/mxfp4-arm64 @ 6c6993dae + feat/mxfp4-weights @ e29a3d7f — the
  OPEN MXFP4 lane (post-#43-merge work, operator-kept open).

NOT ARM'S (untouched): feat/qfn-ngram-continuation (the base branch),
feat/opencode-hybrid-decode (the base owner's harness), feat/qwen38-
w4a4, feat/qwen-flash-next, feat/clover-moe, feat/continuous-batching-
c1-ownership, claude/deepseek-*, master — other programs'/owners'.

### HOST LEASE VALIDATION (~11:11Z): direct SSH to .80 — single-occupancy confirmed

Operator asked why sparks AND main both appear on .80. Direct SSH
(factory@192.168.178.80, zgx-5457) at 11:11:26Z: docker ps = exactly
ONE container (main-b1ab, Up 2 min — Main's Release A/B, in the build
phase: cc1plus storm, load 20.5 from a 9.7 15-min avg); NO sparks
container; the M=1 window is queued behind main-b1ab, not running.
Serialized lease chain confirmed first-hand. Reporting rule going
forward: every host report states LEASE HOLDER + QUEUE explicitly
(association ≠ execution).

### SPARKS DISCLOSURE RECONCILIATION (~11:36Z): the 10:25Z M=1 attempt is NOT evidenced on either host

Sparks disclosed (11:32Z) an M=1 build attempt "starting ~10:25Z" on
.80 that created build-i04 + /work/i04-run2.log. Direct verification
(11:34Z, both hosts): (1) .80 docker ps -a — NO lane container ever
(only main-*); (2) /work/i04-run2.log — absent on BOTH hosts; (3)
build-i04 on .80 — build.ninja/CMakeCache stamped 08:00-08:04 +0100 =
the ORIGINAL morning M=1 attempt (the failed-empty one), NOT 10:25Z;
(4) .79 — no build-i04, no logs. **Conclusion: no overlap ever
occurred; .80 single-occupant throughout; the 10:34Z grant to Main was
valid (idle).** Sparks asked to reconcile what actually ran at 10:25Z
(shell history) and correct the disclosure's artifact claims. NO fault
assigned — honest-reconciliation culture; the queue discipline stands
(no launch until Main's explicit handback + arm-perf GO + docker-ps
re-verification).

### STATE AUDIT # (11:54Z, 90m cadence)

- **.80**: main-b1ab Up 44m (lease holder = Main, running) — MATCHES
  last verified; main-b1 Exited 0 45m ago; no lane container.
- **.79**: NO container (idle); evidence: idea25/w2-b1/ b1-{1T,fast-
  10T,mixed-20T,slow-10T}.log @ 11:01–11:09Z — fresh. Query sent to
  sparks re: W2 seal re-run/tournament/census state (complete/running/
  queued). Possible drift candidate pending reply.
- **Station**: SSH reachable (host 5f6041930e9d) but the `gpu` user's
  shell has NO docker CLI — cb-c1-canonical unverifiable via this
  path; CB-lead-owned (01a1ad51) — not re-probed deeper.
- **A2/B1ATTR2 exports**: LANDED — /home/factory/llaminar-i8mm/
  b1attr2-runs/ (b1-27b-mtp-full.json, b1-27b-mtp.out, b1-27b-nomtp*,
  b1-35b-mtp-full.json @ 07:42–07:45Z local).
- No session claiming progress without artifact paths beyond the above.

### AUDIT RECONCILIATION (12:05Z): path confusion resolved — no overlap confirmed

Sparks' seq-155 correction + my artifact checks: (1) i04-run2.log
DOES exist — at `/home/factory/llaminar-i8mm/build/i04-run2.log` (NOT
`/work/` — no /work on .80; BOTH sides had the path wrong, incl. my
first check using the stated path); stamp 11:07:32Z, **empty**;
i04-cfg.log 11:07:32Z (a cmake-cfg ran, produced nothing); i04-build
logs 08:04 +0100 (morning). So a script touched .80 at 11:07:32Z with
NO container + empty log — supports sparks' corrected account (heredoc
broke), but the timestamp = 11:07:32Z ≠ their ~10:25Z record — one-line
clock reconciliation asked. **Lease question ANSWERED: no container,
no overlap; main-b1 (10:34–11:08Z) + main-b1ab (11:08:45Z+) = sole
containers throughout.** (2) .79 = the W2 host: idea25/w2-b1/*.log @
11:01–11:09 local (+0100) = the 4-placement B1 re-run at 5c2cef47a,
all PASSED = K14-I16B1-5C2CEF47-20261112 seal evidence — the audit's
"re-run?" = ANSWERED by artifacts. Tournament 3 ISA binaries built
(build-cd-*); census next; then tournament. (3) Discipline note:
silent-failed launch attempts report to coordinator IMMEDIATELY (the
90m audit = verification path, NOT a reporting path).

### W2 TOURNAMENT RED → ROOT-CAUSED + FIXED (12:25Z): IQ3_S compact sign-plane double transposition

Sparks reported the W2 tournament RED (9/9 uniform byte-exact failure at
Perf__CPUCompactIQ3SDecode.cpp:491 — "compact decode differs from the
prepared plane at block=0 group=0 quad=0", all ISA cells × M=1/4/15, 6
ms, no timing lines). **Diagnosis (coordinator, contract-backed)**: the
diagnostic bench's IQ3_S sign plane has a double transposition vs the
production contract (CPUNativeQuantisedDecode.h case 11: sign bit =
payload[9+g/2] bit ((g%2)*4+lane)):
1. **Fixture**: source nibble at 4*(column%2) — must be 4*(group%2)
   (even columns read odd-group signs → fails at the first vector).
2. **Compact decoder**: sign_word from plane[quad/2]+plane[quad/2+1]<<8
   + (quad%2)*8 shift — the plane is one nibble per column (byte
   column/2, nibble column%2) → quad's 4 columns = bytes quad*2/quad*2+1
   with NO shift (odd quads read wrong bytes; shifted reads zero-fill
   upper lanes). Index/ninth-bit/scale/min paths were already correct.
**FIX LANDED**: origin/codex/arm-idea16-moe-decode @ **adf8efc32** (on
5c2cef47a; 2-line arithmetic corrections + contract comments).
Pre-validated: standalone fixture+decoder vs the production sign
contract passes 8×16×4. Reviewer focused pass dispatched (01a1a8d9).
Re-run gate = .79 tournament (rebuild build-cd-* from adf8efc32; 3 ISA
× M=1/4/15) after the idea-08 BOM census (independent, running next).
First-run-RED = never-green diagnostic bench, NOT a PR defect (file
exists only on the lane; non-installed diagnostic namespace).

### IDEA-08 TOURNAMENT: SOURCE HOLD — my fix incomplete; numbers void (12:30Z)

Reviewer one-pass verdict + Main's routing: **SOURCE HOLD at adf8efc32**.
The two sign corrections (adf8efc32) = confirmed RIGHT, but the bench has
FOUR more source defects (reviewer line numbers, verified by me against
the production authorities):
1. Oracle1 reference (line 128): ExpandedSource::weights16 uses
   group*256+quad*4; production column offset = group*256+column*4
   (CPUNativeQuantisedWeightPacker.h:595-609: destination+group*256+
   zmm*64+lane*kValuesPerGroup = group*256+column*4) ⇒ must be quad*16;
   at quad1 it compares expanded columns 1..4 vs compact 4..7.
2. Lines 270-273/328-331: BOTH walkers advance half-scale and raw-
   coefficient pointers by quad*4; admittedScales/admittedMins are
   per-column uint16 (CPUNativeNEONPreparedProjection.h:596-614: +
   column) ⇒ a quad's four columns = quad*8 bytes. Both arms share the
   mistake ⇒ Oracle2 is a shared-walker/metamorphic check, cannot prove
   production equality. (Per-quad single-scale aggregation is also not
   the production per-column program.)
3. Lines 604-647: SMMLA pairs only when M even ⇒ M=15 runs 15 SDOT
   rows, no SMMLA+tail; M=1 single-row OK.
4. Timed outputs overwritten, never consumed/validated (682-698 recheck
   decoded WEIGHTS only) — no retention proof of the timed FP32 work.
**CONSEQUENCE**: any tournament/census numbers from 5c2cef47a or
adf8efc32 = NOT idea-08 decision inputs. Repair = post-merge
(implementation-fenced, outside PR #43); re-run only after a fixed tip
gets a reviewer ACK. Sparks instructed: census ONLY (independent pure
model-load input, keep + pin script/artifact paths); tournament re-run
CANCELLED. adf8efc32 kept as stage-1 (reviewer-confirmed sign fix);
remaining findings queued for post-merge.

### OPERATOR COORDINATION GRANT (~13:55Z) + PERF LANES SPAWNED

Operator: "You are allowed to coordinate this... spawn 1-3 additional sessions
via herdr using pi and glm 5.3 with thinking max. or reuse existing sessions."
Posture change: performance PREP work is authorized during the merge window
(nothing touches PR #43's tip). Two lanes spawned (both verified
GLM-5.3 think:max):
- **arm-i08repair** (subagent-chat-01a1d014-0adf-7100, /tmp/llaminar-i08-wt @
  adf8efc32 detached): repair the 4 reviewer findings in the idea-08
  tournament bench; branch codex/arm-idea16-tournament-repair; reviewer ACK
  gates any re-run.
- **arm-p25design** (subagent-chat-01a1d014-22cb-7252, /tmp/llaminar-p25-wt @
  d5e04fdce detached, read-only): idea-25 design doc; branch codex/arm-p25-
  design; arm-perf integrates.
Coordinator keeps: MTP rider run-sheet (queued behind the M=1 window),
the census follow-up, the roadmap (NEW FILE:
post-merge-performance-roadmap.md = the operator-commissioned full review,
tiered + sequenced).

### STATE AUDIT + RESULTS WAVE (~15:30Z)

**IDEA-04 M=1 WINDOW SEALED — idea-04 CLOSES (measured-and-negative).**
K14-I04M1-D5E04FDCE-20261112: 6/6 cells exit=0 (.80 results/w-m1act-
20261112T141025Z; hard-fail guard verified). Q8_K 10-thread regression
GONE: t10 median 1897 µs vs t8 2211 (−14%); monotone 1→20 on every
encoding (Q8_K 15561→1631, Q8_0 same, Q6_K 25620→2076, Q4_K 12915→1083);
knees Q6_K −18% (8→10), Q4_K −7% (10→12); no cliff. The C4/R2 landing
resolved the original signal; NO implementation needed. Run-1 (13:11Z,
ae51 anchor, 4 shapes) = discarded by sparks after the re-anchor
instruction; run-2 (14:10:25Z, d5e04fdce) = the seal. ANCHOR NOTE:
build-i04 was pruned BEFORE coordinator verification — anchor rests on
sparks' run record + trees/tip@d5e04fdce (coordinator-verified) + CSVs;
hygiene reminder sent (keep the last sealed build dir until the seal is
coordinator-confirmed).

**MTP MODE-ARTIFACT QUESTION SETTLED — the −27% is REAL.** The greedy
rider seal K14-I16RIDER-880429059 (temp0+seed42, RESOLVED sampler) shows
acceptance 38.89% at current_depth 10 with stochastic_accepts=0 — FULLY
GREEDY still accepts only 38.89%, 18.39 tok/s vs no-MTP 25.2 = −27%.
NOT a sampling artifact; the draft path itself regresses. The tracker's
99%@depth-7 = a different protocol epoch. T2 CONTROL AUTHORIZED (fresh
no-MTP arm, same anchor, same-day pair — sparks executing). POST-MERGE
IMPLICATIONS (roadmap updated): (a) the policy candidate strengthens —
model-class/topology-aware MTP default for MoE+CPU could recover ~27%;
(b) NEW investigation: why did draft acceptance collapse (38.89% now vs
99% recorded earlier — protocol/version diff must be identified);
(c) the verifier LM-head routing B-candidate remains the kernel-side fix.

**CENSUS = INCOMPLETE (corrected).** plan.json = orchestration CONFIG
doc; plan.log = aggregate BOM (weights 26.7GB/total 34.9GB). NO per-
tensor prepared-encoding enumeration exists — that input is still owed
(observed per-tensor encoding + bytes through the production loader/
packer; sizes the idea-08 prize). Sparks asked to propose the mechanism.

**Lane ops**: both new lanes working (i08repair verified all 4 findings
correctly, drafting the walker fix; p25design reading the codebase).
Thinking-level downgrade detected (max→low→high at 14:13Z, harness-side,
both lanes); restoration requested per the operator's explicit max
requirement. Station download ~105GB/111.3GB. CB NUMA exchange closed
read-only (their investigation continues; our no-change statement +
confined-path evidence delivered).

### T2 CONTROL SEALED (~15:58Z) — MTP finding fully sealed; census enumerator GO

**K14-I16RIDER-T2-880429059-20261112**: no-MTP greedy 25.31 tok/s (temp0/
seed42 resolved, digest 0a4c0002a4a6de72) vs MTP-dyn 18.39 (same anchor,
same day) = **−27.4% at FULLY GREEDY, token digest IDENTICAL between
arms** — the MTP path preserves the exact token stream (verifier byte-
equivalence contract HOLDS); the regression is purely draft/verify
ECONOMICS (acceptance 38.89% @ depth 10, stochastic_accepts=0). The
99%@depth-7 tracker record = separate protocol epoch. POST-MERGE: the
model-class/topology-aware MTP default policy = top quick-win (~27%
recovery, zero output change); acceptance-collapse root-cause
investigation opens; verifier LM-head routing = the kernel-side fix.
M=1 run-1 discard confirmed deliberate (re-anchor kill; no exited
record — closed, no fault). Census correctly OPEN; enumerator GO'd:
read-only production-loader/packer diagnostic on the W2 lane branch
(per-tensor {encoding, source/prepared bytes} CSV, observed not derived,
non-installed namespace, .79) — completes the idea-08 input.
Lane thinking restored to max on both new lanes (root cause:
coordinator-shell PI_REASONING_LEVEL=high leaked into the herdr spawn
and overrode :max at first message injection; fixed via herdr
agent send-keys shift+tab; spawn-env hygiene lesson memorized).

### STATE AUDIT (~16:30Z): STATION DOWNLOAD = SIZE-COMPLETE (status was stale)

Direct du -sb: /home/gpu/models = 111,731,360,250 B ≈ 111.7 GB — all 4
Qwen3.8-Flash-Next-UD-Q4_K_XL shards present (49.7 + 32.8 + 28.8 + 0.011
GB = the expected ~111.3 GB total), plus qwen3-0.6b-q4km. NO active
download process. The repeated "105G / ~94%" audit readings were a
GiB/GB unit artifact (105 GiB = 111.7 GB — the same data); the download
has been effectively complete for some time (shard mtimes read Nov 6
22:0x — either preserved remote mtimes or finished then with the
status never updated). The watcher never reported completion.
POST-MERGE implication: idea-23 model-level validation can proceed;
integrity verification (checksum or first load) belongs to that step —
size-completeness alone is not load-certification. Other audit items:
.80 idle ✓ (post-T2); .79 enumerator GO'd ~30 min, no on-host activity
yet (code phase presumed — watch item); unknown .79 artifact build-edcc
(12:49Z, pre-GO — clarify with sparks' enumerator report); both Mac
lanes think:max ✓ (p25 on codex/arm-p25-design 69% ctx; i08repair 101k
ctx).

### IDEA-25 DESIGN LANDED (~16:45Z): 25-...-design.md @ a40bbb48b (cherry-picked)

arm-p25design delivered the complete design (702 lines, citations pinned
to d5e04fdce; lane branch deleted after integration per the branch rule).
**Phase order adopted** (see the design doc + roadmap): Phase 0 = non-
attached sub-boundary counters (6 host-wall timers + 2 counters; zero
behavior change; gate ≥95% wall attributed) → Phase 1 = zero-code pilot
via the EXISTING --moe-overlay-prefill-segment-rows flag (2160-token
prompt: 5 chunks → 2) → Phase 2a = maintenance-deferral (the per-chunk
histogram sync = hidden device wait, 288 D2H × event-wait per boundary;
first wait drains the whole 340ms replay — the DOMINANT suspect, not
launch cost at 39-78µs) → Phase 2b+c = family lease + back-to-back
submission (no CUDA conditional nodes needed) → Phase 3 = defaults only
after every x86 A/B is same-or-better. **Landmine flagged**: if
shouldRebalance() fires mid-prefill, discardAllCachedGraphs → next chunk
RECAPTURES (seconds-scale) — Phase 0 counters prove/disprove on .79.
Post-merge implementation = station lane (Spark-first), Phase 0/1 host
runs via arm-sparks.

### CENSUS v1 ANALYSIS (coordinator, ~18:45Z): IDEA-08 PREMISE REFUTED FOR THE 35B; predicted-bytes bug sent back

I analyzed K14-I08A-CENSUS-5C2CEF47's CSV directly (753 rows, right model
verified: Qwen3.6-35B-A3B-MTP-UD-IQ3_S): (1) **BUG**: the *_exps tensors'
predicted bytes are PER-EXPERT, not per-tensor (blk.34.ffn_down_exps pred
917,504 vs src 220,200,960 ≈ 1/n_experts; blk.39 gate/up_exps ~1MB vs
115MB) — the SUMMARY's 2.26GB-vs-15.3GB "footprint≠pack" reconciliation
is invalid; sparks regenerates (× n_experts) + re-seals. (2) **MAJOR —
the encoding map refutes idea-08's premise for this model**: the bulk is
ALREADY compact — ffn_gate/up_exps = CompactMultiScale on 40/41 layers
(~7.1GB src), ffn_down_exps = NibbleLUTSourceScale ×37 + NibbleLUT ×1 +
Q6 ×3 (~6.1GB); expanded encodings = only blk.39 (the UD outlier layer:
IQ3SExpandedSourceScale gate/up 230MB + 8 tiny ExpandedInt8) ≈ 0.25GB of
15.3GB. The "IQ3_S expansion = 2.7× decode bytes" story does NOT apply
to the 35B's decode traffic. Idea-08's residual value on this model =
dead compensation bytes (NibbleLUT/Compact families) + general layout
hygiene — a MUCH smaller prize. (3) **Decode-gap attribution shifts**:
output.weight + token_embd = Q6 ~445MB prepared EACH; the LM head reads
that every token at M=1 → now the dominant single-tensor decode cost
(ties to idea-16's LM-head 45GB/s finding + the B4/LM-head lane). The
census did its job: sized the prize before building the wrong thing.
Roadmap idea-08 entry + the decode-gap section need revision on the
re-seal.

### TOURNAMENT REPAIR @ e0f6506c9 → REVIEWER (~18:44Z)

arm-i08repair complete (branch codex/arm-idea16-tournament-repair, one
file +102/−42): all four findings verified against production
authorities + fixed (Oracle-1 quad*16 in BYTES — unit nuance vs the
finding's phrasing; walkers = per-column uint16 planes via
reinterpret_cast + quad*4; timed path = floor(M/2) SMMLA pairs + DotProd
tail matching the production tile split; retained per-row outputs +
byte-for-byte retention oracle). Sign fix untouched; no CMake changes;
standalone proofs uncommitted per the sign-fix pattern; syntax-only
clean in all 3 cell envelopes. Focused-pass request dispatched to
arm-reviewer (01a1a8d9) with the 4 verification points; the .79
tournament re-run fires only on ACK. Lane stands by for fix-forwards.

### REVIEWER VERDICT on e0f6506c9 (~18:56Z): SOURCE HOLD — findings 1-3 closed, finding-4 fix-forward

Confirmed by the reviewer: byte units (quad*16 BYTES), walker word
identity (IQ3_STensor.cpp:483-504 → WeightPacker:527-536/:639-655, no
float round-trip), and the row/leaf tile split (M1 DotProd; M4 2 SMMLA
pairs; M15 7 pairs + DotProd row14; scalar loop #else-excluded) — all
SOURCE-CLOSED. **Open**: the retention oracle validates only the FINAL
sample; earlier timed iterations are overwritten (concrete negative:
omit one_pass at :708 in every timed iteration — warmup outputs still
pass). Required: per-invocation live-output observability at the timing
boundary + untimed dirty-output/check-before-reuse controls; comparison
outside the timed interval. Fix-forward dispatched to arm-i08repair
(acceptance test = the reviewer's negative must FAIL the oracle).
Also noted by the reviewer: Oracle2 remains shared-walker/metamorphic
(not independent full-engine proof) — accepted framing for a diagnostic
bench; uncommitted proofs = lane reports, not reviewer-authenticated
evidence.

### FINDING-4 FIX-FORWARD @ 74d228047 (~19:22Z) → reviewer ACK request

Per-invocation observation boundary: volatile quiet-NaN sentinel dirty
before the clock (DSE-proof edge) → boundary materialization after the
clock edge (sentinel scan + per-sample FNV-1a digest into distinct
evidence slots) → untimed chain (per-sample determinism + arm equality).
Acceptance negative proven standalone (omitted one_pass fails on the
FIRST sample at M=1 and M=15; partial/wrong/stale/arm-disagree each
fail their control). Open design question for the reviewer:
materialization at boundary-after-clock vs inside the timed interval
(one-line flip if required). ACK request dispatched; .79 re-run gated
on it.

### REVIEWER VERDICT on 74d228047 (~19:31Z): SOURCE HOLD — two F4 residues → fix-forward #2

Validated: sentinel fixture-bound safe (bounded arithmetic worked through
by the reviewer); placement ruling = scans/checksums/comparisons stay
OUTSIDE the timer (do not move materializeEvidence inside). **Issue 1**:
the volatile dirty preserves the poison writes but does not order/
materialize the NONVOLATILE walker stores before the END timestamp (first
consumer is after end) — a compiler-only start boundary + live-output
memory escape before the end-clock edge is required. **Issue 2**:
concrete digest break — FNV-1a with odd P: flipping 0x80000000 into two
ADJACENT float words cancels in the digest (both stay finite) → a
nonfinal-sample pair corruption passes sentinel + chain + final memcmp;
the digest cannot be the acceptance authority — untimed live-span byte
comparison per completed two-arm sample before buffer reuse required
(digest = diagnostic only). Fix-forward #2 dispatched to arm-i08repair.

### CENSUS CORRECTION OF RECORD (~19:40Z): my "premise refuted" conclusion was WRONG — corrected data CONFIRMS idea-08 at 1.73× with a reassigned target

The v1 "premise refuted / bulk already compact" conclusion (my analysis,
above) was derived from the encoding NAMES while the byte column was
per-expert-broken. The CORRECTED census (SUMMARY-v2, per-expert fix):
predicted 26.62GB vs source 15.34GB = **1.73× expansion**. Carrier =
**CompactMultiScale gate/up experts 6.88→16.11GB (2.34×, +9.2GB)** +
NibbleLUTSourceScale down 6.12→... 5.28→6.83 (+1.55GB); Q6 = genuinely
native (1.06×); IQ3SExpanded (blk.39) = negligible (+0.4GB). Idea-08
CONFIRMED; compaction target REASSIGNED to CMS + NibbleLUTSourceScale;
potential ~42% prepared-quantised-byte reduction (26.6→~15.3GB if
source-sized) = the decode-bandwidth prize on the 35B. Framing note:
26.62 predicted vs 26.69 plan-BOM = same-estimator agreement (the planner
BOM is estimator-derived), NOT independent validation; normative anchor =
the preflight estimator-byte-equality class; observed final = the
post-merge seam. Sparks amends SUMMARY-v2's finding paragraph (CSV
stands). LM-head finding stands (Q6 445MB×2 M=1 read = dominant single-
tensor decode cost). The earlier v1 entry is superseded by this one.

### Reviewer compaction (~19:35Z): triggered from coordinator (operator-delegated)

arm-reviewer had no self-service compaction; typed /compact via pane
send-keys (character-split) — "Compacting context..." confirmed; todos
#210/#211 (repair review + compact-decode blockers) intact. Reviewer
clear to resume passes post-compaction (next: fix-forward #2 ACK).

### FIX-FORWARD #2 @ c1a002b50 (~19:58Z) → reviewer ACK request

Issue 1: seq_cst atomic_signal_fence at both interval edges (zero
instruction cost; signal-handler visibility semantics force walker-store
materialization before the end-clock read; standardized wrapper of the
empty memory-clobber asm). Issue 2: lane independently reproduced the
reviewer's FNV cancellation attack (digest 0x3ea0d00f unchanged) and
REMOVED the digest entirely — acceptance = untimed per-sample byte
comparisons (arm-vs-arm per row + byte-exact sample-0 reference, before
buffer reuse); sentinel freshness unchanged. Deliberate removal flagged
for ruling: the post-loop final-contents memcmp (subsumed + was an
attack surface). All negatives proven standalone. ACK request queued to
the compacting reviewer; .79 re-run gated on it.

### CENSUS CLOSED CORRECTLY (~20:03Z)

SUMMARY-v2 amended + re-pinned: idea-08 CONFIRMED at 1.73x (26.62 vs
15.34 GB); carrier = CompactMultiScale gate/up (2.34x, +9.2 GB) +
NibbleLUTSourceScale down (+1.55 GB); Q6 native; blk.39/IQ3S negligible;
~42% prepared-byte reduction potential = the 35B decode-bandwidth prize;
LM-head 445MBx2 M=1 read-cost stands; consistency line = same-estimator
agreement framing (normative anchor = estimator-byte-equality preflight
class; observed final = post-merge seam). The idea-08 input is COMPLETE
(predicted/estimator, correctly labeled) pending only the post-merge
observed seam.

### REVIEWER VERDICT on c1a002b50 (~20:09Z): F4b CLOSED + memcmp-removal ACCEPTED; F4a open → fix-forward #3

Closed: the acceptance redesign (per-row arm comparisons before reuse +
sample-0 references; digest gone; single-arm adjacent-sign-flip and
common-mode drift both covered) and the final-memcmp removal (last pair
takes the same checks as every other pair). Open — F4a: retained_* are
AUTOMATIC storage; atomic_signal_fence does not make non-escaping
automatic storage signal-observable (no pointer/extent escape to an
opaque consumer), so the walker stores still lack a mandatory pre-end-
clock materialization edge (first explicit reader :794-796 is AFTER end
:786); a naked memory-clobber asm alone is also insufficient. Required:
an instruction-free compiler primitive with an explicit live-output
memory operand / pointer-extent escape before the end clock + the input/
start boundary; fences may remain alongside. Scope note from the
reviewer: Oracle2 = shared-walker/metamorphic with production leaves
(accepted framing); sample0 = consistency anchor, not a new oracle.
Fix-forward #3 dispatched (object-bound empty-asm escape suggested).

### FIX-FORWARD #3 @ 3b109793c (~20:16Z) → reviewer ACK request

F4a closed in the required shape: object-bound escape asm
`("" : : "r"(span), "r"(bytes) : "memory")` at both interval edges
(pointer+extent escape to an opaque consumer; memory clobber forces
prior-store materialization before the end-clock read; start-boundary
escape also proves poison stores precede the timed region); fences kept
alongside; one span reference per invocation binds dirty/escape/walker/
scan to the SAME object; dual-compiler verified (AppleClang 21 +
GCC-14.4, all 3 envelopes); -O2 replication shows the full store
sequence strictly between the clock reads. ACK request dispatched.

### REVIEWER BOUNDED SOURCE ACK @ 3b109793c (~20:25Z) — repair cycle COMPLETE at source level; .79 tournament GO

F4a source-closed (escapeSpan :649-650 + invocation binding :790-793,
extents 256/1024/3840B for M=1/4/15, start-escape before the walker,
end-escape before the end clock); #210/#211 completed for this scope.
Reviewer precision notes (record, not blockers): start-escape after the
begin timestamp (poison placement = preceding volatile stores); escape =
compiler-level retention contract, not hardware store-drain; Oracle2 =
shared-walker/metamorphic; owner-reported portability/negatives. ACK
admits NO economy/integration/rebase-transfer/scheduling grant.
Main's position: bench stays on the lane branch OUTSIDE PR #43,
diagnostic-only for post-merge idea-08; Main not applying it.
Base ruling: run at ACK-pinned 3b109793c, NO rebase onto d5e04fdce
(reviewer ACK non-transferable across rebase; B1 touches only
FloatingPointGemmKernel.h which the IQ3_S bench never exercises →
identical measured production path; re-pin to merged develop tip happens
at idea-08 implementation time). Sparks GO dispatched (K14-I08A-
TOURNEY-3B109793C, 3 ISA binaries, M=1/4/15, GCC-14).

### STATE AUDIT 2026-11-12T20:53Z (exceptions-only — 2 exceptions, flagged)

Verified from direct probes: .79 worktree tip-repair @ correct ACK-pinned
3b109793cc (pre-claimed prep verified: mirror + checkout + external
seeded ✓); .80 idle (docker: only Main's exited campaign containers
9-17h old — consistent). **EXCEPTION 1**: tournament GO dispatched
~20:32Z but .79 shows zero build processes / no build-cd dirs at
20:56Z — launch not yet materialized; sparks (session busy on bash)
pinged for immediate launch confirmation per the silent-failure rule.
No launch CLAIM made, so not yet claim-vs-reality drift — watching.
**EXCEPTION 2**: the station trycloudflare hostname
(continent-wit-intervention-elder) no longer resolves — tunnel rotated
or down; station unreachable for both programs. Station session is
operator-managed; CB lead informed via this record. No ARM action was
pending on the station (idea-23 is post-merge), so no program impact;
tunnel identity needs refreshing from the station side when next
needed. All other sessions match last verified state (reviewer
compacted+idle 60% ctx; i08repair idle 80% post-fix; Main idle
on-call; CB lead thinking — their own program).

### AUDIT CORRECTION: station "tunnel down" exception RETRACTED (my error)

The trycloudflare hostname I probed is the "josh" intercom remote's
rotating dynamic tunnel — NOT the station (operator correction). The
station is the configured intercom remote "dgx-station" and is ALIVE:
two live peers verified (c290 ARM station session, idle; c527 CB C1
worker, thinking). Standing method fixed: fleet hosts are reached via
intercom list-remote (wsl, hetzner-docker, josh, dgx-station), never
hardcoded SSH hostnames. Audit exception 2 retracted in full; exception
1 (tournament launch-lag ping to sparks) stands unchanged.

### OPERATOR AUTHORIZATION (~2026-11-12T20:58Z, verbatim): "I would like you to start Idea-08 in a new session"

Idea-08 (compact ARM prepared-weight layout) implementation lane
AUTHORIZED to start NOW in a new session — within the merge-window
prep envelope (lane branch off d5e04fdce; PR tip untouched; x86
unchanged by construction via per-ISA layout identity; large
refactorings permitted per the standing ~13:12Z grant: "spawn 1-3
additional sessions via herdr using pi and glm 5.3 with thinking max").
This is the third session of that grant (i08repair, p25design closed).
Inputs sealed: corrected census CONFIRMS 1.73x expansion, carrier =
CompactMultiScale gate/up (2.34x) + NibbleLUTSourceScale down; repaired
tournament bench (ACK @ 3b109793c) = the validation methodology
template; .79 re-run in flight separately.

### SILENT LAUNCH FAILURE (Main-observed 21:11Z): .79 tournament relaunch #1 failed; relaunch #2 dispatched

sparks' 21:00Z relaunch: tourney2-run.log = `bash: /w/tourney2.sh: No
such file or directory` — scp never produced the file on .79 (path
mismatch); no container, no build-tr-* dirs; launcher sleeping 1500s.
Tree /w/tip-repair correctly @ 3b109793c (only prep verified true).
The 13:08 w2-tourney output = pre-repair tip, 9/9 RC=1 Oracle-1 quad-0
(expected at that tip) — NO valid tournament result exists yet.
Relaunch procedure dispatched (interrupt sleeper; ssh-heredoc the
script at /home/factory/idea25/tourney2.sh; verify file + build dirs +
first output; report evidence). OPERATOR CONTEXT: the tournament result
is requested for the PR #43 description — the sealed table (ISA x M,
prepared vs compact time + ratio) is now blocking; Main receives it on
seal. Lane-defect classification: launch mechanics (silent scp
failure), first-run-RED of the launch, not of the bench itself.

### IDEA-08 LANE SPAWNED (21:16Z): arm-idea08 (third session of the operator grant)

Session 01a1d19e-e119-7141-8cb8-8c1814bde037, pane wF:tR, glm-5.3
think:max VERIFIED after brief injection (env-cleaned spawn held).
Worktree /tmp/llaminar-idea08-wt, branch codex/arm-idea08-compact-layout
@ d5e04fdce (read-only base; integrates post-merge via Main). Brief
delivered: design-doc-first (expansion-byte anatomy, compact layout,
full ABI chain, per-ISA identity for x86, byte-equivalence contract,
bench plan per the 3b109793c observability standard), then phased
implementation; sparks hosts exclusively via coordinator; artifacts-or-
silent. Separately: tournament relaunch #2 dispatched to sparks after
Main caught the silent scp failure (PR-description blocking).

### SEALED+VERIFIED: K14-I08A-TOURNEY-3B109793C-20261112 (9/9 GREEN, first valid idea-08 timing)

Coordinator verified from .79 artifacts (w2-tourney2/: 9 RC=0 cell logs +
progress.log, 40 samples/cell, byte-oracle OK in every log, medians
match the report). Table (expanded/compact µs, ratio): ASIMD M1
5.15/36.6 (7.1x), M4 20.5/145.0 (7.0x), M15 73.6/530.0 (7.2x); DotProd
M1 4.03/33.4 (8.28x), M4 16.0/127.5 (8.0x), M15 56.4/478.2 (8.5x);
I8MM M1 4.02/34.8 (8.62x), M4 9.3/67.9 (7.3x), M15 34.6/271.0 (7.9x).
9.5 vs 4.25 bpw (2.2x fewer bytes). **VERDICT: decode-from-compact is
7.0-8.6x SLOWER despite 2.2x fewer bytes — the prepared expansion is a
WINNING speed-for-bytes trade at these shapes on GB10; decode is
compute-bound (dequant), not bandwidth-bound. Idea-08's "compact =
decode-bandwidth prize" premise is REFUTED for the IQ3_S family.**
Surviving scope: dead-byte trimming inside expanded layouts (no access-
pattern change, capacity-only), capacity-driven compaction where
admission-bound (7-8.7x decode price stated), and the CMS/NibbleLUT
burden-of-proof. I8MM = strongest family (expanded I8MM ~2x ASIMD at
M4/M15). Main received the table for the PR #43 description (metadata
only, tip untouched). arm-idea08 lane informed + reframed BEFORE its
design doc. Census note stands: expansion bytes are real (1.73x) but
are a deliberate winning trade, not waste.

### STATE AUDIT 22:23Z (exceptions-only): NO exceptions — all verified

.79 idle (0 procs), seal evidence intact (w2-tourney2 10 files), build-tr
dirs still present (pruning pending, minor, sparks informed); .80 idle
(0 containers, 0 procs); station alive via dgx-station remote (c290
idle, c527 CB C1 active — consistent); PR #43 description UPDATED by
Main at 22:01Z with the tournament table (verified via API: body
contains the tourney data; state OPEN/MERGEABLE). arm-idea08 delivered
its reframed design doc (below) — thinking→delivered, no drift. CB lead
at 89% ctx (their program, noted).

### IDEA-08 DESIGN DOC DELIVERED + VERIFIED (22:25Z): dcebc7ed8 — capacity-only reframe; operator gate after Phase 0

Branch codex/arm-idea08-compact-layout @ dcebc7ed8 (verified: one
docs-only commit on d5e04fdce), design doc 08-arm-prepared-weight-
layout-design.md (694 lines). Key findings: (1) the sealed 7.0-8.6x
negative does NOT attach to a pure dead-byte trim (tournament priced a
decode-PROGRAM change; this design changes no kernel load set — removes
only never-loaded bytes: CMS payload padding 448B/unit, non-Q2_K emins
256B, ARM-dead compensation 128B x86-only; read-set proof table doc
§2.1); speed-neutrality accepted — NO decode-speed claims remain, bench
reframed to a NON-REGRESSION gate. (2) Value = capacity only: -6.0GB
prepared (26.62->~20.6GB, -22.6%) = ~26% more routed experts resident
per admitted CPU ExpertOverlay tier + 6GB KV headroom on UMA; NO
currently-measured topology is admission-blocked by the expansion
(GB10 hosts the 35B at 34.9GB BOM with headroom). (3) Lane self-
assessment: value dropped a tier; do-not-close-outright; OPERATOR GATE
after Phase 0 (typed layout spine, zero byte change, reusable either
way). Local baseline: 128/128 targets, ctest 295/295 green at d5e04fdce
(Mac NEON portability loop). Routing: lane PROCEEDS to Phase 0, holds
at the gate. Scale-dedup/secondary-packing V2 = OUT OF SCOPE (access-
pattern changes), not deferred.

### IDEA-08 PHASE 0 COMPLETE + VERIFIED (23:23Z): d3e936fbd — spine landed, HOLDING at operator gate

Lineage d5e04fdce -> dcebc7ed8 (docs) -> d3e936fbd (spine). Coordinator
verified on origin: 9 files +596/-43 (gemm/ ABI headers + new
Test__CPUNativeQuantisedPreparedLayout.cpp 260 lines + CMake +17).
kHostPreparedLayout = InterleavedCanonical on every ISA (zero byte
change by construction; compact arm fails closed). Tests: NEON
portability 297/297 = 295 baseline untouched + 2 new registrations
(Unit + ProductionTestPreflight). Register ratchet note: first spine
version pushed 21 NEON tile programs 1-2 spills over budget (layout
check in the hot compensation predicate) — caught by the tile-register
proof; fixed by resolving compensation presence once at admission into
a stored 0/128 metadata delta (hot accessors strictly <= baseline;
proof PASS 64/64). Phase 2 (archive v8/wire/movement) + CUDA/ROCm
remain canonical-only until threaded. GATE QUESTION (operator, DUE):
CONTINUE to Phase 1 (compact stride tables + packer placement on
AArch64 + admission + estimator overloads + ARM-local bit-identity) =
-6.0GB prepared capacity/residency; or CLOSE with the spine retained.

### OPERATOR QUESTION (23:5xZ): "save weight bytes without the latency increase?" — tiered-residency option sized

Answer delivered: (1) dead-byte trim (Phase 1, zero latency by
construction, -6.0GB); (2) **compact-at-rest / expanded-on-promotion**
via the existing hot-expert-cache + residency machinery (zero hot-path
latency change, near-full byte prize, one ms-scale expansion per
promotion off the critical path, PMA accounts both forms; combined
floor ~17-21GB for the 35B); (3) cheap-LUT middle representation only
where codebooks permit (NibbleLUT 1.29x family yes; CMS 256+-entry
pair grids structurally no — NEON lacks vector gather). Regime note
recorded: the tournament priced a cache-resident compute-bound
microbench; production decode is partly DRAM-bound (idea-16 45GB/s LM
head) — which is why tiering targets RESIDENT bytes, and why a faster
compact kernel is not the path (structural gather cost stays above the
2.2x breakeven). arm-idea08 dispatched a design addendum (hot-set math,
promotion cost, cold-miss policy, combined floor, prefill interaction,
delta vs Phase-1-only) while holding at the gate.

### OPERATOR WEIGHT-COMPACTION TASK (Main-relayed, 23:41Z): L1/L2/Q5_K plan — details settled, decision delegated + made

Operator task (verbatim intent): discuss the compaction idea with the
coordinator to settle details, then reviewer, then back to coordinator
to decide the plan, then start. Validation hosts: WSL x86 + sparks ARM.
FACTS from Main (source + census + GGUF): 35B gate/up experts = 78
IQ2_S + 2 IQ3_S + 2 Q2_K; CMS unit 1536B anatomy (4 planes, IQ2_S uses
planes 0..2, plane 3 padding; emins Q2_K-only; u16 coefficient; d
repeated per K block). L1 stride-only: 1536->1024 (IQ1_M->768), CMS
16.1->10.7GB (~-5.4GB). L2 IQ2_S-exact (loads-only changes: sign byte
plane, u8 coefficient, superblock d): 656B/unit = source size, CMS
->~6.9GB. 27B: Q5_K->ExpandedInt8 1.73x; native-Q5_K bitplane form
(Q6 precedent) ~1.3-1.4KB. Coupled surfaces: strides->estimator/PMA/
slot pool; ExpertTierWeightDeviceLayout + CUDA/ROCm tier repack
(cpu_data_stride); PackedWeightsSerialization; AVX2/512 shared template
(x86 gate); scalar reference + tests hard-coding 1536.
COORDINATOR DETAIL-SETTLING (delivered 23:5xZ): Q1 = speed-neutral/
capacity-win (bandwidth slack; Main corrected my active-set figure to
~1.0B weights/token ~=0.75GB prepared — accepted, recorded); Q2 = L1
first, L2 SEPARATE later gated phase (IN-PLAN per operator intent
"save LIKE IT DID", not out-of-scope; design §2.2/§9.3 to be revised at
plan consolidation); Q3 = +7 measurement additions (ratchet gate both
ISAs, direct byte-economy evidence PMA/RSS/DRAM counter, full sweep +
Q2_K has_emins mixed strides, mixed-stride cross-tier migration,
prefill cell, Q2_K/Q3_K canonical cell + IQ1_M Unit-level, x86
same-or-better framing stated); Q4 = base d836a276f disjoint (3 ROCm
perf commits, verified); only arm-idea08 in-flight on these files.
Q5 = no persistent prepared-byte stream known (transient wire/tier/
slot-pool only; rider "shutdown persisted" = KV payload); wire version
bump regardless + pre-flight source audit.
RECONCILIATION ACCEPTED by Main: no parallel ABI — Phase 1 (= L1)
builds on the spine d3e936fbd; design pre-review with the reviewer (6
questions: read-set proof incl. residual correction/verifier/
diagnostics/absolute offsets; GB10 CUDA<->ARM tier repack under
compact-A64 both GPU backends; mixed-ISA cluster movement/
reinterpretation; cross-ISA PMA pricing target layout;
PackedWeightsSerialization archive v7 persistence; L2 admissibility).
LANDING DECISION: operator verbatim "perf is responsible" (~23:59Z) —
coordinator CONFIRMED post-merge fast-follow (implement+validate now
on the spine branch, land after the develop merge; topology-run cost
stated; revisit only if the merge stalls indefinitely). Sparks
dispatched the Q1 bandwidth-observation artifact (.80, no-MTP + dyn-MTP
achieved GB/s vs ~273 ceiling, K14-Q1BW seal). Next: reviewer findings
-> fix plan -> lane starts as implementer.

### STATE AUDIT 23:53Z (exceptions-only): NO exceptions — all verified

.80: sparks' Q1 bandwidth observation LAUNCHED live (q1bw container
llaminar-arm-engine-sdk-py:e0d6d34e7, nohup + q1bw-run.log + 300s tail
verification; base64-decode-on-host + bash -n — the post-silent-failure
launch pattern, working). .79 idle (0 procs). Station: c290 idle, c527
idle (CB C1 between phases — lead-owned, consistent). Sessions match:
reviewer THINKING (Main's 6-question design pre-review — in flight),
arm-idea08 THINKING (tiering addendum — in flight), sparks tool:bash
(Q1 launch — matches host evidence), Main idle 41%, CB lead thinking
(compacted, 50% ctx — healthy), i08repair idle. A2 rider exports
confirmed landed (i16rider/ + i16rider-t2/ paths verified earlier this
session). No claim-vs-artifact drift anywhere.

### REVIEWER SPINE FINDING (00:04Z, Main-confirmed): latent fail-closed gap in view/seal admission — Phase-1 entry item

At spine d3e936fbd: WeightView.h:456-459 divides by the stride BEFORE
rejecting an unimplemented layout; preparedUnitSections
(PreparedFootprint.h:233-234) returns ZERO stride for
InterleavedCompactA64 -> a compact descriptor reaches division by zero
(also via MetadataSeal::validate :539-542) instead of precise fail-closed
rejection. Spine test UnimplementedLayoutFailsClosed (:192-217) covers
tables/footprints only, not view/seal admission. Latent (production
never selects compact on the spine). Disposition: FIRST Phase-1 entry
item — explicit zero-stride/unimplemented-layout rejection in view
admission + seal before any division, view/seal-level regression; the
fail-closed invariant generalizes to every future layout. Lane informed
(folds into its Phase-1 queue alongside the tiering addendum). Plan
holds until the six-question review completes.

### Q1 BANDWIDTH ARTIFACT SEALED+VERIFIED (00:11Z): K14-Q1BW-880429059-20261113 — NOT bandwidth-bound, >=13x slack

Files: .80 /home/factory/llaminar-i8mm/q1bw/ (perf-nomtp/mtp.txt +
plan + rider outs; counters re-verified directly: no-MTP 17.52M
mem_access + 6.04M stall_backend_mem / 63.73s; MTP 6.10M + 2.58M).
Method: host sudo perf stat (paranoid=4; read-only) around the existing
35B rider binary per cell, greedy t0/s42/n=256. no-MTP ~10-21GB/s
upper bound (32/64B line; whole-run incl. load+prefill so steady
decode <= this) vs ~273GB/s ceiling = >=13x slack; MTP 2.4-4.8GB/s;
stall/access ~34% = latency stalls not throughput cap. Decode
reproduced 25.4-26.1 / 18.5 tok/s. VERDICT: L1 byte savings = capacity,
not single-request latency — artifact-backed. Caveats in seal: PMU
mem_access is not calibrated DRAM traffic; analytic bound (~0.75GB/token
active set) remains normative; upper-bound direction only strengthens
the conclusion. Relayed to Main (closes its Q1 plan input). Both hosts
free.

### IDEA-08 TIERING ADDENDUM VERIFIED (00:14Z): 3b00c69d7 + coordinator units correction

Lane deliverable on origin (docs-only, +274 lines): §10 tiering sizing +
spine finding as Phase-1 entry item 0. Headlines: hot path reads
canonical expanded by construction; R(h) = 15.96 + 0.0939h GB;
break-even vs Phase-1-only h < ~49/layer (net-replace ~107), vs
canonical h < ~113; combined floor 16.3-16.7 GB; promotion 54-105us
per instance (tournament tile = expert tensor shape, direct
translation), repays 2-3 activations, fabric-throttled fill (~25-30min
for h=32/layer at defaults); cold-miss serve-compact-slow +0.3%/inst
(T2-only, needs program-B kernel sweep), block-until-expanded rejected;
cold-heavy prefill +8-32% marginal, expand-on-prefill not viable; T1 =
insurance variant (no bytes). DECISIVE UNMEASURED INPUT: hot-set
cardinality h on real routing traffic (no seal carries it) — candidate
evidence step: rider-protocol run with residency demand-histogram
export (DecodeExpertHistogram) to measure h on the 35B; Main's call at
consolidation.
COORDINATOR UNITS CORRECTION (recorded + sent to Main): the earlier
"25.31 tok/s = 39.5us/token" was wrong — it is 39.5 MS/token.
Conclusion unchanged and strengthened with Main's corrected 0.75GB/
token active set: expert-read bandwidth ceiling ~363 tok/s vs 25.31
achieved (>=14x slack); Q1 artifact (10-21GB/s whole-run, 4-8% of
peak, ~34% stall/access) consistent.

### PLAN CONSOLIDATION PROGRESS (00:19-00:24Z): IQ2_S premise verified, tiering descoped, mirror corrected

MAIN'S IQ2_S READ VERIFIED IN SOURCE (d5e04fdce): dotHalfRowPairs
Codebook 13 (CPUNativeQuantisedMultiScaleKernels.h:399-416/:470)
decodes the native iq2s_grid in-kernel via Ops::gridPair — program A
INCLUDES the grid decode; the 1536B unit = native payload + layout
overhead, NOT decoded weights; L2 (656B) = same program, load-shape
changes only. The addendum's §10.1 premise (canonical=expanded,
source-scale=program B) was wrong — contradicted its own §1.5; lane
informed with the correction. CONSEQUENCES (accepted into the plan):
Phase 1 + L2 reaches ~source bytes ON THE HOT PATH with no tiering
(15.96GB floor territory); T2 tiering only for expanded-baseline
formats (35B's 2 IQ3_S tensors ~0.4GB; 27B Q5_K); h measurement off
the critical path (cheap background evidence only). PLAN SKELETON
AGREED: Phase 1 spine-fix + compact-A64 strides -> Phase 2 versioning
(wire v8, layout-conditional CUDA/ROCm repack) -> Phase 3 L2 behind
register-ratchet proof + own paired bench -> Q5_K-native separately;
tiering-for-expanded-forms parked. MIRROR OPTION (pre-expand IQ2_S for
speed) PARKED with corrected bound: coordinator's initial ~45% decode-
share estimate was WRONG (single-core rates without 20-core scaling —
Main corrected): honest bound ~1.5-3ms/token (~4-8% of 39.5ms, likely
less: IQ2_S u64 grid 8 lanes/lookup cheaper than the measured IQ3_S
grid), cost +10GB against the byte goal. Evidence gate: measured
expert-FFN stage share at decode; dead if the share is small. NOT part
of this plan. Awaiting the reviewer's six answers to fix the plan.

### REVIEWER #212 PROGRESS (00:25Z): L1 read-set argument structurally sound; two ACTIVATION GAPS -> Phase-1 entry items 1-2

(1) ExpertTierWeightStream.cpp ~:755-762 copyCPUEncodedColumn: CMS
encoding-only logic copies 2x u16 planes + u32 emins at
data_stride+4*columns — with IQ2_S compact (data 768, stride 1024) the
emins access lands OUTSIDE the trimmed unit (partial/rectangular tier
paths; full-unit memcpy hides it). Fix: derive planes from
PreparedUnitSections (emins = has_emins/Q2_K only). (2)
WindowPacking.inc :178-183 emins written unconditionally; :208-210
compensation written unconditionally — safe under canonical Phase 0,
must be presence-guarded before compact activation. Regressions: (a)
mixed-stride partial/rectangular tier copy (IQ2_S + Q2_K side by side);
(b) packer section-presence checks. GENERAL PHASE-1 RULE: every writer
and copier derives section presence/offsets from PreparedUnitSections;
no encoding-only assumption survives. Entry list now 0 (zero-stride
fail-closed), 1 (tier-stream copy), 2 (packer presence guards). Lane
informed + holding. Final reviewer report pending (separates activation
gaps from the Phase-0 division defect).

### CONSOLIDATED PLAN CLEARED + IMPLEMENTATION STARTED (00:31Z): Stage A dispatched to the lane

Main's consolidated plan (reviewer #212 folded in; SOURCE HOLD on S1 +
design-not-ACKed-as-written -> items became the plan) CLEARED by
coordinator with two amendments: (1) reviewer gates at stage boundaries
(Gate A pass before Stage B activation; Gate B ACK before the
fast-follow PR); (2) doc corrections sequenced into the A0 checkpoint.
STRUCTURE: production selector stays canonical until the whole chain is
layout-aware; activation = Stage B. STAGE A (zero byte change): A0 S1
fail-closed (view admission + MetadataSeal::validate reject before any
division; regression calls real admission) + doc corrections; A1
section-presence authority from PreparedUnitSections everywhere
(producers WindowPacking 178-183/208-210 + direct-int8/deferred/
rotation; copiers ExpertTierWeightStream 755-762 + gather/scatter/
direct; accessors WeightView 367-371 + PackedWeights 446-451 fail
explicitly on absent emins; host reference conversion Stream 1344-
1356/1554-1565); A2 GPU conversion from CPU ENDPOINT layout (never
__aarch64__/compiler host; typed manifest contract; CUDA 267-274/376-
385/502-506/557-585; ROCm 552-558/657-666/775-779/828-870 incl. IQ3_S
absolute offset); A3 identity/transport (layout in transfer descriptor;
MPI mismatch rejected before allocation ExpertWeightTransfer 374-377/
509-544, DGO 64661-64694; remote-projection manifest/hash; storage
descriptor equality — equal strides != layout identity Stream 818-825,
ResidencyFabric 3259-3277; wire v8 with explicit v7); A4 planning
(participant prepared-layout in typed inventory; remote pricing +
slot geometry ExpertPreparedMemoryGeometry 181-193; workspace
estimators CPUWeightPreparationMemory 45, CPUProjectionWorkspaceContract
37, WeightMemoryEstimator 550-552; cross-ISA CPU<->CPU refused at
topology time; PMA sole ledger). GATE A: Unit+preflight green Mac/ARM +
x86; byte-identical prepared bytes both ISAs; compact unit coverage
(mixed-stride tier copy, packer presence, endpoint-selected GPU
conversion, transport mismatch rejection). STAGE B (activation:
CompactA64 on AArch64, x86 canonical) GATE B: byte-identical outputs
all formats/codebooks M1..15 serial+grouped on sparks + WSL; GB10
CUDA<->ARM tier round-trip; ABBA 35B+27B nomtp+dynMTP + prefill cell;
PMA BOM/RSS. ACCEPTANCE = latency same-or-better AND capacity better.
STAGE C: L2 IQ2_S source-size as new physical identity (shared
superblock base; byte-9 + u8 coefficient planes; geometry contract for
K-origin/partial superblocks/tails/wire; RN32(RN32(base*(0.5+n))*0.25);
register budget on compiler evidence; own paired bench). LATER: Q5_K-
native; tiering only for expanded-baseline formats; mirror parked
(corrected bound ~4-8%, evidence gate = measured expert-FFN stage
share). ROLES: lane single writer from A0; Main supervises + validates
+ owns post-merge PR. A0+A1 DISPATCHED to arm-idea08 (00:35Z); Main
reports the start to the operator.

### OVERNIGHT WATCH 00:50Z: nominal — Stage A underway, no exceptions

arm-idea08 WORKING on A0 (thinking; ctx 91% — compaction will handle,
branch+origin carry the state). PR 43 OPEN/MERGEABLE (no merge yet).
Branch tip still 3b00c69d7 (A0 not yet pushed — 15min in, expected).
Reviewer idle (#212 delivered+absorbed into the cleared plan). Sparks
idle, hosts free (.79 0 perf procs; .80 0 containers/0 runs — Q1 seal
verified+closed earlier). Main idle post-start-report. Next expected
events: lane A0 checkpoint push -> Main review -> reviewer pin.

### STAGE-A A0 COMPLETE + VERIFIED (01:04Z): 12fa7b2dd — S1 fail-closed + doc corrections; A1 started

Single-writer checkpoint on 3b00c69d7. Code: admitStructure derives
PreparedUnitSections once, rejects unimplemented layout with precise
diagnostic BEFORE any division (:453 before :466 block_stride
consumption); seal/fromPublished route through the one admission
(:261/:278); overflow guard consumes the validated sections.block_
stride. Regression UnimplementedLayoutIsRejectedByViewAndSealAdmission
calls real admission (pre-fix SIGFPE -> post-fix std::invalid_argument,
6 encoding representatives). Docs (same push): gate-based x86 claims,
capacity -33%/+50%/+34-35%, typed-conversion cross-ISA, L2-in-plan,
§10.10 premise correction, §8 consolidated plan. Evidence: layout 8/8,
portability 297/297, register proof 64/64, zero byte change. Diff 3
files +291/-93. --no-verify on Mac per Phase-0 practice (Linux
pre-commit applies at CI). Forwarded to Main (review + reviewer pin);
lane started A1.

### A0 MAIN REVIEW OK (01:05Z): routed to reviewer as the S1 pin; boundary-gate clarification

Main checkpoint review passed (admission-before-guard, shared
admission, real-seal regression, minimal +13/+29). Reviewer holds the
S1 fix pin with the #212 disposition + plan summary. RECORD NOTE:
--no-verify Mac commits are accepted for lane checkpoints; the Stage-A
BOUNDARY gate supplies the Linux coverage (Main-run Linux Integration
build + Unit + preflight on sparks ARM + WSL x86, byte identity both
ISAs) — hook coverage supplied there, not skipped. Lane continues A1.

### A0 REVIEWER SOURCE ACK (01:12Z): 12fa7b2dd — S1 CLOSED; 5 doc fixes ride with A1

Reviewer ACK: rejection before division, regression drives real
MetadataSeal::validate across 6 representatives, already in
ProductionTestPreflight. Design remains qualified. Doc fixes routed to
the lane: (1) wire versioning exact (v7 explicit, pre-v7 rejected;
archive bump != cache-key change — keyed by binding/device); (2) remove
nonexistent "commissioned" CPU<->CPU lane; separate A2 CPU<->GPU
conversion (required) from CPU<->CPU repacker (not built/authorized);
(3) codegen/miss-penalty overclaims -> gate-based; (4) Q8_0 correction
(native int8+half, 2176B = source, NO grid — the 7-8.6x IQ3_S timing
does not transfer across ExpandedInt8; IQ2_S "SLOW class" = unmeasured
hypothesis); (5) Stage C L2 = new physical identity with geometry
obligations, not flat 656B stride; capacity % = idealized predictions.
Attribution: :261 fromPublished structural call, :551 validate call.
A1 continues with the fixes folded in.

### STATE AUDIT 01:23Z (exceptions-only): NO exceptions

arm-idea08 compacted (91%->42% ctx, harness-handled) and is WORKING on
A1 (~50min in; branch tip still 12fa7b2dd — A1 in progress, expected
given its scope). PR 43 OPEN. .79 idle (the "2 procs" probe was grep
self-match; verified empty). .80 idle (old exited containers only).
All sessions in expected states (reviewer/sparks/Main idle post-A0
ACK; CB lead 90% ctx — their program). A2 rider exports landed
(verified earlier). Next event: A1 checkpoint push.

### STAGE-A A1 COMPLETE + VERIFIED (02:06Z): 94679834d — section-presence authority; A2 dispatched

11 files +537/-187 (Stream 165, Packer 121, WindowPacking, View/
PackedWeights, remote-projection + residency fabric + device layout,
new ExpertTierDirectCPUProjectionContract.inc 119, test +49).
Producers/copiers/accessors all derive from PreparedUnitSections
(encoding-only logic eliminated incl. the :755-762 out-of-unit emins
read — wire-byte-identical for canonical); presence equivalence proven
all 21 formats x 2 rotation modes vs legacy predicates; mixed-stride
tier-copy regression (IQ2_S+Q2_K+Q8_0, partial columns, rectangular K,
exact live-byte accounting, foreign-layout rejection); host reference
conversions section-derived (8 sites: 2 cited + 6 found); hot path
untouched (admission-time presence resolution; tile proof 64/64);
portability 297/297, layout 9/9. CAVEAT (Main-review attention): the
mixed-stride regression needs the Linux moe build — lane ran it via a
minimal linked runner against the real implementation + syntax-verify;
real execution at Linux CI/boundary gate. All 5 reviewer doc fixes
folded. Routed to Main (review + pin, parallel with A2). A2 DISPATCHED
(02:10Z): endpoint-layout GPU conversion (CUDA :267-274/:376-385/
:502-506/:557-585; ROCm :552-558/:657-666/:775-779/:828-870 incl. IQ3_S
absolute offset; manifest target-layout contract; typed conversion not
fallback; CPU<->CPU mismatch rejects).

### A1 MAIN REVIEW OK + HANDBACK RUN BOOKED (02:08Z)

Main review: critical copier verified (copyCPUEncodedColumn walks
comp/scales/secondary/emins from cpuSections(); canonical plane offsets
unchanged incl. CMS emins at 4xcolumns; strides layout-derived;
descriptor equality includes layout). Reviewer pin routed. STANDING
RULE applied (Main): lanes compile AND RUN Linux-only test binaries
natively before handback — sparks booked for the focused A1 handback
run at 94679834d (~25min: GCC14 Integration NEON_I8MM, moe unit target
with ExpertTierDirectCPUProjectionContract + prepared-layout +
expert-tier stream tests, JUnit; seal K14-A1HB-94679834D). WSL x86
takes the same set at the Stage-A gate. A2 parallel-approved on
94679834d; rebase if A1 needs a fix.

### BASE FINDING via A1 handback (02:15Z): preflight guard FATALs in no-parity-model configs — not a lane defect

sparks' .79 full Integration configure @ 94679834d FATALs at
tests/v2/CMakeLists.txt:38323: V2_Integration_
DynamicConvergenceFiniteHorizonLifecycle named in the preflight list
(:29675) but unregistered — its add_v2_test (:13862, NO_MODELS,
FORCE_CPU_ONLY) sits in the Qwen35MoE parity block whose executable
(:13802) did not register. COORDINATOR VERIFICATION: registration +
list + guard byte-identical at base d5e04fdce; lane never touched
tests/v2/CMakeLists.txt => BASE defect in the config class lacking
parity model sources. Mechanism: the guard's prune covers BACKEND
omissions only (V2_BACKEND_OMITTED_TESTS); model-source/parity-block
omissions leave the name in the list — exactly the case the guard's
comment claims cannot happen ("never as an unregistered name").
Regression window: inside the PR's develop ancestry (tournament-era
880429059 base configured the full suite fine on .79). PR CI passed =>
canonical companions satisfy the block; non-canonical spark-side runs
bite. ROUTED: Main (diagnosis confirm + fix path [develop-side prune,
symmetric to backend prune, tip frozen] + sanctioned unblock [35B
symlink candidate / official option / hold]); sparks holds .79 at
94679834d, failed cfg preserved, no edits. Lane continues A2
unaffected. BROADER FLAG: post-merge spark-side full-suite runs on
develop hit this until fixed. Artifacts: .79 /home/factory/idea25/
{a1h.sh, a1h-cfg.log}.

### BASE-FINDING CORRECTION + A1 REVIEWER INTERIM FIX-FORWARD (02:22-02:24Z)

CORRECTION (Main diagnosis accepted, supersedes coordinator prune
suggestion): the .79 configure FATAL was ENVIRONMENT — absent
external/cnpy, network-none configure clone fails with only a WARNING,
silently skipping the Qwen35MoE parity block incl. EIGHT preflight
members; the guard at :38323 then correctly FATALed (it worked as
designed — caught a silent preflight-coverage drop). Real defect =
cnpy-missing downgrades to WARNING when preflight members depend on it
(fix: hard configure error, or vendor cnpy like onednn) — base/develop
issue for the base owner, NOT the lane, NOT PR #43, NOT the guard.
Sanctioned unblock: sparks copies cnpy into the tree (the .80 campaign
copy), ZLIB check, re-configure at 94679834d — resumed. BASE-OWNER
RECORD: any network-none full-suite configure on develop silently
drops preflight coverage until fixed.
A1 REVIEWER INTERIM (bounded, continuing): REAL fail-closed gap —
DeviceLayout::valid() (:284-295) compares strides vs layout helpers but
never requires valid/nonzero sections; attack = Q8_0 + CompactA64 +
both strides 0 -> valid() TRUE while cpuStorage().valid() false
(authorities disagree; CUDA validChunk ~:955-959 + ROCm ~:941 trust it
-> zero chunk capacity accepted). FIX dispatched to lane: derive
PreparedUnitSections once in valid(), require isValid() before any
stride comparison (one authority); regression = device-free negative
(zero-stride compact rejected; DeviceLayout/cpuStorage agreement across
21 formats x 2 layouts); A2 rebases onto the fix.

### A1 REVIEWER FINAL (02:30Z): SOURCE HOLD at 94679834d — correction set dispatched as ONE fixup

A1-1 (HIGH) DeviceLayout sections-before-strides both directions +
agreement regression (already interim-dispatched; lane working it);
A1-2 tier-copy test into ProductionTestPreflight via existing filter
(V2CPUFragmentResidencyTests.cmake:89-96, no new gate); A1-3 fixture
strength (non-periodic content vs the 256-periodic uint8(i*41+7)
pattern + wrong-K-origin/wrong-section negatives; destination-sized
extents with succeed-then-flip-only-layout; canonical-preservation
relabel — IQ2_S/Q2_K both carry emins canonically, absent-emins control
lands with explicit compact tables pre-Gate-A; expected chunk-byte
quotes not observed volume). DOC: v7-exactly global (not per-format);
GPU->CPU destination-layout / CPU->GPU source-layout; Q4_1=NibbleLUT,
2176B=source only for Q8_0 (Q8_1 36B blocks); drop resident-expert/
migration overclaims. CARRIED A3/LATER (recorded, not now): copier
ordinal offsets :781-800; CMS host demotion +0/+128/+256 :1582-1587;
Q6 reference 1536/1664 :1496/:1501; PackedWeights chunkScales/Mins/Comp
+ owner chunkMins no absence guard; ResidencyFabric
requireCompatibleCpuWeights :3264-3275 omits layout. Positives: storage
layout fail-closed, copier canonical-equivalent, producers guarded,
accessor presence, selector canonical. HANDBACK BINDING RULE (Main):
JUnit binds to the reviewed fixup tip — sparks' current 94679834d run
completes as environment/surface validation only (keep artifacts,
labeled), re-run fires at the fixup SHA on coordinator GO.

### STATE AUDIT 02:53Z (exceptions-only): NO exceptions — one observation

.79: handback build LIVE and correct — second configure SUCCEEDED
(CMakeCache 02:44Z after the cnpy fix) and ninja is building exactly
the handback targets (v2_test_expert_tier_weight_stream +
v2_unit_moe_overlay_physical_*), 89 compile procs. OBSERVATION: the
old a1h-run.log still ends with the FIRST failed attempt's
"Configuring incomplete" — when sparks reports the seal, verify the
referenced log/artifacts are the second-attempt set (build-a1h/, new
cfg log), not the stale first-attempt file. Lane: A1 fixup in progress
(tip still 94679834d, ~30min in, expected). PR 43 OPEN. Sessions
nominal (reviewer mid-A1-final processing; Main idle; sparks on the
build).

### A1 FIXUP d8eff03bd (03:10Z): coordinator+Main verified, reviewer pin routed, binding handback GO

Single fixup on 94679834d (6 files +300/-48). A1-1 sections-first
validity both directions + manifest valid() fix; regression FAILS
pre-fix (detects the zero-stride attack via both authorities, 21
formats x 2 layouts; promotion sweep = 19 formats — CpuToGpu does not
admit symmetric CMS sources IQ2_XXS/IQ3_XXS, documented). A1-2
preflight via existing filter. A1-3 complete — periodic fill PROVEN
vacuous (rebuilt with index*41+7, wrong-K-origin control fails);
destination-sized + valid-copy-then-flip; canonical-preservation
relabel; quoted_chunk_bytes. Docs: v7-exactly global, direction-
precise contract, Q8_1 36B-block correction, overclaims dropped;
carried A3 items in §8. Evidence: layout 10/10, portability 297/297,
tier runner 3/3 vs REAL production stream source, tile-register 5/5.
A2 reapplied cleanly on the fixup (10/10 combined; remaining: ROCm
sites incl. IQ3_S absolute offset :828, CPU->GPU manifest contract,
fabric callers, endpoint-conversion regression). Main review OK; pin
with reviewer. Binding handback GO at d8eff03bd (JUnit scope per Main:
V2_Integration_ExpertTierDirectCPUProjection 3 cases + prepared-layout
10 + expert-tier stream unit, all --no-tests=error); 94679834d run =
environment-only label.

### A1 SECOND FIXUP DISPATCHED (03:16Z): reviewer closed A1-1/A1-2; narrow test-only HOLD

At d8eff03bd: A1-1 + A1-2 SOURCE-CLOSED, doc corrections accepted.
Remaining (test-only): (1) wrong-section control (Contract.inc:217-245)
reads wrong coordinates/width — equality copies n=76 (col 12, K1) but
the control uses K0/col17 and reads the u32 emins plane as u16 col*2
(high half of emins col 8 vs secondary col 17 — neither is the copied
word). Fix: bind to the exact copied unit/K/column with real element
widths (memcpy byte slices; CMS = copied 4B emins word vs same-column
4B FIRST PAYLOAD GROUP word = the old 1280/period-256 alias witness;
2B metadata = 2B slices); fix the :250-253 "trimmed-unit" mislabel
(canonical units, emins present). (2) The 19-promotion-formats note is
INVENTED — remove it: the loop covers ALL 21 formats both directions
(CMS = codebooks 9/10/13/14/17 all asymmetric; IQ2_XXS 15 / IQ3_XXS 12
= ExpandedInt8 normalizing to codebook 19 at :426-433); promotion_
admitted true for every entry; the duplicated-demotion branch (:454)
is dead — remove the exclusion + comment, do not advertise a support
gap. Binding handback retargets the NEW tip; sparks informed (stop/
rerun pattern; 94679834d env-only label stands).

### 94679834d ENV-VALIDATION COMPLETE (03:18Z): 7/10 pass, 3 fails = the fixup targets — standing rule vindicated

sparks' labeled environment+surface run: 7/10 passed; the 3 failures
(ExpertTierRectangularRemoteProtocol, ExpertTierGpuArrivalProjection
Contract, ExpertTierNativeCPUCompactProtocol) are EXACTLY the A1
fixup's target surfaces — the lane's Mac minimal runner had them green,
real Linux caught them. Textbook vindication of Main's standing rule
(lanes compile AND RUN Linux-only binaries natively before handback).
JUnit+ctest artifacts: .79 /home/factory/idea25/a1h-results/. d8eff03bd
build NOT started (correct hold); sparks waits for the FINAL tip after
the reviewer's second narrow fixup, then runs the exact Main scope
(--no-tests=error) and seals K14-A1HB-<tip>-<ts>.

### A1 SECOND FIXUP 9222e9c7a (03:25Z): Main+coordinator verified; binding handback GO; A2 resumed

Test-only 2 files +43/-43 on d8eff03bd. Wrong-section control rebound
to the exact copied coordinate (n=76/col 12/K1) with real element
widths (CMS 4B emins vs same-column 4B first payload-group word =
1280/period-256 alias witness; Q8_0 2B at the same coordinate); :250-253
claim corrected. Invented promotion exclusion removed (all 21 formats
both directions; dead duplicated-demotion branch + promotion_admitted
gate deleted; 42 canonical positives pass). Evidence: layout 10/10,
tier runner 3/3 vs production sources, portability 297/297. Main diff
check OK; reviewer pin routed. BINDING HANDBACK GO dispatched to
sparks at 9222e9c7a (Main scope: DirectCPUProjection 3 + prepared-
layout + expert-tier stream, --no-tests=error, JUnit; seal K14-A1HB-
9222E9C7A-<ts>; cross-check: the 3 env-run failures must pass). A2
resumed on the new tip (second stash popped; manifest endpoint-layout
parameter settled; v3 wire canonical-bytes until A3 v8).

### A1 REVIEWER BOUNDED SOURCE ACK (03:27Z): 9222e9c7a — A1-1/2/3 closed; handback = ACK + green binding seal

Bounded SOURCE ACK at exact 9222e9c7a: control coordinates verified;
21-format x both-directions x both-labels sweep noted as DEVICE-FREE
coverage (not GPU execution qualification). CARRIED (not closed):
compact absent-section producer/copier/accessor controls BEFORE Gate
A; remaining section-offset/owner contracts (A3 list); A2, A3
(transport/raw-copy identity/v8), A4, Gate A/activation, L2. ACK does
NOT transfer to A2. HANDBACK FORMULA: this ACK + sparks' green binding
Linux seal K14-A1HB-9222E9C7A = A1 handed back. Lane continues A2
(pin to Main when ready).

### STAGE-A A2 CHECKPOINT 85ddd8572 (03:46Z): coordinator-verified, routed to Main; A3 staged

17 files +507/-82 on 9222e9c7a. Endpoint-layout GPU conversion chain
complete: wire planes derived once from CPU endpoint section tables
(fail-closed); valid() re-derives+compares, gpu_has_emins requires a
representing plane (typed conversion); ABI: struct 88B,
MappedTransferPackedWork 128/128 vs 16-word budget (compiled
static_assert — no growth without ABI bump). Manifest reserved byte 0
carries layout (GPU->CPU destination / CPU->GPU source-from-weights);
canonical=0 => wire unchanged; byte outside v3 seal; A3 v8 = promotion
seam. All cited CUDA+ROCm sites consume derived planes with presence
guards (incl. IQ3_S absolute offset :828); residual offset scan clean;
canonical values numerically identical. Regression: 6 kernel families,
preflight-registered; manifest callers thread explicitly; host-ref
round trip byte-exact. Mac: layout 10/10, tier runner 4/4 (real
production sources), portability 297/297, tile 12/12, ABI compiled
check. OPEN at Main: (1) A2 Linux boundary (CUDA compile .80; ROCm
.hip location; integration suites), (2) A3 dispatch timing (held for
A2 review — manifest-seam coupling). Lane stages A3 design (v8
promotion + hash inputs + receiver stride re-derivation + carried
identity items). A1 binding handback at 9222e9c7a running in parallel.

### A2 MAIN REVIEW OK + DECISIONS (03:48Z): CUDA boundary on .79 cu13; ROCm compile-only; A3 DISPATCHED

Main checkpoint OK at design level (planes derived fail-closed, valid()
re-derives+compares, emins never silently dropped). Reviewer pin routed
with 3 focus questions: remaining codebook-based presence decisions in
GPU kernels; plane quotient vs rectangle/partial-tail addresses;
unhashed reserved byte before v8. A2 LINUX BOUNDARY: CUDA = sparks on
.79 with cu13 image (llaminar-arm-engine-sdk-cu13), Integration
HAVE_CUDA=ON sm_121 native + HAVE_ROCM=OFF at 85ddd8572 + cnpy; NATIVE
GB10 GPU run of Test__CUDAExpertTierWeightKernels + CUDA rectangular
contract + CPU tier contract (4 cases) + prepared-layout suite; JUnit
--no-tests=error; spill guard at compile; queued AFTER the A1 binding
seal (K14-A2CUDAB-85DDD8572). ROCm = no AMD GPU: compile-only
ROCmExpertTierWeightKernels.hip on WSL if SDK present (inventory
requested); runtime ROCm = hardware-ineligible on our side, owned by
CI companions at the fast-follow PR, NOT claimed. x86 CPU = WSL at the
Stage-A gate. SEQUENCING = momentum: A3 DISPATCHED on 85ddd8572 (wire
v8 hashed field + receiver stride re-derivation + carried identity
offsets + ResidencyFabric layout) with Main's constraint: nothing
hard-coded beyond the v8 promotion until the A2 pin ACKs; A3 rebases
if the reserved-byte seam changes.

### A2 STOP-AND-FIX + A1 BINDING 7/10 + WSL INVENTORY (03:50-03:53Z)

A2 STOP-AND-FIX (reviewer, Main-confirmed): REAL current-canonical Q6_K
partial-unit regression at 85ddd8572 — CUDA :468/:653 + ROCm :749/:935
replaced literal 16*columns (Q6 high-bit tail start after the 16B/col
LOW-nibble plane) with cpu_wire_payload_per_column*columns (=24 for
Q6) -> high bits at 24C..32C instead of 16C..24C, overlapping scales
(24C)/mins (26C), 4C past the 28C unit end; only PARTIAL/rectangular
Q6 transfers hit (full units take the bitplane path — masked by
full-unit tests). Fix: Q6 low/high split = payload-format property —
dedicated Q6 high-plane quotient or literal 16 + static_assert; audit
ALL other A2 literal replacements; mandatory partial/rect Q6_K round
trip (cols 12+37) host-ref + plane-address equality both directions.
A2 CUDA boundary run CANCELLED at 85ddd8572 (re-fires at fixup tip).
A1 BINDING SEAL K14-A1HB-9222E9C7A-20261113 = HONEST 7/10: the
fixup's core GREEN (DirectCPUProjection red->green across 946->9222;
prepared-layout, stream unit, ProjectionPieces pass) but the SAME 3
instant-RED (RectangularRemoteProtocol, GpuArrivalProjectionContract,
NativeCPUCompactProtocol — same binary as passing cases, Linux-moe-
tree cases never run on Mac, green in .80 canonical campaigns at PR-
tip lineage). A1 NOT handed back until resolved. sparks capturing
verbatim assertions (2 min); lane diagnoses (suspected A0/A1
admission-strictness vs legacy fixtures — diagnose from verbatim).
WSL INVENTORY: no ROCm SDK/GPU (distro nvcc 12.0 not a project
toolchain); Ryzen 7800X3D 8C/16T AVX-512+VNNI+BF16 31GB; x86 gate
image exists. ROCm compile/spill = fast-follow CI image build; ROCm
runtime = hardware-ineligible (stated in PR, not claimed); WSL = x86
CPU qualification at Stage-A boundary (Integration CPU-only, Unit+
preflight, x86 prepared-byte identity vs d5e04fdce); CUDA = .79 cu13
at A2 fixup tip.

### A2 REVIEWER FINAL (04:07Z): SOURCE HOLD — full consolidated fixup set dispatched

A2-1 Q6 subplane origin (named quotient, static_assert-tied to
packer, NOT total payload; Stream :779/:794 + CPU oracle reference).
A2-2 repo-wide API migration (added callers: MappedTransferProgress-
Epoch :1525-1527, MoEOverlayROCmRemoteProjection_MPI :548-556, ROCm
rect .inc:27, three-tier integration; NO default, NO compatibility
overload). A2-3 presence honesty (non-CMS secondary codebook-gated at
10 cited sites + Q6 unconditional — plane-driven OR admission FAILS
on missing endpoint plane, never silent skip; correct "every site
plane-driven" claim). A2-4 reference + controls (configureCpuFromGpu
:396-405 takes destination layout; 5 independent field-tamper
controls; relabel mismatched = known-unimplemented). A2-5 Q6
regression in V2ExpertTierRectangularWeightTests.cmake:39-59 (C=12/37
+ rectangular both directions, C=64 control, repeated/reset, guarded
extents; CUDA native GB10, ROCm compiles PR CI). Reviewer precision:
host-only six-family roundtrip does NOT execute partial GPU paths
(all run reports unauthenticated); reserved byte rejects noncanonical
only because tables unimplemented (not selector-canonical — no
alternate-layout identity qualification). A3 BINDING: layout into
EVERY versioned identity BEFORE compact tables valid (tier-stream v3
bump; remote-projection manifest :709-727 drops layout +
remoteCpuProjectionDeviceLayout :890-894 forces canonical; Packed-
Weights v8). [Main] compact tables invalid/unadmitted until A3
identity lands. [Main] lane builds CUDA/ROCm/perf/parity to syntax
level before handback; next A2 pin routed only after sparks cu13
HAVE_CUDA=ON compiles everything. A0/A1 ACK922 preserved; #212 in
progress; #213 Main-owned. A3 dispatch folds the expanded identity
scope; A3 rebases onto the A2 fixup.

### Reviewer boundary clarification (04:23Z): source vs execution scopes stay separate

Reviewer awaits Main's fresh exact successor request post-build-
sequencing; no autonomous review; 85ddd8572 SOURCE HOLD + A0/A1
ACK922 unchanged. CLARIFICATION (recorded as standing): native/CI
registrations and successful compilation ALONE do not close native
GPU execution coverage or establish partial-unit correctness — the
successor source and any supplied execution reports retain their
SEPARATE review/evidence scopes. No execution, scheduling,
application or activation grant implied. (Consistent with prior
bounded-ACK discipline: ACK is source-scope; runs are sealed
separately.)

### STATE AUDIT 04:23Z + DIAGNOSIS CORRECTED: the 3 reds = unbuilt binary, never ran

Audit + sparks verbatim: the 3 still-red A1-binding tests invoke
v2_unit_moe_overlay_distributed_residency_protocol — NOT BUILT in
build-a1h (verb2 exec: No such file; gtest filters vs
v2_test_expert_tier_weight_stream = 0 tests from 0 suites — names
don't exist there). ProductionTestPreflight members from
V2ExpertTierRectangularWeightTests.cmake:16/:23/:83 with
FAIL_REGULAR_EXPRESSION "0 tests from 0 test suites" — the framework
catches exactly this. They failed identically at 94679834d and
9222e9c7a (missing-binary exec, content-free). MAIN'S NAMED SCOPE AT
9222e9c7a = ALL GREEN. Sparks' "3 = exactly the fixup targets"
inference WRONG (mine too — propagated as the cross-check
expectation); the honest 7/10 + verbatim artifact discipline caught
it. Qualification call with Main: (a) accept in-scope-green or
(b) build the one missing target + run the 3 + family airtight
(coordinator recommends (b); A2 boundary build needs the binary
regardless — the A2-5 Q6 family lives in the same file, and the 3
tests exercise the lane's touched surfaces with .80-campaign green
baselines). .80 q1bw sealed/no containers; station CB lead-owned,
no drift signal; A2 rider exports pending with the fixup (in scope).

### A2 CONSOLIDATED FIXUP 433e84632 (04:59Z): coordinator-verified, Main review + boundary build dispatched

19 files +616/-38 on 85ddd8572, ONE push, all five items + binding A3
(verified: kHostPreparedLayout gone from Stream, no default arg).
A2-1 Q6 low-plane quotient 16 (static_asserts ==16/!=24 both kernel
files; CUDA :483/:668, ROCm :764/:950; copier+oracle = reference;
device-free regression 5/5). A2-2 all 37 callers migrated (grep zero
unpatched). A2-3 admission fail-closed: valid()s require EXACT
metadata presence contract (comp iff !Q6/CMS; secondary iff asym or
NLSS/IQ3S/Q6/CMS; emins iff CMS); canonical tautological; "every
site plane-driven" corrected (doc §8 + commit msg). A2-4 reference
takes required destination_layout (20 callers; 5 field-tamper
controls; relabel). A2-5 per-backend V2_Integration_${backend}_
ExpertTierPartialQ6PayloadPlane (12/37 + 64 control, dirty replays,
0xa5 guards, live-row bytesMoved). A3 binding in doc §8; compact
tables invalid until identity lands. Mac: 297/297 + 10/10 + 5/5;
Mac can't compile CUDA/ROCm/perf/parity (known blockers); TUs
syntax-swept; Q6 test bodies name-verified. DISPATCHED: sparks A1
completion run (5 min, 9222e9c7a, 3 members + family, seal amended)
then A2 boundary build at 433e84632 (cu13, HAVE_CUDA=ON sm_121,
HAVE_ROCM=OFF, cnpy, EVERYTHING incl. distributed_residency binary;
native GB10 test set incl. new Q6 registration + 3 members; JUnit
--no-tests=error; K14-A2CUDAB-433E84632). A2 pin to reviewer waits
on Main review + build. A1 qualification call pinged at 36m.

### WATCH 05:20Z: A1 completion GREEN; A2 boundary build running; A3 resumed

A1 completion run @ 9222e9c7a GREEN (a1h-completion.log CT=0;
junit-completion.xml tests=5 failures=0 @ 05:11Z) — the 3 previously-
red members + family pass with the binary built. COMPLETE A1 artifact
set now exists: reviewer ACK @ 9222e9c7a + named-scope green +
completion green + amended honest seal + disclosure. Main's
qualification call = acceptance (routed 05:21Z). A2 boundary build
RUNNING on .79 (tip-a2b @ 433e84632, configure OK 05:16Z, build log
growing; hardened pattern). Lane re-pinged to resume A3 on 433e84632
(v8 promotion + carried identity offsets; v8-only constraint until
A2 pin ACKs; A3 = the gate that unlocks compact tables). Main idle
with A2 fixup review pending (21m) — within threshold after the
04:59Z ping. PR 43 OPEN. .80 sealed/idle.

### A3 IDENTITY BINDING LANDED c7afb6eae (06:09Z): all three versioned surfaces sealed

14 files +819/-96 on 433e84632 (verified: v8 prepared_layout sealed
field @ PackedWeightsSerialization.h:86; exactly-v8/v7 admitted, v6-
global). Tier stream v4: layout in manifest hash; receivers re-derive
strides from carried tables; NEW seal control — compact byte with
correct hash still rejects on unimplemented tables (no hash/stride
substitution for layout identity). Remote projection ABI 5: sealed
byte inherited (no drop); derivation from carried byte (canonical
force removed); unimplemented -> zero strides -> validity rejects;
requireCanonicalCpuRoot = CPU root must BE the sealed layout (no
CPU<->CPU repacker; mismatch typed-rejected); wire 244B unchanged.
Archive v8: reserved[18]->[17] promoted + sealed; one fixture
canonicalized 512/640->1024/1280 (non-canonical unrepresentable);
all others stride-probe-verified. Carried offsets landed (Q6
quotient static-assert-tied; CMS demotion absolute; chunk arithmetic
preparedSections() + chunkMins guard; requireCompatibleCpuWeights
includes layout). Regressions: ArchiveLayoutIdentity (Unit+
preflight), RemoteProjectionIdentitySeals, OwnerMetadataAccessors.
COMPACT TABLES INVALID/UNADMITTED per sequencing rule; stride tables
unlocked but HELD for Main's A2 pin ACK. Mac 297/297 + 11/11 + 5/5;
archive suite Linux-only; protocol TU Mac-uncompilable. A3 BOUNDARY
BUILD DISPATCHED at c7afb6eae (queued behind A2 tests on .79;
extended set incl. full Unit gate + new archive TU + distributed-
residency; K14-A3CUDAB-C7AFB6EAE). Main unresponsive 48m re A1
acceptance + A2 review (two pings) — momentum dispatch under
standing authorization; pin routing stays Main's; morning flag if
still silent.

### A2 BOUNDARY BLOCKED (06:26Z): real g++-14 compile defect — fix-back to the lane

A2 build at 433e84632 (cu13 CUDA sm_121) fails: NativeQuantisedFormat-
Info.h:115 defaulted operator== ill-formed under g++-14 (member type
non-comparable); 4 TUs. Evidence: .79 a2b-build.log (391/454). Mac
clang syntax-pass vs Linux g++ reality again. DECISION: lane fix-
forwards on the A2 line (fix the member comparability, not delete
the ==) -> new A2 tip; A2 pin moves there (never routed — Main's
build rule held it for exactly this). A3 rebases. Strict exact-tip
runs: A2 boundary at fixed tip (K14-A2CUDAB-<fixed>), then A3
boundary at rebased tip (ccache makes it cheap); queued c7afb6eae
build SUPERSEDED (carries the same defect). A1 completion sealed
green stands. Sparks holds, no source edits (correct); a2b logs =
defect evidence for the reviewer.

### VERBATIM CORRECTS DIAGNOSIS (06:40-06:50Z): 4 errors, 2 classes; EQ fix insufficient; mis-paraphrase on record

Direct a2b-build.log extraction: Class 1 (x2) = defaulted operator==
'only available with -std=c++20 or -std=gnu++20' — std-mode mismatch
in 2 TUs (others build -std=gnu++23); NOT member comparability (the
relayed "member lacks equality" note = mis-paraphrase, corrected on
record; container = aarch64 g++-14 not x86-64). Lane's explicit
member-wise comparison at 40cfbaf3e resolves it (C++17-compatible).
Class 2 (x2, NOT covered): CUDAExpertTierRectangularWeightContract
.inc:103 'const auto* q6=nullptr' deduction failure + :161 'bytes'
not declared — TYPE/SCOPE errors in the A2-5 registration body
(name-verification cannot catch these). Lane fixing: both .inc
errors on the A2 line (real A2 tip = follow-up on 40cfbaf3e), ROCm
counterpart swept for the same pattern, A3 rebases, full type/scope
sweep of the new Q6 bodies. Sparks holds; a2b logs = evidence;
verbatim + corrected diagnosis go to the reviewer with the A2 pin.
Relayed-diagnosis verification rule vindicated again (error text,
not just test results).

### WATCH 06:50Z: Main 90m silent — consolidated digest sent; morning flag set

Branch tip 907717f0d (A3 rebased; .inc fix-back in flight — lane
compacted to 28% ctx, working). Main idle 90m on A1 acceptance +
A2 review routing (pings 04:59Z/05:53Z unanswered). Digest sent
06:50Z: (1) A1 = one word, artifacts complete since 05:11Z; (2) A2
review target MOVED to the corrected tip (compile fixes; verbatim +
corrected diagnosis ready for the pin); (3) A3 source review can
start anytime. PR 43 OPEN; sparks holds .79 (A1 completion sealed
green); reviewer held for successor request. MORNING FLAG: if Main
still silent, the operator report notes the 90m+ unresponsiveness
(material work unblocked — all evidence produced and stable; only
pin routing + the A1 word pending).

### A2/A3 COMPILE FIXES COMPLETE (06:59Z): b37537145 + c24673b2a; boundaries dispatched

Corrected chain verified on origin: 433e84632 -> 40cfbaf3e (EQ) ->
b37537145 (REAL A2 TIP; concrete QuantizedVerifierFormatCase* q6 at
:103 in BOTH backend .inc files) -> c24673b2a (A3 rebased). Root
cause class 2: auto*-from-nullptr ill-formed in every std mode; the
'bytes' error = g++ error-recovery cascade (one root error, not two
defects). Class 1 re-verified in container mode: the 3 gnu++17 TUs
compile clean with the EQ fix; defaulted == was their only C++20
dependency. Reviewer flag: whether prefill/trainer targets gain
-std=gnu++23 = build-config question, outside the lane. New lane
gate: host-only syntax probes of both real contract bodies under
g++-14+clang gnu++20 (would have caught :103 pre-boundary); contract
TUs are C++20+ by construction (ends_with). Gates at c24673b2a:
297/297 + 11/11 + 5/5 + probes 0 errors. DISPATCHED: sparks A2
boundary K14-A2CUDAB-B37537145 (full A2 set + distributed-residency
binary) then A3 boundary K14-A3CUDAB-C24673B2A (adds full Unit gate +
archive TU + A3 regressions); verbatim-first reporting enforced.
Main routing: review targets = b37537145 (A2) + c24673b2a (A3);
A1 acceptance still pending.

### A2 BOUNDARY DISK-FULL (07:35Z): capacity not code; prune+resume dispatched

a2c build at b37537145: all 3225 compiled targets CLEAN through the
fix set (the source fixes held); the 41 "errors" = 39 ld failures +
2 depfile writes, all "No space left on device" (df: 1.8T 100%, 94M
free — tonight's three trees consumed the 81G headroom: a2b 38G +
a2c 33G + a1h 5.6G + tips ~7G). DISPATCHED prune order: (1) build-a2b
+ tip-a2b (~40G, superseded failed attempt; logs stay as reviewer
evidence); (2) build-gb10i 35G after verifying its seal evidence is
outside the tree; (3) build-a1h+tip-a1h (artifacts preserved in
a1h-results/, coordinator-verified) + old tip checkouts if <60G free.
Then resume a2c (objects intact; failed links + ~440 targets only) ->
test phase -> K14-A2CUDAB-B37537145 -> A3 boundary at c24673b2a.

### WITNESS FAILURE AT A2C (08:20Z): envelope root-claim REFUTED; base-vs-lane discriminator dispatched

Tile-register witness fails at a2c resume: 3 programs spill over
budget (3:2:0:2:0:0 5>3; 3:2:1:2:0:0 5>3; 4:2:0:1:0:0 4>2), g++-14
GNU-14 NEON_I8MM. Sparks' root claim ("a2c envelope adds +fp16fml,
A1 was +i8mm-only and clean") REFUTED from the cfg logs: a1h and a2c
configure IDENTICALLY (+simd+dotprod+i8mm+fp16fml, NEON_I8MM); the
lane diff does not touch the envelope; the A1 TARGETED build never
generated the witness (no A1 observation exists). This = the first
witness run under container g++-14 on this branch. DISCRIMINATOR
dispatched to lane: run the witness script on BASE d5e04fdce probe
source with identical flags — base-fails = pre-existing base gap
(clang-tuned budgets never validated under Debian g++-14; route to
Main/reviewer, not a lane regression); base-clean = lane's Phase-0
tile changes regressed g++-14 while clang-only 12/12 evidence stayed
green (fix register pressure Phase-0-style, NO budget increase; add
a g++-14 leg to the tile gate). Plus local Homebrew-g++-14 repro with
exact ninja flags. Sparks: a2c resume HELD; no envelope reconfigure,
no witness/budget edits. Disk event closed: pruned to 77-80G free,
seal evidence preserved (a2b logs kept as reviewer evidence).

### MILESTONE: A1 HANDED BACK (08:25Z); A2-3 design-defect fixup routed

MAIN FORMALLY ACCEPTED A1 at 9222e9c7a (basis: reviewer bounded ACK
+ named-scope green + completion 5/5 junit-completion.xml + amended
honest seal with disclosure). STAGE A's FIRST STAGE CLOSED.

A2 source review @ b37537145 (Main): Q6 static-assert fix OK,
caller migration OK, field-tamper controls OK, EQ fix OK. ONE DESIGN
DEFECT before the pin: A2-3 validators hard-code CANONICAL presence
(comp == !(Q6||CMS); emins == CMS) — would reject Stage B's compact
layout (drops comp on ARM; drops non-Q2_K emins); reintroduces
encoding-only presence logic A1 removed; tautological today which is
why nothing failed. CORRECT = DATA-requirement contract: has_
secondary == required secondary (keep); emins: source->has_emins =>
has_emins (Q2_K) + !CMS => !has_emins (canonical zero-filled emins
for other CMS stays allowed); compensation = derived x86-kernel aid,
presence belongs to LAYOUT TABLES not validators (GPU conversion
touches it only if the endpoint plane exists; x86 admission already
requires canonical). Regression: synthetic table without comp +
without non-Q2_K emins via explicit layout arg must PASS; missing
Q2_K emins or required secondary must FAIL. Fixup routed to lane
(b37537145, A3 rebases); pin = fixup + sparks A2 boundary result.
A2 boundary at b37537145 SUPERSEDED (a2c held ccache-warm, no test
phase); re-fires at fixed tip + witness-discriminator resolution.
Main reviews A3 after the A2 fixup (it rebases); wants both boundary
results (build OK/fail + JUnit counts) when sealed.

### A2-3 FIXUP bc91b0d71 + WITNESS FRAME-CHANGE (08:55Z): accounting artifact hypothesis

A2-3 fixup (3 files +153/-46): named contract sectionPresence-
SatisfiesSourceContract in the footprint header, both validators call
it. Spec + one forced refinement: emins scoped source.has_emins &&
encoding==CMS => has_emins (Q2_K ROTATED consumes emins into payload
legitimately; unrotated Q2_K IS CMS so the specified regression still
fails); !CMS => !has_emins; non-Q2_K CMS zero-filled OR dropped both
pass; compensation REMOVED from validators (layout tables own it).
Regression: Stage-B synthetic (no comp/no emins/Q6_K) passes; the
three FAIL cases exact; 21-format canonical both rotations
tautological. Gates: 12/12 + 297/297 + 5/5 + probes 0 errors.

WITNESS REFRAME: branch moved g++-14 codegen but the 3 failing
programs' movement = ACCOUNTING ARTIFACT — A1's view-field spans
(metadata_secondary_/metadata_emins_) shifted the frame so a q15+q20
stp/ldp PAIR (2 instr, adjacent slots) became str/ldr singles (4
instr, non-adjacent); same registers/values, zero pressure change;
witness counts instructions -> +2 = exactly 3->5/3->5/2->4. Mac
g++-14 != container (directional only). Container census dispatched
to sparks (seconds/side; prediction base 3/3/2 vs tip 5/5/4). If
confirmed: Main/reviewer choose (a) Phase-0 restructure removing the
pre-existing pair spill or (b) accounting/witness policy; if base
also 5/5/4 = stale budgets = base gap. Sparks sequence: census ->
A2 boundary at bc91b0d71 (target-set builds; witness not a seal
blocker) -> A3 boundary at 365c29aeb. Main's A3 review can start on
365c29aeb.

### MAIN DECISIONS 08:58Z: A2/A3 source-OK; witness = option (a) source fix before the pin

A2 @ bc91b0d71 source review OK (contract keeps secondary as source
data; emins directional; rotated-Q2_K refinement right; compensation
out of validators; regression shape right; pin waits on the fixed-tip
boundary). A3 @ 365c29aeb first-pass source OK (archive v8 sealed
byte/exactly-v7/global pre-v7/strides re-derived; stream v4 hashed;
ABI 5 sealed+derivation+typed mismatch rejection — matches reviewer
(c)); pin waits on its boundary. WITNESS: option (a) FIX THE SOURCE;
budgets + witness policy UNCHANGED. Cause = A1's metadata_secondary_/
metadata_emins_ view members changed the tile-carried object/frame
(q15/q20 pair -> singles). Phase-0 lesson: restore base layout of
tile-captured state — fold into existing member/padding (sizeof+
order = base) OR keep out of tile state (presence via seal/section
authority in cold accessors). Acceptance: container census reads
base counts (3/3/2) at fixed tip + tile proof green on shipped
arches; no budget edits/allowlist. GATE: fix only if census confirms
base 3/3/2; if base 5/5/4 = stale budgets = base-owner item, STOP.
Witness fix BEFORE the A2 pin (fixup on bc91b0d71, A3 rebases, A2
boundary at the FIXED tip). Sparks re-dispatched: census routes;
hold boundaries for tip update either way.

### CENSUS CONFIRMED (09:08Z): base 3/3/2 vs tip 5/5/4 — accounting artifact verified; witness fix GO

Container g++-14 census on .79: base d5e04fdce = 3/3/2 (the budgets
exactly) vs tip b37537145 = 5/5/4 — the lane's stp/ldp-pair-split
prediction confirmed instruction-for-instruction. NOT a base gap.
Evidence: /home/factory/idea25/{base-d5e,tip-a2c}/{census.log,
build-census.log}. Option-(a) witness fixup GO on bc91b0d71 (restore
base tile-captured object/frame layout; fold members or cold-
accessor resolution); acceptance = census 3/3/2 at fixed tip + tile
proof green on shipped arches (sparks re-runs the census leg at the
fixed tip). A3 rebases; both boundaries re-fire at the fixed tips.

### WITNESS FIX d0b43d81b (09:17Z): option (ii) whole-object restoration; boundaries dispatched

Fixup 3 files +34/-25 on bc91b0d71 (verified on origin: the four A1
view members GONE from the view; sizeof 56 == base). Option (ii):
hot accessors = canonical predicates over tile-carried members
(= canonical section tables by the pinned sweep — authority
preserved); layout-aware authority in COLD paths only — fromPublished
derives tables at admission + EXPLICITLY rejects non-canonical (the
gate is explicit, not a missing-tables artifact); cold layout()
accessor for owner/publication agreement (PackedWeights rerouted,
no other consumers). Acceptance local: probe object instruction-
IDENTICAL to base (0/322 symbols, 5929=5929 spills, all 176 programs
at base counts). No budget/witness edits. Gates: 11/11+12/12,
297/297, 5/5, probes 0 errors. A3 rebased -> 848030067. WATCH
CAUGHT a 30-min dispatch gap (lane report verified but sparks re-
fire unsent) — closed at 09:50Z: re-census acceptance -> A2
boundary K14-A2CUDAB-D0B43D81B -> A3 boundary K14-A3CUDAB-848030067;
pins await the runs.

### WITNESS FIX Main-OK (09:52Z); Stage-B design note routed to lane §8

Main OK @ d0b43d81b pending the container census (3/3/2 = acceptance
authority). DESIGN NOTE (Main, verbatim, into lane §8 now): Stage B
must NOT reintroduce per-view runtime layout state in hot accessors —
view admission accepts exactly kHostPreparedLayout, so hot accessors
(chunkScales/chunkMins/chunkComp/emins offsets) compute offsets from
CONSTEXPR host-layout tables (compact on aarch64, canonical
elsewhere): zero new members, no per-call branching on stored
layout, compile-time constants per encoding. Stage-B tile proof +
census then check compact ADDRESSES, not object-layout drift.
Witness budgets unchanged; any Stage-B delta explained instruction-
by-instruction. Pins go out on green boundaries at d0b43d81b +
848030067.

### Lane §8 record confirmed (09:55Z): 0607ca73a, boundary-neutral, left in place

Stage-B hot-accessor rule recorded VERBATIM in §8 (commit 0607ca73a
on 848030067, --no-verify Mac checkpoint per standing rule) with the
witness provenance (A1 per-view members -> object growth -> spill-
frame shift -> stp/ldp split) so the constraint carries its reason.
Chain: d0b43d81b (A2 tip) -> 848030067 (A3) -> 0607ca73a (doc,
boundary-neutral — seals unaffected; left above the tips, no rebase
churn). Next: census acceptance + both boundaries -> pins on green.

### AUDIT 10:23Z EXCEPTION: a2d test phase failed operationally (caught pre-report)

Artifact check (a2d = A2 boundary @ d0b43d81b): BUILD_OK but ctest
died at CTestTestfile parsing — the target-set build never generated
v2_integration_captured_transfer_channel_undiscovered_Debug.cmake
which the full test registration includes; 0 tests ran; the script
still printed DONE (pipeline-masked-exit class). CORRECTIVE
dispatched: continue to the FULL build at d0b43d81b in the same tree
(objects retained; witness generation should pass at base counts
post-fix), rerun the A2 filter + JUnit + --no-tests=error, seal
K14-A2CUDAB-D0B43D81B from the complete run; report the container
re-census (step 1, still outstanding); runbook note — DONE markers
must reflect per-phase exit codes. A3 boundary (848030067) queues
after. .80 idle; sessions staged as designed; PR 43 OPEN.

### PARALLEL WORK ROUTED (10:30Z): A4 dispatched; Main A3-review completion + pin pre-staging

Operator question "forced to wait for sparks?" -> NO: only the two
PINS sit behind sparks' full build + census. Parallel dispatches:
(1) A4 (final Stage-A stage) to the lane on 848030067 — typed
inventory publication, cross-ISA CPU<->CPU topology-time refusal
(typed diagnostic before allocation/wire), PMA sole-ledger alignment;
rebases if boundaries demand fix-forwards; Mac gates at checkpoint.
(2) Main (idle) to complete the FULL A3 source review at 848030067
(clean rebase of the first-pass-OK 365c29aeb) so the A3 pin fires on
the seal alone. (3) Pre-stage the reviewer A2 pin package (verbatim
bundle + census evidence + corrected diagnosis + witness provenance)
for immediate successor request on green. With A4 in flight, every
session productive; pins release with zero added latency once the
boundaries land. Sparks: full-build corrective running (a2d).

### A3 FULL REVIEW COMPLETE (11:10Z): Main OK with carried A3-B1; pins pre-staged

Main full source review @ 848030067 OK: archive v8/exactly-v7/global
pre-v7 + stride re-derivation in BOTH deserialize AND direct-MPI
allocateTransferTarget (reviewer's before-allocation point met for
stride identity); stream v4 hash; ABI 5 sealed/no-force/typed root
mismatch; fabric layout equality; accessors section-derived. ONE
BINDING GAP (A3-B1, pre-Stage-B, does NOT block the A3 pin; folded
into A4): receiver EXECUTABILITY unchecked at admission — both
receive paths admit any known-valid-tables layout; harmless today
(canonical-only tables) but post-unlock an x86 rank would allocate +
adopt compact bytes it cannot execute (view rejects only at
fromPublished). FIX: execution-bound receive paths require
admitted_layout == kHostPreparedLayout BEFORE allocation, typed
rejection; byte-preserving relays name themselves explicitly.
Regression: valid non-host v8 header rejected pre-allocation on both
paths via the explicit-layout test hook (compact tables valid in
test only). A3-B1 = the receive-side twin of A4's planning-time
refusal (planning refuses; receivers fail closed if bypassed).
MAIN PIN PACKAGES PRE-STAGED (A2: d0b43d81b lineage + census + std-
mode/auto-nullptr bundle; A3: + A3-B1 flagged carried) — both go out
on green boundaries. A4 (+A3-B1) in flight on the lane.

### A2 BOUNDARY SEALED (11:16Z): witness CLOSED; 2/18 fail -> fix-forward loop

K14-A2CUDAB-D0B43D81B-20261113: FULLBUILD=0, witness PASS 176 at
base counts, re-census 3/3/2 — WITNESS-FIX ACCEPTANCE CONFIRMED (all
3 criteria). Tests 16/18 green (CPU tier/layout/stream/distributed-
residency + Captured Replays + RectangularWeights). 2 FAIL (one
root, shared .inc): CUDA_ExpertTierWeightKernels + CUDA_ExpertTier-
PartialQ6PayloadPlane @ CUDAExpertTierRectangularWeightContract.inc:
121 — forward.columns() 128 vs 64 (CUDA forward view reports 128
where the contract case constructs 64). TWO-READING diagnosis routed
to lane: (a) padded/aligned granularity vs constructed-live = test-
expectation fix (likely — both binaries share the new Q6 body), or
(b) A2 made columns() report allocation width = REAL live-extent
regression = source fix. Fix on the A2 line; A3 (boundary running,
will show the same 2 = expected) + A4/A3-B1 rebase. A2 pin waits on
the green re-run; census/witness/full-build seal portions stand; pin
packages pre-staged.

### A2-B2 DIAGNOSIS (Main, 11:18Z): test-expectation bug; verbatim fix spec routed

The 2 CUDA reds = TEST-EXPECTATION bug in the new Q6 contract (not
live-extent): source P spans [32,544), target P{...,72,160,64,128}
spans [64,192) -> intersection 128 = what production correctly
returned; the comment intends [64,128) = 64-col window but the
target was built with 128 cols. FIX (Main, verbatim): target
P{R::ParticipantInputSlice,256,1024,72,160,64,64}; KEEP ASSERT_EQ(
forward.columns(),64); re-check every derived quantity assuming 64
cols (oracle cpu_first_column=32, block/guard extents, bytesMoved);
same fix in the ROCm counterpart (CI-compile-only — correct by
inspection); do NOT weaken the assertion to 128 (partial window
with exact whole blocks is the point). Sparks re-runs the 2 CUDA
cells (+full set if cheap) at the fixed tip; A2 pin waits on green.

### A2-B2 FIXUP 1f92dca1c (11:47Z): reading (a) confirmed; re-runs dispatched

Chain: d0b43d81b -> 1f92dca1c (A2-B2, 3 files +66/-4 test/inc) ->
17789eb05 (A3 rebase) -> 0a80edd6a (§8 doc). Two stray fragments
carried the 128 (target projection columns=128 + destination root
K=128/blocks_per_row=4); fix per Main's spec (target 64,64; root
K=64/bpr=2; assertion KEPT; derived quantities consistent: units=2,
block_stride 1792; ROCm mirrored). Teeth demo: bad construction
fails exactly as the boundary did. Blind-spot closure: device-free
mirror PartialQ6WindowGeometryPinsLiveRowsAndTwoWholeBlocks (host
contract .inc via production APIs; pins geometry incl. chunk totals
{12,37,64}). Gates: 4/4 + 5/5 + probes 0 errors + 12/12 + 297/297 +
full rebuild 0. RE-RUNS dispatched: in-flight 848030067 completes
(diagnostic; 2 reds expected) -> A2 binding at 1f92dca1c (K14-
A2CUDAB-1F92DCA1C) -> A3 binding at 17789eb05 (K14-A3CUDAB-
17789EB05); strict-tip rule both. Pin asks to Main: spec-conformance
OK on 1f92dca1c + mechanical-rebase confirm for 17789eb05; pins fire
on green + OKs. A4/A3-B1 continues on the new tip.

### PIN CONFIRMATIONS (11:51Z): A2/A3 pins fire on sparks' green seals

Main: (a) 1f92dca1c conforms to the A2-B2 spec (target/root mirrored
CUDA+ROCm, assertion kept; the device-free mirror = good addition);
(b) A3 17789eb05 = mechanical rebase (range-diff: 848030067 =
17789eb05 patch-identical; tree delta = exactly the 1f92dca1c fixup;
0a80edd6a docs-only). BOTH PINS FIRE on sparks' green seals at
1f92dca1c + 17789eb05. A3's pin carries A3-B1 (receiver executability
before allocation) as a KNOWN ITEM — A4 or a fixup must close it
before compact tables are unlocked. Board: sparks on the two binding
re-runs; lane on A4/A3-B1; reviewer loaded with pre-staged packages;
PR 43 OPEN.

### AUDIT 11:53Z EXCEPTION: a3b ctest phase never ran (missing results dir)

A3 diagnostic run @ 848030067: full build CLEAN (BUILD=0, DONE_BUILD,
1.48MB log — build evidence stands) but ctest never executed — the
output redirect failed on the never-created /w/a3b-results/ (CT=1,
zero tests). Second DONE-marker-class incident tonight. Corrective
dispatched: mkdir + rerun the ctest phase only (2 shared-.inc cells
expected red at this pre-fix tip, rest green = the diagnostic
artifact set), then the binding sequence (A2 @ 1f92dca1c -> A3 @
17789eb05; both pins armed). Runbook addition: post-phase artifact-
existence checks in all run scripts. PR 43 OPEN; .80 idle; lane on
A4/A3-B1 (in flight).

### OPERATOR DIRECTIVE (12:00Z): "the main arm session should always push for transparency"

Standing rule: Main pushes every checkpoint/review/rebase/merge/
application state to the canonical remote — no Main-local-only
states; reports reference pushed SHAs. Extends handback + push +
handover-cleanup policies (the new explicit part: Main's
INTERMEDIATE states are also always pushed, reported by pushed SHA).
Relayed to Main; current compliance verified clean (PR branch
d5e04fdce pushed; compaction chain lane-pushed through 0a80edd6a;
today's reviews were report-based with the lane holding commits).
Recorded in program memory (project.llaminar.arm.main_always_pushes).

### TRANSPARENCY SWEEP COMPLETE (12:37-12:50Z): always-push adopted; unpushed state fully dispositioned

Main adopted the operator's always-push directive (every apply/rebase/
integration pushed immediately; reports cite pushed SHAs). Main's
compliance check: clean (checkout = d5e04fdce = origin; no Main-local
commits; one stray debris file removed). COORDINATOR PUSHES (standing
handover-cleanup rule, on the lanes' behalf): six completed local-only
branches pushed to origin — codex/arm-addon 8fcc0fe0b, arm-b9-
strengthening 825fe8deb, arm-s4-nibble-witness a16591eed, arm-decode-
team-placement c3060d607 (96-commit lineage), i8mm-typed-placement
ea5402a54 (also the ideas-25/7 base), arm-idea26-openmp-wakeup
3b9f0cad0. OPERATOR DECISION on the two KEEP items (arm-witness-post-
index c4536cf3c, arm-half-perf-cell b512e25a3): KEEP LOCAL ONLY —
stays local-only as preserved; no push. HELD: bundle1-perf-harness
(successor re-rebase plan + force-with-lease from 5d606a108).
UNPUSHED SUPERSEDED (content in the PR): arm-b1-fused-fork-pooling,
arm-b4-vocab-budget, arm-idea23-hc-classification. Repo's entire
unpushed state = exactly the four dispositioned categories; nothing
completed is local-only anymore.

### BRANCH REMOVALS (operator directive, 12:50Z) + A4+A3-B1 CHECKPOINT ROUTED

OPERATOR RULE (verbatim): "Anything that is obsolete should be
removed. And any kind of evidence should be on the arm branch if
really needed for it, if just interim for validation then it can be
dumped." EXECUTED: the six branches I pushed earlier were REBASED
lineages (verified: not ancestors of d5e04fdce; content in the PR)
= obsolete/interim -> ALL SIX REMOVED from origin + locally
(arm-addon, arm-b9-strengthening [B9 closed/benched, decision
evidence in tracker], arm-s4-nibble-witness, arm-decode-team-
placement [96], i8mm-typed-placement [94], arm-idea26-openmp-wakeup);
their two lingering retired-lane worktrees removed (clean at tips);
the three local-only superseded lineages deleted too (b1-fused-fork-
pooling, b4-vocab-budget, idea23-hc-classification). REMAINING
locals: KEEP items (witness-post-index, half-perf-cell — operator
local-only decision), idea08 (ACTIVE), idea16-moe-decode +
idea16-tournament-repair + idea23-hc-planning (all origin-durable),
perf-review + qfn (origin), bundle1 (successor plan held). Earlier
"origin-durability" pushes of retired lineages = over-preservation,
corrected.

A4+A3-B1 @ 569de492b (18 files +761/-24; A0-A4 COMPLETE, Stage A in
review position): A3-B1 requireHostExecutablePreparedLayout gates
BOTH receive paths pre-allocation (execution-bound by construction;
no relay path exists); test hook + regression (compact Q4_0 with
valid strides rejected by THE GATE on both paths + controls). A4-1
inventory cpu_prepared_layout (build property, never probed, unknown
FATAL, wire round-trip + tamper). A4-2 overlay binder uniformity
refusal pre-allocation (typed; auto+authored; resolved layout on
plan; 4 regression cases). A4-3 pricing through the same section
authority + typed BOM input + slot config; documented §8 deferral
(pool owners = CPU participants -> A4-2 uniformity suffices; Stage-B
flips kHostPreparedLayout). Design note: planning defaults =
ambient build truth (vs A2's explicit movement identity); receive
gates = the bypass backstop. Gates: 12/12 + 297/297 + 4/4 + 5/5 +
probes. ROUTED to Main: review request + Stage-A gate sequencing
(strict-tip note: A2/A3 seals still fire at their tips + the gate at
569de492b incl. the WSL x86 leg; cost call Main's).

### A4 REVIEW: F1+F2 ROUTED; STAGE-A GATE SEQUENCING SET (12:55Z)

Main @ 569de492b: A3-B1 right (pre-allocation, both paths), A4-2/
A4-3 OK. TWO FIXES pre-pin: F1 kInventoryVersion stays 7 while A4-1
adds an inventory-record byte -> bump to 8 + comment + regression
(v7-shaped record rejected BY VERSION). F2 ExplicitPreparedSections-
ForTest = process-global mutable fn-pointer consulted by production
deserialize/allocate (test override reachable from production
validation; non-atomic across threads) -> replace with DI: internal
admission helpers take the sections resolver as a parameter (internal
overload); public paths use the production authority only; tests
use the internal overload; same regression shape. ONE fixup on
569de492b. SEQUENCING (Main): (1) sparks A2 2-cell rerun at
1f92dca1c -> A2 pin on green; (2) A3 pin to reviewer NOW on source
at 17789eb05 (patch-identical to 848030067; only the 2 expected
shared-.inc reds; NO separate A3 binding run — the Stage-A gate on
the descendant covers it); (3) F1+F2 fixup; (4) STAGE-A GATE ONCE
at the fixed tip: sparks = Linux Integration NEON_I8MM + HAVE_CUDA=
ON cu13, full Unit + ProductionTestPreflight via run_production_
prerequisites.py, CUDA expert-tier native set, JUnit + ARM prepared-
byte identity vs d5e04fdce; WSL = x86 Integration CPU-only, Unit +
preflight, x86 prepared-byte identity vs d5e04fdce; (5) A4 pin with
gate evidence -> Gate-A reviewer pass on the full Stage-A range
d5e04fdce..fixed tip. Sparks re-dispatched (a3b diagnostic + A3
binding run superseded/skipped); WSL leg dispatches with the fixup
SHA. Main routes the A3 pin now.

### A3 REVIEWER INTERIM (12:59Z): R1+R2 identity gaps — pre-staged into the combined fixup

Reviewer @ 17789eb05 (Main-confirmed): R1 = division-by-zero class
(baseManifest derives strides from the sealed layout WITHOUT section
checks -> compact-labelled 0/0; requireCanonicalCpuRoot accepts the
matching zero-stride root; sealCpuWire :631 divides by
cpu_block_stride/64==0 before seal/valid). FIX: baseManifest derives
PreparedUnitSections once + throws on !isValid() BEFORE strides;
sealCpuWire rejects <64/not-aligned as guard; regression = both
CPU-root builders, zero-stride root rejected before arithmetic.
R2 = identity not propagated (remote manifest valid() :492-496
re-derives from ENCODING-ONLY canonical overloads, never consults
identity.cpu_endpoint_layout -> compact-labelled canonical-stride
manifest is VALID; the test even asserts it ~:546; remoteCpu-
ProjectionStorageLayout :834-842 builds without cpu_layout ->
valid CANONICAL storage layout). FIX: valid() uses layout-aware
sections from the sealed identity (valid + strides equal); storage
layout passes .cpu_layout; test flips to expect INVALID; grep
remaining encoding-only stride-overload uses outside tables/
pricing scalars, LIST same-class validators in the commit. HOLD:
reviewer still finishing A3 — ONE combined fixup (F1+F2+R1+R2+
final items) on 569de492b when the verdict lands; Stage-A gate runs
once on that tip. A2 2-cell rerun stands; A3 pin routed on source.
