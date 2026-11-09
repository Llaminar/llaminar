# Native ARM CPU inference: implementation, numerical proof and tuning plan

Created: 2026-11-04; frontier updated 2026-11-08.
Base: `feat/qfn-ngram-continuation` at `b65ee52fd`. Working branch
`codex/qfn-ngram-continuation-arm64` (draft PR #32), force-pushed after every
commit; `arm-validation/<sha9>` branches pin each commit for the Spark hosts.

**Status: implementation complete for the ARM CPU scope; post-merge performance work is tracked separately on the `codex/arm-perf-review` branch.**
The objective remains native CPU inference with the robustness of the Intel
path, finished by a Release `Perf__` scaling round. Detailed historical
evidence (K-series snapshots, seals, falsification controls) lives in
[the evidence record](ARM_CPU_KERNEL_IMPLEMENTATION_EVIDENCE.md).

### Current frontier — 2026-11-08

Installed and verified:

- **Kernels.** Native NEON/NEON-DotProd leaves for every former AVX2/AVX512
  surface: NativeQuantised M=1/grouped/prefill/fused for all 21 source formats,
  attention (FP32, AQ8, Q8_1, TQ4/TQ8, Q16, sparse/QSA), RoPE, SwiGLU,
  softmax, RMSNorm (FP32/Q16/INT32), GDN, short-conv, Q8 requantize, sampler
  top-k, immutable lookup, Qwen4Exp HC residual read.
- **Batch invariance.** Serial and grouped rows are byte-identical on every
  ISA: explicit `std::fma` where GCC contraction differed per call site
  (skinny floating projections, short-conv decode, NativeQuantised BLAS epilogue
  `fma(alpha, product, round(beta*prior))`).
- **GB10 runtime.** Joint integrated-GPU + host pool in PMA; dense socket
  indices; physical cores by thread-sibling sets (tooling and runtime);
  launched processes keep the OpenMP-admitted CPU set (a non-ARM defect that
  ran whole teams on one core); cluster workers bound fastest core first;
  native projection tiles claimed dynamically (byte-safe).
- **Exact overlays (track K).** Resolver over the two numerically neutral NEON
  schedule dimensions (row tile, chunks per task) with a compile-time validated
  exact table; collector `Perf__CPUNativeNEONExactOverlay`; generator
  `native_quantised_dispatch/neon_exact_overlay.py` (exact-only, alias-pooled,
  replicated, no-regression, fail-closed). The 2,519-key GB10 table installed
  in `144ece037` was byte-neutral but an end-to-end regression (head A/B, seal
  `K14-OVERLAY-AB-HEAD-20261108`: removing it raised 35B decode 4.2% with MTP
  and 2.0% without, prefill 9-11%, 27B unchanged, text byte-identical), so it
  is retired. Microbenchmark-selected keys did not transfer to in-model
  execution. The installed table is empty; every key resolves to the total
  policy. Re-collecting overlays from in-model context is a follow-up.
- **CPU serving robustness (GB10, Release).** Real plan/benchmark/serve works
  for Qwen3-0.6B, Qwen2.5-0.5B (q4_0, q4_k_m), Qwen3.6-27B dense and
  Qwen3.6-35B-A3B MoE, including the MTP (nextn) GGUFs, MTP off and dynamic,
  HTTP chat with RAM-tier prefix reuse, and byte-identical greedy text between
  MTP off and on. Defects found and fixed on the way:
  - planner `d_ff` read the MoE router's expert axis (`21e411c7d`);
  - CPU nextn MoE experts are registry-owned, not raw parents (`0d12e65dd`,
    `ba8b32090`);
  - a decode-first (prefix-hit warmup) serial workspace family omitted the
    prefill rows (`878a33fea` lineage, `V2_Integration_DecodeFirstSerialWorkspaceFamily`);
  - libgomp's default 300k idle spin made every short region pay a team
    wake-up; the launcher now installs a bounded `GOMP_SPINCOUNT=3000000`
    (`V2_Integration_OpenMPIdleSpinPolicy`). 35B-MTP dynamic 13.23 -> 18.73
    tok/s, MTP off 13.68 -> 14.34, 27B-MTP 3.56 -> 4.23; an idle server parks
    all workers.
- **CPU-only gate hygiene.** Registrations whose subject only exists with a
  compiled CUDA/ROCm family declare `REQUIRES_BACKEND` (or the executable's
  `V2_TEST_REQUIRED_BACKEND`); CPU-only builds omit and count them (291).
- **Tests.** ISA-forced cells register `neon`/`neon-dotprod` on ARM; x86-only
  scheduler tests are x86-gated with ISA-neutral byte sweeps kept; portable
  all-format thread-scaling `Perf__` with byte authentication and measured
  grouped/serial amortization. Native macOS portability suites pass 259/259
  (plain and dotprod) on the M1 Max.
- **Pooled CPU MoE bundle (`408ff90e5`).** The router/gate-up bundle runs all
  members' tasks in one dynamic workshare (byte-identical). On GB10 35B it cut
  the bundle from 668 to 176 us per call (routed experts 803 to 274 us, which
  also includes the bounded spin policy).
- **GB10 integrated GPU (sm_121).** The branch builds spill-free at sm_121, at
  120;121 and at the complete 80;86;89;90;120;121 set, with the fatal spill guard
  active. The remaining 12.x spills were fixed without register-array demotion:
  - NativeQuantised prefill and flash attention are persistent TU shards. nvcc
    `--split-compile` gave nondeterministic register allocation and is now
    forbidden by `V2_Integration_DeterministicGPUCompilation`.
  - Tensor prefill, BK256 and the HD256 one-warp context oracle have 12.x-only
    scheduling fixes.
  GB10 runtime gates for that slice are tracked as idea 21. Logical plan lines now keep the integrated GPU's pool observation
  (`98e427c7c`, `V2_Integration_IntegratedGPUPoolObservationMerge`); before,
  every GB10 CUDA start failed `validateMemoryPlan`. 35B CUDA+CPU runs at 132
  tok/s prefill and 97 tok/s decode, and an over-subscribed authored tier is
  rejected at planning ("Rank host pool over-subscribed ... incremental
  40.4 GB, pool 39.5 GB"). Three fresh CUDA+CPU server processes return
  byte-identical greedy text (seal `K14-UMA-MERGED-BEF53EDB6-FINAL-20261108`);
  CUDA+CPU and CPU-only text legitimately differ (different accumulation
  programs).

GB10 serving reference (Release, 20 cores, -c 8192, greedy, n=256, per-run MTP
mode asserted; seals `K14-ABC-*`, `K14-OVERLAY-AB-HEAD-20261108`):

| Model | Commit | MTP dynamic decode | MTP off decode | Prefill |
|---|---|---|---|---|
| Qwen3.6-35B-A3B IQ3_S | `878a33fea` baseline | 18.18 +/- 0.40 | 14.14 | 24-25 |
| Qwen3.6-35B-A3B IQ3_S | `268cd81de` + bundle (+ overlays) | 27.19 +/- 0.18 | 20.48 | 37 |
| Qwen3.6-35B-A3B IQ3_S | head (bundle, overlays retired) | 28.51 +/- 0.29 | 20.95 | 40-41 |
| Qwen3.6-27B Q4_K_S | head | 4.24 | 4.64 | 10.5 |
| Qwen3-0.6B Q8_0 | pre-spin policy | — | 93.5 | 330 |

Greedy text is byte-identical across all A/B/C commits, with and without
overlays, and between MTP off and on. Qwen3-0.6B greedy text is byte-identical
between GB10 (NEON dotprod) and x86 (AVX512 and AVX2), seal
`K14-CROSSARCH-PROBE-GB10-20261107`.

GB10 topology: two clusters of 5 X925 (CPU 5-9, 15-19) + 5 A725 (CPU 0-4,
10-14); an X925 is about 2.8x an A725 per thread on these kernels. Dynamic
tiles over all 20 cores with `OMP_PLACES=cores` is the measured best policy.
The final round (`K14-PERF-FINAL-264970E14-20261108`) shows the 10 X925 cores
alone match 20 mixed cores for M=1 decode (217.7 vs 226.4 us), mixed wins
11-12% for grouped M=15, and A725-only is 2.4-3.7x behind X925-only.

Final Release `Perf__` round at `264970e14` (GB10):

- Thread scaling, all formats, mixed 1..20 / X925-only 1..10 / A725-only 1..10:
  pass, every timed row byte-authenticated against the serial oracle.
- Fused verifier, all formats, 20 threads: 45/45.
- Other CPU kernel owners at 20 and 1 threads: pass except two economy
  findings handed to the performance follow-up (FA2 tile tournament p95
  regret 8.9% over a 5% bound; GDN grouped verifier below serial for nearly
  every M). The NativeQuantised sweep harness lacked an admitted workspace (fixed,
  `5f3493f38`); the dispatch sweep is x86-only (`eeb392561`).

Cross-ISA and host gates (failures classified, no unexplained failure):

- x86 CPU-only Integration (Ryzen 7800X3D, WSL2): byte-exact suites pass on
  AVX512 and AVX2; one branch defect found and fixed (`05930d91a`).
- M1 Max Docker VM: Unit 938/958, preflight 2192/2250; every failure is the
  NUMA-less host class, a known upstream registration, a CUDA-device
  surrogate, or VM slowness of subprocess-heavy suites.
- GB10 CPU-only Integration gate at `a8d380cd0` (seal
  `K14-GATES-A8D380CD0-20261108`): Unit 951/958, preflight 2250/2252; every
  failure is a known upstream target-selection registration (3) or a
  CUDA-device profiler surrogate (6).

Performance follow-ups after merge are tracked on the separate
`codex/arm-perf-review` branch: I8MM for grouped rows,
27B dense MTP and its depth controller, region/barrier count per step, LM head,
GDN verifier economics, FA2 policy regret, placement by core class, in-model
overlay collection, GB10 CUDA register pressure, NUMA-less hosts.

## 1. Mandate and completion criteria

The code owner's scope is authoritative for this work:

> Create and tune a set of CPU inference kernels for ARM, including all current
> AVX2/AVX512 surfaces: attention, RoPE/SwiGLU, and quantized NativeQuantised
> GEMV/GEMM; preserve MTP batch invariance under the Llaminar numerical contract.
> Scope robust inference and produce a comprehensive plan, finalized with a
> `Perf__` testing round that proves scaling. Exact NativeQuantised overlays suffice;
> the full generic prefill/decode retraining pipeline is not required.

Deliverables:

1. Native AArch64 implementations of every admitted CPU operation and tensor
   format, wired through the existing CPU factories, graphs, preparation and
   workspace contracts. Small bounded scalar metadata/tails are legitimate;
   scalar whole-operation substitution and x86 intrinsic emulation are not.
2. Independent numerical/byte oracles, contiguous runtime-M coverage, ordered
   state publication and worker-count coverage on the actual optimized paths.
3. A complete native host runtime for the Mac, not libnuma/ELF/MPI stubs or
   `--no-mpi-bootstrap` as an ordinary serving mode. Linux ARM64 with CUDA13,
   real OpenMPI, two GB10/SM121 hosts, ordinary formats plus MXFP4 E8M0 and
   cross-host inference is also an explicit target; no Mac test proves that OS,
   hardware, sealed ABI or cluster surface. See the dated Spark handoff below.
4. Native total dispatch plus authenticated, measured ARM exact overlays.
   No generic ML fitting/retraining campaign; no reuse of x86 measurements as
   ARM evidence; no claim of a newly trained generic certificate.
5. A final Release `Perf__` round covering kernels and complete operations,
   one worker through physical-core capacity, and matched production inference.
   Functional correctness and economy must both pass before support is claimed.
6. Updated help/build/test/provenance documentation, focused production-defect
   preflight entries, preserved unaffected x86 programs and explicit regressions
   for repaired defects, and no committed local corpora or
   profiler/result payloads.

Not included: Metal/GPU execution on Apple hardware, SVE/SME/I8MM kernels
without matching hardware, ARM image/release certification by inference from
an x86 image, or a full new generic-policy training transaction. Their absence
must be explicit; none can be silently selected as a compatibility route.

## 2. Observed starting point

### 2.1 This host

Observed through `uname`, `sysctl` and installed compilers, not inferred from
processor marketing names:

| Fact | Observation |
|---|---|
| Hardware / OS | Apple M1 Max; macOS 27.0.1, Darwin 27; AArch64 |
| Memory / workers | 64 GiB; 10 physical/logical cores; 8 P + 2 E; no SMT |
| SIMD | DotProd=1, FP16=1; I8MM=0, BF16=0; reported SME families=0 |
| Cache line | 128 bytes |
| P-level cache observation | 128 KiB L1D; 12 MiB L2 shared by four CPUs |
| E-level cache observation | 64 KiB L1D; 4 MiB L2 shared by two CPUs |
| Compilers already exercised | AppleClang 21.0.0; LLVM 22.1.8 |
| Profiler evidence | Isolated native AQ8 encoder CPU Counters recordings succeeded; named IPC/cache and hot-phase qualification remain open |

The cache observations are **sharing-domain facts**, not private per-core L2
capacities to multiply by ten. Isolated recordings now exist, but raw counter
arrays without authenticated event identities/hot-phase attribution do not
qualify candidate IPC/cache features. Eight requested workers are not proof of
P-core-only execution. Public Darwin QoS/affinity mechanisms do not provide Linux-style
arbitrary hard CPU-ID pinning; record advisory placement and actual evidence.

### 2.2 Existing branch work

- Target-aware CMake ISA selection and Mach-O OpenMP install-name binding.
- Native NEON FP32 vector dot/four-row dot, AXPY, scale and add.
- Native capability/ISA admission, explicitly selected ASIMD/SDOT numerical
  leaves, source-order block decoding and checked resident prepared read views.
- All-six-encoding projection microtiles for 1..4 physical rows, covering all 21
  source identities against serial bytes and independent source-order oracles.
  Canonical native entrypoint/owner/workspace wiring now exists, but actual
  Tensor factory construction and complete engine/model execution remain unproved.
- Shared native Pair/Four-row worksharing covers original request slices,
  M=1..16/31/33/65, both numerical policies and actual 1..10-worker budgets.
  Canonical serial/grouped/prefill/FP32-beta/fused/two-term entrypoints execute
  over real retained small owners; HC activation has an independent numerical
  oracle. Actual tensor preparation/factory/model graph gates remain open.
- Native reduction-storage/ISA identity now prices the Q8-only ARM workspace
  through the canonical BOM and is retained in MPI/service provenance, without
  fabricated caches or root-local remote ISA guesses.
- Production SwiGLU primitives now execute native FP32/BF16/FP16/Q8 and exact
  heterogeneous expert arithmetic, with original-row/tail and typed retained-team
  byte proofs. An independent accuracy failure was fixed with split-ln2
  exponential/stable sigmoid, not a relaxed gate. Portable half narrowing is
  native; partial aliases and malformed typed extents reject before mutation.
- The fused SwiGLU/Q8 publication-to-grouped-down seam executes in one retained
  team for every source and both policies, covering every M1..16/31/33 and
  K32/64/96/1056 against actual serial production calls. Shared publication
  policy/source-alias checks are registered on every CPU ISA.
- Native RoPE covers floating pairs/inline fragments, in-place Q8/Q16 block
  cycles, normalized/legacy/fixed Q8 conversion, fixed/dynamic Q16 and streaming
  hybrid FP32 heads. Independent widened-integer and memory-rounded tests retain
  each distinct numerical program, exact-reuse metadata ordering and single/odd
  geometry without head mirrors. Cached team replay, coefficient overflow and a
  signed-byte/ratio endpoint have executed focused regressions. Equivalent x86
  geometry/widening fixes, full stage/arena ownership and economy remain open.
- Anchored AQ8 key encode/decode now executes native exact rational searches
  and separate-rounded reconstruction. All widths, boundaries, exponents, legal
  codes, zero residuals, immutable-anchor aliases and actual workers are covered
  by the shared production codec suite. Paired native encoder/decoder tuning and
  an isolated 1..10-worker `Perf__` sweep provide bounded kernel evidence only;
  full attention/cache lifecycle and qualified IPC/cache features remain open.
- Compressed TQ4/TQ8 score/numerator math now has native exact centroid lookup
  and ordered FMA programs over checked synchronous original borrows. All source
  codes/bitplanes, radii/thresholds, geometry/aliases and original-head worker
  grouping have independent proofs. This is correctness-only leaf evidence;
  full FA2/KV binding and resource/economy remain open.
- Native forward/transposed TQ rotations now reuse production grouped-dot/AXPY
  programs over checked original matrix/vector/result borrows. All row/K fringes,
  aliases (including unused owned matrix capacity), general dimensions and
  original-head worker grouping are covered. Cold verification now fails closed
  on nonfinite observations; an original-source NaN false-certificate negative
  and focused preflight reproduce the repaired invariant.
  Four unavailable-ISA substitutes were retired; x86 partial-vector loops were
  bounded without changing aligned arithmetic. Cold-generation bytes match the
  original program at twelve same-toolchain cases, not across standard libraries.
  Matrix-content identity/publication, actual quantize/cache binding and economy
  remain open; do not infer physical matrix equality from seed alone.
- Native four-storage softmax now binds checked original rows/batches, uses a
  fixed FP64 summary, accurate SIMD exponential including subnormals, and bounded
  in-place publication. Independent program bytes and double probability gates
  cover geometry/prefix/storage/worker fringes. Q8 FP16 schema, signed/-128
  extrema, representable minimum output scale, polynomial linear stage,
  coordinate narrowing, local F16C alignment and retained-team/empty completion
  defects have focused preflights and bounded original-source witnesses.
  Public kernel copy/factory boundaries, legacy unavailable-ISA/BF16-scalar/F16C
  cleanup and resource/economy remain open; this is not a full FA2 certificate.
- Original-cache native attention math now covers one/two/four independent
  FP32/FP16/BF16/I16 cache rows, shared-query two-chain dots, ordered shared-output
  FMA accumulation, widened integer dots and the original query quantizer.
  Actual M1 and independent references cover complete fringes and admission.
  Partial production-class dispatch wiring is in progress, not certified FA2.
- Actual FP32 CPUFlash C++23 instantiation exposed and now focused-regresses
  incomplete-type MPI constructor cleanup and aligned-storage include ownership.
  Borrowed constructors moved out of line; no MPI protocol changed. The real
  attention class now parses as actual FP32 C++23 after converting four orphan
  timing reductions, retaining small active teams, using native prefetch and
  shared Q16 scaling. Native FA2 compiler/runtime identities now map explicitly;
  the actual class parses warning-free under C++23 and `-Werror=switch`.
  Its initial canonical256-row span consumes neither x86 timings nor invented
  private-cache capacity; checked byte/integer geometry has focused preflights.
  Helper/KV/factory/runtime admission remains unfinished. Later class edits are
  outside the frozen Spark certificate; syntax/leaf/scope checks do not qualify
  their runtime. All four softmax and both rotation entries resolve.
- Original AQ8 scores now use checked immutable query/anchor/native-ISA bindings
  and register-only decoded fragments instead of `key[D]`. All head widths and
  query tiles1..4 retain their declared component/M1 words; independent RN and
  actual non-power-of-two publication controls protect both decode boundaries.
  Passing only pointer roots by value removes initial hot descriptor stack loads.
  A separate explicit native Q8 V-accumulation leaf replaces the ARM/AVX2 plain
  loop while retaining the rounded combined-weight/FMA program. This common leaf
  also changes the symmetric Q8 caller; full-core coverage must retain it.
  Source review found AQ8 entries/include still hidden in x86-only guards. Only
  AQ8 was separated; genuine GNU14 preprocessing proves the actual adapter is
  present and production class/regression-TU syntax passes. Its new full-core
  preflight uses real ring/workspace/tensor/verifier/two-request entrypoints with
  NaN-poisoned outputs, but has not linked/run. Query widths2–4 remain leaf-only,
  and the focused fixture is not PMA admission, wrap/context-summary lifecycle
  or a complete FA2 certificate. A cold-bound one-candidate dot `Perf__` now
  retains raw samples and rejects no-op authentication; bounded register-reuse
  timing is not production MTP/FA2/scaling qualification. See the evidence record.
- The latest completed five-Mac focused worktree matrix passes185/185 in baseline/DotProd
  Release, explicitly optimized Integration DotProd ASan/UBSan, strict-warning
  Debug and LLVM Release; all142 configured preflights pass20 repetitions
  (2840 independently counted raw passes), including native
  reduction/production-TU RMSNorm ownership/bytes/portable half, original-Q16
  geometry/expansion/ISA-specific reductions and actual legacy callers, original
  AQ8 query-dot tiles, explicit Q8 value FMAs, nonempty/dirty-output benchmark
  authentication, actual immutable-lookup row decoding, real CMake initializer
  lifecycle and independently compiled C/C++ configuration witnesses, configure-host
  topology/receipt, real CTest retirement metadata and image-fixture contracts. Actual XML/raw records were independently parsed without fail/skip.
  The earlier151/115,157/119,159/121,167/128, published170/130 and176/136 checkpoints
  retain narrower scopes. Inspection exposed empty Integration caches in the
  first focused GNU14 lookup run and old sanitizer configuration; those specific
  passes remain unoptimized diagnostics. The failed stale-cache admission and explicit optimized successor
  are separate evidence. The185/142 worktree is not a whole-engine certificate.
  The intervening136-test sanitizer
  projection-batch timeout remains RED. Its unchanged hashed binary passes20
  isolated-window repetitions, then the complete current matrix passes; no
  timeout/worker budget was changed and contention remains only a hypothesis.
  All17 dependency lifecycle/serialization cases and six genuine-cache decoder
  cases are covered. Historical native Linux/arm64 GNU13.3 and GNU14.2 baseline/DotProd
  matrices pass126/126 each; the prior93-entry inventory passed20 repetitions
  in Linux GNU13 baseline. The immutable Spark successor separately passes125/125×4.
  These are distinct source scopes,
  not certification of newer changes from historical evidence. Immutable projection fixtures/serial oracles now
  reuse storage across every worker budget, avoiding sanitizer timeouts without
  raising the ordinary timeout or reducing coverage. See the evidence record
  for exact scope, empty-team/geometry defects and negative regression proof.
- Shared AVX2/AVX512 compact templates pass syntax instantiation. Separate real
  WSL hardware now passes the repaired snapshot's complete focused AVX2/AVX512
  projects80/80 and82/82, with55preflights each and all four typed cases20/20.
  This is native hardware evidence, not Rosetta or a full Intel engine gate.
- The complete refresh Python suite currently has 63/118 failures at Linux-only
  shell-driver `lscpu` discovery; native tooling support is required, not skips.
- Full Darwin `src/v2` configuration still rejects the unimplemented Linux-only
  memory/placement backend; libc++ atomic-shared-pointer closure also remains.
  A genuine separate Linux ARM GNU14/OpenMPI4.1.6/NUMA/OpenSSL SDK closes C++23
  prerequisites and now actually builds/installs canonical oneDNN. A genuine
  GMock package closes the next SDK gap; full engine configuration/generation
  passes. Actual compilation then exposed OpenMPI ARM C-only internal atomics
  in its C++ component: a real C11 adapter now owns ORTE structures, while C++17
  retains the single typed environment/readiness authority. Component/helper/probe
  build and four actual native Linux preflights pass on explicit native POSIX
  session storage, without SDK macros, atomic emulation or weaker ownership.
  No dependency was disabled or replaced. Resumable installation/manifest,
  exact consumer library, compiler ARG1 and diagnostics have focused regressions;
  actual whole core/factory/runtime and complete gates remain open.
  No model or final `Perf__` gate has run.

The initial evidence and full-build command are retained in
[the first-slice handoff](2026-11-04-native-arm-port.md). Subsequent native
numerical building blocks and their current checks are recorded separately in
[implementation evidence](ARM_CPU_KERNEL_IMPLEMENTATION_EVIDENCE.md).
Re-run tests after changes; historical passes are not current-head certificates.

### 2.3 Linux ARM64 / Spark validation

The owner explicitly requested a two-host Linux ARM64+CUDA13/OpenMPI campaign,
including ordinary GGUF and MXFP4 followed by cross-host inference. The
[dated Spark handoff](2026-11-05-linux-arm-spark-handoff.md) records immutable
source hashes, exact focused build commands, source ownership, observed20-core
heterogeneous CPU/cache geometry, missing g++13.3 `<print>` support, the
codebook25/policy4/17B E8M0 ABI, SM121 qualification gap and shared-UMA admission
requirements. Initial authorization is frozen focused CPU testing only; no
whole-engine/model/CUDA readiness or real-weight access is inferred.

The first frozen123-file snapshot is **RED**:112/117 pass on each of both
hosts×both ISA profiles, with the same five SwiGLU/typed-tensor failures and
isolated3/3 reproductions. Native Linux GCC13.3 reproduction on the M1 pins
expert mismatches to five undesired fused Newton steps. Independent volatile
references also expose typed decode/add contraction, including cases where
native and scalar diagnostic agree on the same wrong word. Register rounding
edges now preserve the declared program in native and scalar routes; no
threshold, inverse, ties-away policy, tolerance or global compiler flag changes.
New independent regressions fail the frozen production code and pass locally;
all125 local focused tests and93×20 preflights pass as scoped above. A new
immutable successor (`SHA256SUMS` `391c8481…6128`) now passes125/125 on each
of both Sparks×both profiles; all five requested strict cases pass20/20.
This is focused CPU correctness, not whole-host scaling or engine qualification.

The supplemental real WSL Ryzen7800X3D/GCC14.2 typed target builds under AVX2
and AVX512, but its native test fails73/166 assertions and scalar passes, each
reproduced3/3. The new all-ISA oracle incorrectly assumed ties-away for the
existing AVX512 fixed32 Q16 nearest-even program. Preserve those explicit
per-ISA programs and independently diagnose any unretained decode-product
edges; do not converge arithmetic blindly or relax the byte gate. A native
read-only witness now shows73 remaining mismatches on each ISA against its own
publication program, all matching FusedA with production-wrapper FMA attribution.
`X86FP32.h` and the three fixed32 add routines now retain separate products;
manual per-ISA references and operand-order/exact-reuse regressions preserve
rounding policies. Immutable `eb3120f8…69a37` now passes native AVX2/AVX512
focused projects80/80 and82/82 plus all four typed cases20/20. New tests against
unchanged old headers fail73 native and3 alias assertions on each ISA; scalar
and signed-rounding controls pass. The unchanged independent probe reports
zero declared-program mismatches on both ISAs. AVX512's95 generic-away
mismatches correctly preserve its different nearest-even program. Named wrapper
inspection finds zero accidental FMA sites with no new stack/frame growth in
that bounded witness; no throughput certificate is inferred. Broader x86,
production attention integration and final economy remain open.

Official Ubuntu ARM GNU14.2 packages now provide a genuine locally compiled/
linked/executed C++23 print and atomic-shared-pointer closure, without a fake
header or changing the old SDK. The full engine must still actually build and
pass its complete applicable gates before a small real GGUF launch; focused
kernels are not a runnable server.

The actual full Linux dependency attempt exposed compiler-memory exhaustion:
individual oneDNN reorder TUs consume about2.1GiB, while the VM has8GiB.
Canonical optimization/ISA flags and unrestricted Ninja parallelism remain.
Isolated exact-TU resource diagnoses are not full-build passes. The old CMake
also erased partial work without an install stamp and discarded Ninja stdout;
`cmake/OneDNNBuild.cmake` now separates configured and installed ownership,
retains failed-build work, binds toolchain identity and reports both streams.
An empty origin observation previously triggered destructive source re-cloning
without retaining Git status/stderr. Existing-source authentication now rejects
without deletion; the underlying old probe error is not retrospectively known.
Seventeen real CMake/Ninja/Git device-free cases exercise these states, missing
libraries/other headers, modified artifacts and mandatory compiler arguments,
without a compiler substitute or an SDK implementation. Review also exposed
legacy fabricated x86 cache defaults: the new genuine CPUID/topology record now
keeps unknown/shared caches unknown, and both FA2 and inventory consume it.
Retirement intent is durable across missing receipts and partial failure. New
two-call and interrupted-retirement tests expose actual outside writes/bad
same-mtime bytes in the unchanged staged662 owner; its build authorization was
withheld and it remains immutable. The current full matrix passes after repair.
Full engine compilation next exhausted the8GiB VM on two real core TUs. The
fresh immutable `3bf350a5…a80ba` packet received a one-Spark CPU-only build window
and exact SDK authentication. Independently checked raw records show genuine
oneDNN installation, then a first configuration failure from unchecked multiline
`lscpu` fields at the unchanged finalizer; application/gate compilation never
began. All100 raw artifact hashes and original affinity/sysfs/compiler/cgroup
scope were audited. The repaired configure-host observer now uses genuine
inherited affinity and kernel sibling groups, preserving original IDs and
explicit asymmetric vectors. Existing shared budgets reject unsupported
nonuniform projections instead of inventing capacity. Receipt retirement runs
before every producer/projection and successful publication is atomic; nine
real Python/CMake cases and early/late gate-target inventory pass on Mac/native
Linux, with separate recorded20-core Spark and live10-core VM observations.
These are not live inference/rank topology certificates.

Actual native MPI-bound registration also exposed old SDK CTest3.28 JSON omitting
required timeout-owner fields despite runtime property support. A genuine
hash-pinned4.0.3 SDK successor exports SIGTERM/36 and passes the unchanged real
helper's19+4+130 source/Unit-metadata cases. Current native9 topology+3 tool+243
production-pipeline Unit cases pass with read-only source and lexical fixture
storage; selected14 image/fixture cases pass on Mac, whose broader pipeline
path-fixture failures remain separately RED. The common installer/test-runner
retain exact native/package tool closure; the independent latest-CMake producer
and fixed ordinary-build parallelism are retired. The original compiler-policy
envelope remains, without weakening missing-response codegen authentication.

The new SDK is explicitly archive`4e8a5adc…8e434`/index`d211c0e0…68bf0`; all9
encoded/rootfs layers retain the exact old7-layer base. This is not CMake4 vendor
or full-image certification. No old3bf/662 retry, worker cap, source patch or
silent SDK substitution is authorized. Export a new source successor, bind this
new SDK explicitly and obtain fresh lab/capacity admission for one BUILD-ONLY
attempt. Whole engine gates remain open.

That new immutablec76/e981 attempt now has independently SHA-checked returned
raw evidence: actual20-core/1-package/1-thread configure/generate passes97.265s,
and the complete ordinary app/gate target command reaches real compilation.
It stops90.352s later at an unused x86-only header in KVCacheAppendStage.cpp,
without OOM. Main removes the unused include and adds explicit architecture
preflight; actual current GNU14 stage syntax passes with the exact captured
production command. No old packet was patched/retried and no complete target,
Unit or serving pass is inferred. A new full-build successor is still required.

The separately admitted73da/5adf successor, exact SDK4e8a/d211/a78, now configures
in93.262s and stops its normal app+complete gate-target build after222.436s at
`CPUImmutableLookupKernels.cpp:17` (`immintrin.h`). Main verifies return archive
984a29…c863/all196 raw hashes, actual2138 parent commands/3430 SHOW-ONLY cells/
3129 genuine timeout-owner pairs, unchanged7063-entry immediate maps and37.56GB
peak/OOM0. No tests or app exist; its one build authorization is consumed and
owner window released. Prior KV append now has an actual successful Ninja object
record, not an engine pass. This failure is immutable and must never be patched
or retried under another source claim.

The native lookup repair goes beyond its header: original FP16/BF16 widen in
NEON, IQ4 uses original register table lookups/products and bounded output, and
all formats retain original row coordinates. Partial IQ4 physical blocks no
longer flatten padding into another logical row or silently publish nothing;
AVX2/AVX512 receive matching bounded masks. Empty retained calls complete after
all geometry/alias checks, without a new team/post-release scan. Independent
payload/fringe/guard/native-ISA controls and all24 source/M1/team comparisons use
the actual production TU. Final optimized GNU14 runs8 cases and explicit
registrations/repeats; old-coordinate/empty policy countercontrols fail as
expected. Generic arithmetic of the other20 quantized formats is unchanged,
not newly certified SIMD/economy. Full service/PMA/core/engine and Release
performance still require a new sealed successor, fresh admission and real gates.

The focused project's actual compile metadata also exposed after-project empty
Integration caches. C/CXX defaults now initialize before language setup without
clobbering authored final or normal/cache initializer values. Stale empty caches
reject; a separately named optimized sanitizer successor retains the old RED.
Real CMake cache tests are SDK-free, and actual C/C++ executable witnesses prove
optimization/NDEBUG without runtime compilation. Exact O3/g and whole-TU policy
remain separate compile-command obligations. See the evidence record for the
initial noexec fixture and corrected authored-initializer regression history.

The RMSNorm frontier now has a bounded native production-TU slice. Actual
retained1×12288 all-ones execution reproduced per-caller partial ownership
(output1.7320248 instead of0.999995), plus repeated small in-place normalization
and empty early return. All three focused regressions are repaired and explicitly
registered in preflight. Complete row/tile transactions replace FP32 TLS banks
and atomic column partitions; unused detached FP32 stages/scratch overload were
retired. Native FP32/FP16/BF16 tiles share gamma over1..4 original rows, preserve
staged versus per-row inverse programs, and publish portable half tails without
a decoded row bank. Independent formulas and actual M=1 bytes cover M1..16/31/33
and workers1..10 in the new production-TU tests. Native BF16/FP16 factory source
now selects original-storage primitives and its actual C++23 TU parses cleanly.
The earlier151/115 checkpoint has separate current-source GNU14 native
RMSNorm8+3 XML passes. The implementation evidence retains each newer focused
matrix's exact source/count scope. None of these checks qualifies factory
runtime or full inference.

Original-Q16 now has a separate current-worktree native tile program: FP32
products/squares precede fixed FP64 chains and original FP32-mean inverse;
complete-block/expansion/gamma/options checks and retained empty completion have
three failing frozen-source regressions. Native GNU14 ARM15 production+9 actual
legacy+3 reduction cases and all170 Mac focused registrations/five profiles pass;
130 preflights pass20 repetitions. Actual production M1 bytes and independent
rounded-square programs cover original geometry/workers without a decoded bank.
Reference trees follow declared ISA/force-scalar, preserving eight-chain native
and sixteen-chain AVX512 arithmetic; a K128 RN fixture proves their different
means/inverses. Six legacy unconditional AVX512 callers were independently
reproduced failing before repair. Public native calls now preserve their original
accuracy obligations; the named entry separately proves implementation or exact
rejection, and rejects nonfinite output as evidence. The canonical Unit name and
explicit legacy/tree preflights use the actual core-free production TU.

The genuine AVX2 body is now executed natively, not merely syntax-qualified.
An exact-local/offline authenticated SDKf492 supplies real CMake/CTest4.0.3;
its failed BuildKit and ignored-empty-entrypoint predecessors remain RED.
Main independently checked the source73da/d18 WSL return archive/all178 raw
hashes:20 registered cells/ISA, eight focused cells×20=160 passes/ISA, then the
actual AVX512 binaries/runtime-AVX2 lane14 RMS+9 legacy GTests, no skips/failures.
Actual ISA/FP/compile-command/terminal evidence and complete7063-entry immediate
before/after maps agree. This closes only frozen73da's focused x86 correctness,
not later AQ8, full engine/gates, runtime/factory lifetimes or performance.
Four-row AppleClang/GNU14 static resources have no hot stack accesses/calls
(28/23 SIMD identities); this is not PMU, latency/scaling or final performance
qualification.

RMSNorm remains unfinished: residual and quantized factory/workspace routes,
Q8 epsilon/stale gamma/no-op/narrowing defects, INT32 AVX2 low-bit loss and
row-quantizer alias/scale programs require independent executed repairs and
complete gates. Q16 full factory/runtime/snapshot and actual capacity/lifetime
qualification remain open. Preserve distinct audited programs instead of
copying legacy outputs or accepting cosine-only metadata/code tests. Final
resource, latency, scaling and complete model/MTP campaigns still apply.

### 2.4 Audit coverage and limits

[The source-surface inventory](ARM_CPU_X86_SIMD_INVENTORY.md) enumerates all
85 files matched by the recorded x86-header/type/intrinsic/ISA-marker scan,
including execution stages and transport, not just `kernels/cpu/`.
This is a work inventory, not proof that all unguarded or transitive compiler
hazards are known. A target-native header/TU closure build and operation matrix
must close that gap. Generated x86 policies, OS APIs, tests and profiler tools
are additional dependencies even when the lexical scan does not match them.

## 3. Numerical and lifecycle contract

### 3.1 Distinguish the three equality obligations

| Obligation | Required oracle / acceptance |
|---|---|
| ARM grouped verifier vs ARM serial decode | Native-byte equality for every live output/state row, including signed zero where published; independent production M=1 calls are diagnostic oracles only |
| Placement-invariant expert computation | The existing CPU/CUDA/ROCm contract, including activation bytes, source-factor reconstruction, rounded block contributions and fixed reduction trees; an ARM-native FMA rewrite is not equivalent by algebra alone |
| Model accuracy / intended precision | Existing typed model-parity tolerances, stage CSV diagnostics and reviewed generation controls; cross-ISA ordinary non-expert token equality is not assumed solely from same source code |

Two identically incorrect ARM serial/grouped kernels do not certify accuracy.
Use source-format independent decoders and shared normative scalar arithmetic,
then the production M=1 oracle, then model/stage diagnostics. Token agreement
alone is insufficient; relaxed cosine/L2 cannot accept MTP byte drift.

Normative sources, rather than a copied numerical specification:

- [`MoEProjectionNumericalContract.h`](../../../../src/v2/kernels/common/MoEProjectionNumericalContract.h)
  binds `BackendNative` versus `GPUAlignedExpert`, 32-value Q8 blocks, the
  quantized expert partition tree and floating expert logical lanes.
- [`DeviceQ8ActivationNumericalContract.h`](../../../../src/v2/kernels/common/DeviceQ8ActivationNumericalContract.h),
  [`DeviceHalfMetadataContract.h`](../../../../src/v2/kernels/common/DeviceHalfMetadataContract.h),
  [`DeviceFP32NumericalContract.h`](../../../../src/v2/kernels/common/DeviceFP32NumericalContract.h)
  and [`PreparedHalfReciprocal.h`](../../../../src/v2/kernels/cpu/primitives/PreparedHalfReciprocal.h)
  own scale publication, RNE, reciprocal and separate rounding boundaries.
- [`CPUNativeQuantisedContributionContract.h`](../../../../src/v2/kernels/cpu/gemm/CPUNativeQuantisedContributionContract.h)
  currently supplies x86 vector equivalents; ARM needs native equivalents of
  the same program, not an unqualified FMA on every multiply/add pair.
- [`FloatingExpertNumericalContract.h`](../../../../src/v2/kernels/common/FloatingExpertNumericalContract.h)
  and [`DeviceSwiGLUNumericalContract.h`](../../../../src/v2/kernels/common/DeviceSwiGLUNumericalContract.h)
  own the floating-expert tree and nonlinear program.
- [`CPUFixedEpochGDN.h`](../../../../src/v2/kernels/cpu/gdn/CPUFixedEpochGDN.h)
  and [`CPUFixedEpochQueryBlock.h`](../../../../src/v2/kernels/cpu/gdn/CPUFixedEpochQueryBlock.h)
  own the fixed eight-position checkpoint and history program.

### 3.2 Preserve arithmetic identity under tuning

- SIMD across independent outputs/queries is safe when each lane retains its
  scalar chain. A horizontal reduction must reproduce the specified logical
  tree, not whichever `vaddvq` happens to be convenient.
- Freeze the serial arithmetic plan for a projection; grouped rows and exact
  overlays inherit it. Worker scheduling cannot change its K boundaries or
  fold order. Ordinary ARM serial policy must be explicit and complete; expert
  arithmetic already fixes its tree. Measure alternative arithmetic only as a
  separately identified, numerically qualified policy, never as a neutral tile.
- Quantized experts use up to 16 ordered partitions, bounded by real 32-value
  blocks; floating experts use 256 logical lanes and the descending tree.
  Their physical implementation may differ, their numerical program may not.
- CPU expert Q8 uses the shared half-rounded scale, deterministic reciprocal,
  RNE and saturation. Generic CPU tensor quantization is a different existing
  program. Keep both explicit; do not silently apply expert quantization to
  every tensor or replace expert publication with `x / scale`.
- Preserve Q6_K/IQ2_S/IQ3_S/IQ4_XS original factor tuples, half/subnormal
  behavior, asymmetric minima and sign/index maps. An INT8-equivalent payload
  does not authorize narrowing effective scales to half a second time.
- AArch64 compiler control must preserve separately rounded operations after
  inlining: test Clang/GCC pragmas/register dependencies and inspect optimized
  instructions. Avoid volatile scratch or global optimization disablement as
  production rounding machinery.
- Keep FPCR rounding/flush semantics explicit and test adversarial values.
  No process-wide precision change, approximate reciprocal/exponential or
  BF16/I8MM use may be hidden behind ISA detection.

### 3.3 Preserve ownership and original coordinates

All persistent weights remain owned by prepared storage; invocation scratch
remains supplied by `CPUInvocationWorkspace`, `CPUProjectionWorkspaceContract`
and `BufferArena` and charged through `PhysicalMemoryAuthority`.
No retained whole-matrix FP32/INT8 mirror or per-invocation repack/allocation.

[`ProjectionRowWork.h`](../../../../src/v2/kernels/common/ProjectionRowWork.h)
retains original request/frontier/live coordinates and physical strides. ARM
must handle full, partial, empty, ragged and large-to-small work without
flattening rows, rebinding buffers or touching inactive outputs. Exact overlay
lookup must use observable original physical/policy geometry consistently.

Stateful kernels must preserve checkpoint/acceptance/reset ordering, not only
the immediate output: KV append, wrapped rings, prefix restore, GDN/short-conv
speculation, rejection and continuation all remain production contracts.

## 4. Architecture of the native implementation

### 4.1 CPU stays one backend

Keep `DeviceType::CPU`, `KernelFactory`, stage interfaces and model graphs.
ISA is a typed implementation/capability of the CPU operation, not a new
placement backend or a model-specific orchestration script.

Extract architecture-neutral prepared descriptors, numerical policies and
work scheduling from intrinsic-heavy headers. Put x86 and ARM execution leaves
behind a small explicit operation boundary, with ISA-specific implementation
TUs/headers. Avoid an AVX-shaped global intrinsic compatibility layer and avoid
including the enormous x86 generated policy tables in ARM numeric TUs.

Suggested implementation seams (names to finalize in code, not existing APIs):

- An immutable CPU ISA/capability descriptor, separate from cache/OS placement.
- A prepared NativeQuantised block view over the existing descriptor and spans.
- ISA-native block dot and rounded contribution primitives.
- A complete projection launch plan carrying numerical identity, live work,
  geometry, purpose and exact workspace extent.
- Architecture-independent producer/ordered-reducer scheduling consuming those
  leaves. Distinct typed Decode/Prefill intent remains authoritative.

Reject unsupported ISA, encoding, codebook or workspace combinations before
publishing output/telemetry. Test unknown enum values and stale identities.

### 4.2 ISA envelopes

Retain AVX2/AVX512 numbers/behavior for existing artifacts. Add truthful ARM
compile/runtime identities rather than allowing `#else -> AVX2` metadata.
The current NEON foundation uses the mandatory AArch64 ASIMD envelope.

Provide native ASIMD widening multiply/add and a separately compiled DotProd
variant with equivalent integer results. Both must be complete implementations
with their own correctness and economy evidence if admitted. DotProd objects
must carry an explicit compiler target and runtime capability check; baseline
TUs must not accidentally contain SDOT. M1 Max is the first measured DotProd
host, not an excuse to assume it on every AArch64 CPU.

SVE, SME, I8MM and native BF16 arithmetic are not first-release requirements on
this machine. In particular, `BF16=0` does not remove BF16 tensor support:
widen the exact bit representation into FP32 and execute the established
FP32 accumulation/rounding policy. Hardware feature absence is not permission
to advertise fewer source tensor formats.

This M1 boundary is not a blanket GB10 deferral. On the primary Spark target,
verify actual native HWCAP/compiler admission, then evaluate an I8MM row-pair
candidate against ASIMD/SDOT for grouped M. Preserve the existing64×32 packing
where exact operand assembly permits it, prove scale-subgroup and integer-bound
semantics, odd rows/tails and the original per-column FP32 program, and require
native economy/resource evidence before installing any exact overlay. MAC count
alone does not establish a speedup. Keep X925/A725/mixed placement and every
positive worker count through20 distinct; weighted output-column ownership may
change scheduling but never the arithmetic K partition/fold. No I8MM kernel or
weighted production policy is currently qualified by the M1 leaf results.

### 4.3 Reuse the physical packed layout first

[`CPUNativeQuantisedPreparedFootprint.h`](../../../../src/v2/kernels/cpu/gemm/CPUNativeQuantisedPreparedFootprint.h)
owns six physical encodings and 64-output-column by 32-K-value units.
[`CPUNativeQuantisedWeightPacker.h`](../../../../src/v2/kernels/cpu/gemm/CPUNativeQuantisedWeightPacker.h)
interleaves four consecutive payload bytes for each output column.

A 128-bit NEON register can therefore consume four independent columns of
that existing stream. SDOT with a repeated four-byte activation word produces
four exact INT32 column partials. Keep the 64-column logical unit while
measuring physical column subtiles; a new packing ABI is not a prerequisite.

For the ordinary expanded signed-byte case:

```text
x86 integer result = sum((qa + 128) * qw) - 128 * sum(qw)
ARM SDOT result    = sum(qa * qw)
```

These are identical integer values within a 32-element block, whose worst
signed-byte product sum is bounded by 524,288. Prove the actual source ranges
and centering convention separately for each decoder; do not double-apply the
x86 compensation when using signed SDOT. Compact unsigned fields and Q6
bitplanes retain their existing format-specific centering corrections.
The baseline ASIMD implementation widens before multiplication/addition;
no saturating pair sum may discard information.

Keep `PackedWeightsFormat::CPU_NATIVE_QUANTISED` when the byte layout is unchanged.
An ISA change alone is not a reason for a second persistent pack or transfer
ledger. Retain source identity, encoding, scale/minimum metadata, numerical
policy and exact storage/clone/migration contracts. Only introduce a new ABI
if measured evidence requires it, with complete memory/transfer/planner updates.

## 5. Work packages and acceptance

The packages below are implementation work, not already satisfied by this
plan. They are sized to yield independently green correctness slices before
wide tuning. Every source defect gets a focused regression and explicit
`ProductionTestPreflight` registration, followed by its broader format sweep.

### A — ISA, cache topology and standalone kernel build boundary

**Files:** `CPUFeatures.h`, `SimdTraits.h`, `AVX2Helpers.h`,
`cmake/CPUInstructionSet.cmake`, CPU CMake source/flag lists, test ISA loops,
`CPUNativeQuantisedGemv.h` and generated-policy binding seams.

- Make ARM active ISA/name/dispatch available outside the x86 preprocessor
  branch. Forced incompatible routes fail; no scalar substitution for a
  missing native implementation.
- Introduce native vector load/store, selection, exact conversion, widening,
  dot, FMA and rounded-operation contracts. Keep operation-specific numerical
  reductions out of a misleading universal `reduce_sum` abstraction.
- Observe cache sharing and CPU feature identities natively on Darwin and
  Linux; no guessed 64-byte lines or per-core multiplication of shared L2.
- Allow core-free Unit/Integration/Perf kernel targets to compile the actual
  production leaves without MPI/NUMA/GPU initialization. Share registrations
  and sources with canonical gates, not a second engine or test kernel.
- Isolate fixed x86-only perf targets/flags as named architecture diagnostics;
  keep equivalent semantic ARM tests rather than dropping the operation.

**Exit:** native baseline/DotProd feature and header-closure tests, truthful
provenance, negative ISA admission tests and unchanged AVX2/AVX512 compilation.

### B — Conversion, tensor decode and activation publication

**Files:** the 31 tensor inventory candidates; `FP16Utils.h`,
`SIMDHelpers.h`, `QuantizationUtils.h`, `CPUImmutableLookupKernels.cpp`,
`CPUNativeQuantisedFP16.h`, `GPUAlignedExpertQ8Primitives.h`,
`CPUNativeQuantisedActivationSums.h`, `PreparedHalfReciprocal.h`.

- Exhaust all binary16 encodings for scale expansion, preserving special
  payload bits where the existing conversion contract requires it. Check BF16
  bit widening and RNE publication, signed zero, subnormals, extrema and ties.
- Port unpack/dequantize/quantize for every admitted tensor codebook and all
  source superblock arrangements. Vectorize lookup tables with NEON table/
  index operations, sign extension and shifts, without losing scale identity.
- Add native block max, half scale, multiply/clamp/RNE/narrow and exact signed
  sums for Q8; retain distinct backend-native and GPU-aligned policies.
- Honor partial blocks and original alignment/strides; avoid out-of-range
  vector loads. No whole-model expanded mirror to simplify later kernels.

**Exit:** independent all-format decoder/packer comparisons; native Q8 bytes
against normative shared oracles, including half-integer boundaries and tails;
ASan/UBSan; focused preparation throughput evidence.

### C — NativeQuantised M=1 for all six prepared encodings

**Files:** `CPUNativeQuantisedGemv.h`, `CPUNativeAVX2Gemv.h` (x86 isolation only),
`CPUNativeQuantisedQ6Kernels.h`, `CPUNativeQuantisedMultiScaleKernels.h`,
`CPUNativeQuantisedContributionContract.h`, packer/writer and footprint contracts.

Implement in dependency order:

| Prepared encoding | Initial native work |
|---|---|
| `ExpandedInt8` | Signed SDOT / widening dot, existing scales/minima, exact compensation equivalence |
| `NibbleLUT` | Low/high nibble decode and codebook table, bounded four-column streams |
| `NibbleLUTSourceScale` | IQ4_XS original half base and raw coefficient; not IQ4_NL metadata |
| `IQ3SExpandedSourceScale` | Existing decoded grid plus original source scale/nibble reconstruction |
| `Q6KNativeDualScale` | Six-bit bitplanes, original half base and signed low/high coefficients |
| `CompactMultiScale` | Q3_K/Q2_K/IQ2_S/IQ2_XS/IQ1_M native prepared payload, dual scales and any effective minima |

The registry, not this table, decides the exact source-format mapping. Q5,
Q8 source variants and all IQuant aliases remain explicit witnesses even when
preparation maps them to an existing physical encoding.

- Visit K blocks in the serial numerical plan's order; implement both numerical
  policies and their ordered partial reduction. Preserve alpha/beta and output
  accumulation semantics where exposed.
- Start with column tiles that fit registers, then tune prefetch, K cache tiles
  and physical width from actual prepared bytes/cache sharing.
- Retain all 21 source identities and 18 CPU runtime codebooks. Do not collapse
  source aliases that differ in original scale arithmetic merely because the
  codebook integer matches.

**Exit:** every source format/encoding agrees with independent source decode
and normative arithmetic; all N borders, legal K boundaries, forced native
variants and positive worker geometries covered. No hot-loop spills for
compute-bound candidates; M=1 bandwidth/economy evidence precedes group tuning.

### D — Grouped verifier, ordinary prefill and fused projection economics

**Files:** `CPUNativeQuantisedGemmKernel.h`, `CPUNativeQuantisedGemv.h`,
`CPUNativeQuantisedPrefillSchedule.h`, `CPUProjectionWorkspaceContract.*`,
`CPUQ8ResidualCorrection*.h`, `CPUInvocationWorkspace.h`, `ProjectionRowWork`.

- Build bounded native row tiles (initially 1/2/4; consider wider tiles only
  with register/economy proof), not one specialization per speculative depth.
  Reuse each decoded weight tile across row accumulators. Odd tails are native
  bounded tails, not replay of the complete request through M=1.
- Give grouped execution the serial plan's K boundaries, scale reconstruction,
  rounded contribution program and final fold. Parallelize independent
  row/column/ordered-partial work without changing arithmetic.
- Preserve Decode vs Prefill intent explicitly. Small ordinary prefill is not
  MTP just because M is small; a large verifier is not ordinary prefill.
- Implement fused gate/up, SwiGLU/down, QKV/multi-projection, heterogeneous
  projection bundles and router Q8 reuse on the actual prepared operations.
- Port primary/residual two-term Q8, reconstruction subtraction and final sum
  with the existing independent rounding points; do not disable the QFN
  precision policy to make an ARM comparison easier.
- Preserve original live-row publications, partial output strides, bounded
  scratch, zero-allocation warmed calls and actual worker task ownership.

**Exit:** byte equality for every M=2..16 plus M=31 and every larger admitted
contiguous capacity; ordinary-prefill M residue/bucket inventory; all formats,
row/column tails, mixed projection formats and positive threads; grouped work
counters and measured amortization, not token agreement alone.

### E — Native dense/floating GEMV/GEMM

**Files:** `FloatingPointGemmKernel.h`, factory adapters, floating expert and
QFN precision policy binding; oneDNN integration only where it already owns
an admitted operation.

- Implement native FP32 accumulation over FP32/FP16/BF16 weights and the
  existing selected half-operand policies. BF16 storage support does not depend
  on BF16 instructions on this host.
- Preserve the 256-logical-lane expert program in SIMD over independent lanes
  and output rows. Ordinary dense paths may use their explicit backend-native
  program, with grouped rows matching it exactly.
- A new Accelerate/BLAS call is not an automatic fix: variable batch-dependent
  reductions cannot replace a byte-invariant verifier or movable expert.
- Verify native oneDNN build/library/runtime closure and unsupported operation
  handling; do not let optional library discovery silently choose scalar code.

**Exit:** FP32/FP16/BF16 grouped byte gates, cross-policy tests, tiny/large K,
terminal heads, view/stripe preparation and measured dense/prefill economics.

### F — Normalization, nonlinear, residual, RoPE and direct stage fusion

**Files:** `RMSNormPrimitives*`, `CPURMSNormKernelT.cpp`,
`RoPEInline.h`, `RoPEPrimitives*`, `SoftmaxPrimitives*`,
`SwiGLUPrimitives.cpp`, `VectorPrimitives.cpp`,
`FusedResidualNormStage.cpp`, `GatedRMSNormStage.cpp`,
`AttentionOutputGateStage.cpp`, `KVCacheAppendStage.cpp`.

- Native vector residuals/gating, per-head and hidden-width RMSNorm, stable
  softmax/max/exp/sum and all ordinary/expert SwiGLU programs.
- RoPE full/partial dimensions, position sources, layouts, FP32/BF16/FP16 and
  integer Q8/Q16 publication. Preserve the original quantized packing/scales
  and no cross-row pairing of blocks at odd row widths.
- Port and prove nonlinear range reduction/polynomial/rounding where used;
  architecture vector width is not permission to call a different approximate
  exp/rsqrt and accept drift in grouped or cross-tier programs.
- Remove direct x86 assumptions from fused stages through reusable operation
  primitives, not model-local imperative code or decomposed serial stage loops.

**Exit:** existing CPU grouped namespaces on ARM with native route counters;
all native output bytes/aliases/tails; M=2..capacity extension beyond existing
small-M fixtures; `Perf__` width and team-crossover sweeps.

### G — Attention, native KV and TurboQuant

**Files:** `CPUFlashAttentionKernelT.h`, `CPUFlashAttentionAQ8.inl`,
`CPUQ16AttentionMath.h`, `CPUAttentionKeyQ8.h`, `CPUQwenSparseAttention.cpp`,
`TQFusedAttentionPrimitives.h`, `turboquant/*`, cache/append/prefix contracts.

- Native QK dots and weighted V accumulation directly over each admitted KV
  representation. Support all canonical FP32/FP16/BF16/Q8/Q16/TQ pairings,
  anchored AQ8 and mixed K/V formats admitted by the current runtime.
- Preserve fixed 256-KV-row summaries and ascending merges. Tune physical KV
  streaming tiles independently; do not change numerical summary size or
  causal visibility to improve a benchmark.
- Preserve Q16 normalization/integer bounds, anchored-key tie boundaries,
  sparse selection/ordering, GQA and participant-local TP head geometry.
- Port Hadamard/TurboQuant rotations and codebook selection without changing
  stored cache bytes. No dequantized full-cache mirror or prefix-cache disable.
- Exercise position-major/head-major inputs, paged/ring wrap, stale generations,
  partial histories, grouped causal rows, ragged request batches and prefix
  export/import/reuse. Stateful publication is part of the measured operation.

**Exit:** all-format serial/grouped bytes, canonical FA2 tile tournament and
causal worker-ownership proofs, long/wrapped/ragged cache lifecycle; attention
and TurboQuant `Perf__` scaling on the native optimized paths.

### H — GDN, short convolution and QFN-specific kernels

**Files:** `CPUFixedEpochGDN.h`, `CPUFixedEpochQueryBlock.h`,
`CPUGatedDeltaNet.cpp`, `CPUShortConvolution.cpp`,
`CPUQwen4ExpSIMD.h`, `CPUQwen4ExpKernels.cpp`, immutable lookup kernels.

- Add native independent-value/query SIMD traits to the existing fixed-epoch
  implementation. Retain backward raw-history contraction, increasing-key FMA
  order, separately rounded decay/beta and forward value updates.
- Do not replace the eight-position checkpoint program with a dense replay
  loop or introduce checkpoint boundaries at grouped/tile/thread transitions.
- Preserve short-conv history/state order and grouped projection policy; exact
  accepted-prefix publication must support every accepted depth and rejection.
- Preserve QFN immutable source/codebook reads, hyperconnection/gating,
  two-term activation correction and source-bound precision choices.

**Exit:** production-stage ordinary/grouped outputs and complete state bytes;
epoch-boundary, request reset and prefix-restore continuations; existing GDN
`Perf__` retained-replay and prefill measurements adapted to ARM.

### I — MoE/router/sampler and complete operation integration

**Files:** `CPUMoEKernel.cpp`, canonical CPU route reducers/records,
`CPUSamplerPrimitives.cpp`, CPU stage/kernel adapters and factory bindings.

- Preserve route tie/scan order, expert ownership, router-to-expert Q8 reuse,
  floating/quantized mixed gate/up/down and dynamic residency publications.
- Keep deterministic canonical route reduction and actual communication/live
  extents. ARM spin/yield and atomics retain acquire/release semantics.
- Native top-k/filtering must preserve deterministic token/tie ordering;
  stochastic sampling retains RNG/draw ownership and non-greedy MTP.
- No host shadow of accelerator state or x86-only archive identity hidden in
  multi-rank placement; cross-architecture expert state remains contract-bound.

**Exit:** real production router/expert transactions with required reuse/worker
counters; serial/grouped outputs, sample distributions and RNG/state tests;
cohort admission, movement and failure/reset lifecycle coverage.

### J — Native OS memory, topology, loader and MPI lifecycle

**Files:** `CPUBackend.cpp`, `HostMemoryCapacity.cpp`, `NUMAAllocator.cpp`,
`HardwareInventory.cpp`, `NUMATopology.cpp`, `ComputeBackend.cpp`,
`DeviceContext.cpp`, allocation/registration boundaries, `MmapRegion.h`,
`MPIBootstrap*`, `MPISessionOwner.cpp`, `NativeMPISessionSchizo.cpp`,
shared-memory/transport and loader preflight infrastructure.

This is a separate, required serving track; it must not block isolated native
kernel work, nor be quietly omitted from the final objective.

- Native Darwin UMA memory observation/allocation, page geometry and
  scheduler-managed placement. Observe actual available memory conservatively
  and feed the existing physical authority. No fake libnuma or duplicate live
  capacity ledger. Explicit unsupported binding policies reject admission.
- Use native page size (Apple Silicon commonly 16 KiB), mmap/file identity,
  reclaim/durability and process RSS/native API contracts; preserve arena/weight
  alignment and allocation leases. Test page/last-page and failure boundaries.
- Native process ownership and retirement equivalent to current pidfd,
  signalfd/subreaper and descriptor proofs; use Darwin-native primitives and
  precise diagnostics, not PID-name scans, oversized timeouts or broad cleanup.
- Resolve the exact OpenMPI dependency: the current schizo module requires
  OpenMPI **4.1.6** internal ORTE layout. Homebrew **5.0.11** discovery is not
  compatibility. Supply a pinned matching native closure or implement/prove
  the new ABI, without bypassing rank environment publication/bootstrap.
- Mach-O loading/install-name/runtime-closure evidence must replace the
  corresponding ELF-specific test mechanics with equivalent proofs. Keep
  Linux implementations and tests first-class for x86 and Linux ARM.

**Exit:** full native CPU build; actual startup, plan/admission, singleton and
multi-rank lifecycle/abort/reset/retirement tests; no model required for this
stage. CPU-only startup must initialize no unavailable accelerator.

**NUMA-less and cache-unpublished Linux ARM hosts.** Intel production hosts
always publish NUMA node topology and CPUID cache geometry; ARM has no user-mode
cache-size query, and some ARM Linux kernels publish neither (no
`CONFIG_NUMA` embedded kernels, QEMU/KVM guests without PPTT cache entries,
Docker Desktop's linuxkit VM). Observed on the M1 Max VM: `set_mempolicy`
ENOSYS, no `/sys/devices/system/node`, cache `index*` entries without `size`.
Inventory now attributes system memory to such a single-package host
(`assignNonNUMAHostMemory`); serving still stops, precisely, at:

- `CPU shorthand requires observed NUMA affinity` (apply mode), and the
  CPU backend/allocator's required NUMA binding;
- `Streaming CPU measurement requires observed aggregate cache coverage`
  (automatic planning), because no cache size is observable.

Supporting these hosts requires (a) a typed "single memory domain without
NUMA policy" CPU endpoint through backend binding, allocator placement,
first-touch proofs and shorthand resolution, with no fake libnuma; (b) a
measured cache-geometry source (bandwidth/latency plateau probe) when sysfs
publishes no sizes, never a guessed capacity; (c) a NUMA-less Unit/preflight
qualification lane. Until then these hosts are explicitly unsupported and fail
fast; DGX Spark/GB10 and ordinary NUMA arm64 server kernels are the qualified
ARM serving targets.

### K — Exact overlays without generic retraining

**Existing owners:** `native_quantised_dispatch/{shape_manifest,format_registry,
candidate_registry,exact_oracle,policy_ir,compiler,profiler_evidence}.py`,
CPU decode/verifier analyzers and runtime policy binding.

- Reuse the one production shape/format/candidate inventory. Register ARM
  implementations/candidates and truthful build/runtime ISA identities there.
  No separate Mac shape list, model name switch or backend-local winner parser.
- Add a checked ARM exact-refresh transaction to the common pipeline. The
  current CPU analyzer/emitter assumes AVX2/AVX512; the CUDA exact-only flags
  are not a supported CPU command today. Extend the real path and test it.
- Install a **total native deterministic policy** before exact measurement.
  Its unseen-shape behavior is a first-class ARM implementation, not an x86
  learned table renamed ARM or a full serial replay. Measure broad generic
  geometry/thread boundaries without fitting a new ML surface.
- Exact overlays may change only numerically neutral candidate dimensions.
  Keys include observable ISA/microarchitecture/worker/format geometry and
  arithmetic identity as required by the runtime. Source aliases must share
  a byte-safe winner or be distinguished by real prepared metadata, never a
  filename the runtime cannot observe. Grouped candidates inherit serial
  arithmetic; an exact entry cannot silently change K parenthesization.
- Retain input/build/source/runtime identity, every candidate/timing sample,
  independent bytes, exact physical resources/profiling and raw evidence.
  Reject stale/missing/partial/prototype evidence and unsupported combinations
  with typed records. Install atomically and prove exact precedence plus
  unseen-M/N/K/positive-worker totality with overlays disabled.
- Do **not** run the full generic decode/verifier fitting, CV/sealed learning
  or CPU-prefill training/install transaction. Ordinary CPU prefill is
  heuristic-only in the current driver, which rejects `cpu-prefill --install`.
  This scope creates no new generic ML certificate and does not weaken other
  existing publication gates.
- Published measurement payloads still belong in `Llaminar/corpora`; local
  result directories stay ignored. Source changes may install small reviewed
  policy artifacts, not raw timing/profiler corpora.

**Exit:** exact-refresh positive/negative tests; real native measured candidate
wins; full-byte and source/profiler provenance checks; total native policy and
unchanged x86 policy artifacts. No accidental generic training invocation.

## 6. Dependency order and meaningful checkpoints

```text
A ISA/build + B conversion/quantization
  -> C all-format native M=1
  -> D grouped/pre-fill/fused + E floating
  -> K exact overlays after arithmetic/dispatch are stable

A/B -> F primitive/stage port -> G attention/TurboQuant
A/B + F -> H GDN/QFN
C/D/E/F/H -> I MoE/sampling transaction integration
J native OS/runtime proceeds as an independent track

all tracks -> full canonical native gates/model inference -> final Perf__ round
```

Each checkpoint must build with Ninja/ccache/unrestricted `--parallel` and keep
its focused tests green. Kernel development may use the core-free test boundary
while J is unfinished, but cannot report serving readiness. Remove superseded
launchers/helpers once a replacement is installed; no second production path
"just in case". Maintain substantive file/method documentation and exact
workspace/ownership comments in touched sources.

## 7. Functional verification matrix

### 7.1 Matrix construction

Use code-owned registries and fixtures rather than another permanent copied
matrix in this plan:

- `NativeQuantisedFormatInfo.h::kAllSourceFormats` / Python `FORMAT_SPECS` for
  quantized sources, with every physical encoding and source-factor alias.
- `NativeQuantisedEquivalenceInventory.h` for ordinary-prefill residue and graph
  bucket boundary witnesses; extend for new physical tile boundaries.
- Existing floating format, grouped operation, cache precision and model parity
  definitions for FP32/FP16/BF16, KV, model/topology and MTP profiles.

Minimum additional ARM witnesses:

| Axis | Required coverage |
|---|---|
| M | Serial M=1; contiguous 2..16, M=31; contiguous extension through any larger admitted verifier capacity; ordinary prefill residues/buckets and unseen values |
| N | Physical NEON column borders and original 64-column logical borders; narrow/tail outputs, large terminal heads, strided/subview/partitioned output |
| K | Legal source block/superblock alignment; single/odd blocks, long reductions, ordered-partition and K-cache-tile boundaries; arbitrary dense floating tails |
| Workers | Every width 1..10 here, including odd widths; synthetic/injected positive-width totality and more workers than useful tasks; nested/top-level OpenMP calls |
| Prepared formats | All 21 source labels, all six encodings, both numerical policies; mixed gate/up/down formats; source provenance preserved |
| Live work | Full/partial/empty/ragged request publications, nonzero frontier, large-to-small reuse, padded capacity and untouched sentinels |
| Numeric edges | Signed zeros, finite extrema, half/subnormal/RNE ties, saturated bytes, cancellation, negative coefficient/minimum, invalid nonfinite admission |
| Lifecycle | Dirty workspace reuse, reset, allocation failure, incomplete/stale prepared state, prefix full/partial restore, all accepted/rejected MTP depths, wrapped KV |
| Execution | Actual native optimized factory/stage route, required counters, warmed no-allocation behavior and correct actual worker width |

A finite sweep cannot enumerate every positive runtime M/N/K/thread count.
Pair residue/threshold/boundary witnesses with typed selector totality and
integer-overflow/unknown-enum negative tests. Never infer totality from exact
model cells alone.

### 7.2 Reuse canonical registrations

Existing anchors (inspect current CTest registration when running):

- `V2_Integration_GroupedVerifierRows_CPU_*` is the owning grouped namespace.
- `V2_Integration_CPU_NativeQuantised_PrefillBucketMInvariantAllFormats`,
  `...PrefillHeuristic`, `...PrefillCacheKTile_*`,
  `...NativeQuantisedWeightPackingAllFormats`, verifier workers, source-window and
  original-equation tests cover production preparation/dispatch semantics.
- Existing GPU-aligned quantized/floating expert, mixed-format transaction,
  Q8 reuse, fused completion and workspace-capacity gates remain relevant.
- CPU attention native-format/wrapped-ring/ragged-request, RoPE, RMSNorm,
  residual, fused norm, SwiGLU, KV append, GDN/short-conv and sampling tests
  need ARM-native entries, not copies silently forced to AVX2.

Unit stays fast and device-free. Add focused integration regressions explicitly
to `ProductionTestPreflight` and then broaden the existing format/backend
sweeps. Native kernel failure loops use up to 20 repetitions when appropriate.

### 7.3 Complete inference qualification

Once J and the kernels build:

1. Build `v2_unit_gate` and `v2_production_test_preflight_gate`; run the complete
   configured Unit namespace and preflight label, not just ARM-focused tests.
2. Run all applicable CPU grouped, format, preparation, KV/prefix/reset and
   multi-rank tests. Preserve unsupported hardware as explicitly unqualified;
   do not manufacture passing GPU evidence on this CPU-only host.
3. Admit real model cells from the canonical typed definitions that fit actual
   physical memory. Cover dense and MoE, MTP off and dynamic depth, real
   stochastic sampling, prefix full/partial hits and optimized kernels.
   Mathematical HF parity is focused diagnosis of newly ported math, not an
   excuse to regenerate reviewed generation expectations on drift.
4. Exercise the QFN source-bound precision/activation policies and ngram/MTP
   continuation behavior present on this branch. Do not disable these features
   to claim the branch runs on ARM.
5. Run the public CPU serving/HTTP path and applicable E2E/lifecycle workloads;
   verify actual route/workspace/publication counters and no leaked native
   processes/resources. Multi-host/cross-ISA claims require actual peers and
   corresponding evidence, not local rank-number assumptions.

## 8. Final `Perf__` testing round

This round is mandatory at the end of implementation, not satisfied by the
six small foundation tests or the existence of benchmark sources.

### 8.1 Extend existing performance owners

| Family | Existing owning source / target | ARM work required |
|---|---|---|
| NativeQuantised decode/prefill candidates | x86: `native_quantised/Perf__CPUNativeQuantised_DispatchSweep.cpp`; ARM: `native_quantised/Perf__CPUNativeNEONExactOverlay.cpp` | Native candidates, source/ISA identities, exact-only collection, same shared shapes |
| NativeQuantised all-format GEMV/grouped/fusion | `Perf__CPUNativeQuantised_GEMV.cpp`, `Perf__CPUFusedVerifier.cpp`, `Perf__CPUNativeQuantised_Sweep.cpp` | Real ARM Auto and forced candidates, complete output authentication and all formats |
| NativeQuantised thread scaling | `Perf__CPUNativeQuantised_ThreadScaling.cpp`; `v2_perf_cpu_native_quantised_thread_scaling` | Current fixture is Q4_K only and samples selected widths: extend to every format/encoding and all 1..physical widths; retain decode/grouped/prefill distinctions |
| Attention | `Perf__CPUFlashAttentionTileTournament.cpp`, `Perf__CPUFlashAttentionVerifierRows.cpp`, `Perf__CpuFlashAttentionSweep.cpp` | Native formats/tiles, canonical summaries, all M and actual worker ownership; adapt Linux-only driver placement |
| GDN/short-conv | `Perf__CPUGatedDeltaNetVerifierRows.cpp`, `Perf__CPUGatedDeltaNetPrefill.cpp` | Exact fixed-epoch replay, accepted-state publication, all admitted widths/heads/threads |
| RoPE | `ops/Perf__CPURoPEKernel.cpp`; `v2_perf_rope_kernel` | Same production typed tests/position layouts and native quantized forms |
| RMSNorm/residual | `Perf__CPURMSNormKernelT.cpp`, `Perf__RMSNormThreadSweep.cpp`, `Perf__CPUResidualAddVerifierRows.cpp` | Native reduce/fused routes and every physical-team crossover |
| SwiGLU/softmax | `Perf__CPUSwiGLUKernelT.cpp`, `Perf__CPUSoftmaxKernelT.cpp`, `primitives/Perf__CPUSwiGLUKernel.cpp` | Dense/expert numerical programs, low-precision internal paths and fused-operation accounting |
| TurboQuant | `turboquant/Perf__TurboQuant.cpp`; `v2_perf_turboquant` | Native quant/dequant/rotation, direct attention consumption and cache footprint |
| MoE router | `performance/kernels/moe/Perf__CPUMoERouterVerifierRows.cpp` | Routing + reused activation + complete expert operation, not synthetic precomputed route IDs alone |

Paths above are under `tests/v2/performance/kernels/cpu/` unless explicitly
otherwise. Extend these authorities and their shared fixtures. Create a new
`Perf__` target only for a genuine missing operation, with canonical registration;
not a parallel Mac benchmark framework or an alternative shape list.

### 8.2 Workload and measurement discipline

- Release executable and linked production core, same compiler/ISA/policy as
  serving. Integration/sanitizer results establish correctness, not timing.
- Retain admitted weight/activation/workspace/state buffers; validate bytes and
  guards before and after timing. Record and restore stateful initial state or
  replay a declared identical trajectory. No allocation, packing, transfers,
  snapshots, validation or profiler startup in the timing interval.
- Sweep decode M=1, every verifier M=2..16 plus 31/capacity boundaries, and the
  canonical physical prefill inventory. Cover small-cache-resident,
  cluster-cache-bound and streaming/out-of-cache weights with production shapes.
- Measure every worker count 1..10 on this host; report 1..8 and 9..10 without
  claiming hard P-only binding. Linux ARM uses real physical-core/NUMA pinning.
  Record observed team width, core classes, thermal/power state and migration
  evidence when available. QoS is not a core-ID guarantee.
- Interleave candidate order (A/B and B/A), warm separately, retain raw repeated
  samples and losing candidates. Report median and p95, sample/noise bounds,
  complete-operation latency, per-row latency, useful bytes/s, integer/FP work,
  speedup and efficiency. Do not overlap timing with builds or other candidates.
- Compare grouped work with M independent production serial calls under the
  same numerical plan and worker budget. Include quantization, mandatory
  reduction, nonlinear epilogue and publication in complete-operation totals;
  additionally report leaves for attribution, not as a substitute.
- Decode can saturate memory bandwidth before every core is busy. Require
  explained, economical scaling and absence of accidental serialization, not
  fabricated linear speedup or a hidden thread cap. A slower E-core-inclusive
  point remains visible; change production worker policy only with evidence.

### 8.3 Profiling and installation evidence

Linux ARM: use the existing exact worker-TID `perf_event_open` evidence
interface, adapting hardware event availability with typed failures. On Darwin,
implement a native evidence adapter around supported Instruments/xctrace data
where it can authenticate the exact process/thread/symbol/interval. The current
`NativeQuantisedProfilerControl.h` includes Linux perf headers unconditionally and
must not be represented as already portable.

One exact `(operation, format/encoding, M,N,K, ISA, worker policy, candidate)`
per profiler interval. Keep setup, oracle work, warmups, canonical timing and
other candidates out. Record symbol/loop identity, instructions/cycles/IPC,
cache/memory traffic when actually accessible, vector registers and hot-loop
spills from disassembly/compiler evidence. No private unsupported counter API,
no profiler-derived latency labels, no guessed zero counters.

If a metric is inaccessible, retain an explicit limitation and establish an
equivalent supported observation or obtain a separate qualifying ARM host;
static instruction counts alone do not prove achieved bandwidth or IPC.
No exact policy publication from incomplete mandatory evidence. The explicit
no-generic-retraining scope does not waive profiling, numerical or provenance
requirements.

### 8.4 Final acceptance and report

The final report must join:

1. Complete kernel/operation/format/ISA coverage and current-head Unit/preflight,
   grouped byte, state/lifecycle and real-model/HTTP evidence.
2. Before/after complete-operation and production inference measurements, with
   identical workload/prompt/sampling/prefix/MTP intent and no deterministic
   diagnostic dispatch. External Intel/llama.cpp comparisons are separately
   hardware-qualified, not passed off as same-machine speedups.
3. All-format, one-through-physical-core `Perf__` curves, explained saturation,
   no accidental single-worker path and measured grouped/prefill amortization.
4. Exact candidate provenance, full-buffer equality, instruction/resource and
   available hardware-counter evidence; unchanged x86 dispatch artifacts.
5. Native serving startup/admission, repeated requests, reset/prefix/MTP
   continuation and clean process/file retirement; explicit unqualified hardware
   or topology claims remain unadvertised.

There is no meaningful final ARM-support checkbox while any admitted operation
runs through a diagnostic scalar route, a grouped output/state differs from
serial, the Mac runtime is stubbed, or the required `Perf__` round has not run.

## 9. Immediate implementation sequence

1. A/B: native ISA/capability and independent integer/rounded FP/Q8 contracts,
   compiled and tested without the Linux-only core; keep the existing six tests.
2. C: retain the prepared ABI, extract its ARM-usable descriptor seam and wire
   native all-format M=1 through the existing projection interface.
3. D/E: bounded grouped/pre-fill/fused leaves, canonical worker/workspace plans,
   floating and two-term policies, followed by focused operation economy tests.
4. F/G/H/I and J: close every remaining SIMD family and OS-serving dependency;
   maintain the source inventory and semantic test matrix as actual code changes.
5. K: exact overlays only, then full native qualification and the final `Perf__`
   round. Do not invoke generic retraining to postpone missing native kernels.
