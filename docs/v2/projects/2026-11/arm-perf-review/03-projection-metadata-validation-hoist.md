# Idea 3 — Move projection metadata validation out of the K loop

Status: **Open** (at reviewed head the inner-loop throw is still present).

## Context

In `src/v2/kernels/cpu/gemm/CPUNativeNEONPreparedProjection.h`:

- `run()` (~:254) walks the K blocks inside the 4-column outer strip.
- For `IQ3SExpandedSourceScale` and `NibbleLUTSourceScale`, the reserved-bit
  check `Ops::metadataFits(...)` (in `CPUNativeNEONMultiScaleOps.h:129`) plus a
  `throw std::invalid_argument` executes on **every** K block (~:330).

The weights are an immutable publication, so the check result cannot change
during one invocation. This is both economy and collective safety:

- The check is pure overhead in the inner loop for a value that was already
  validated (or should be) at admission.
- `run()` executes inside the workshare (workers), so a throw from one worker
  while others are inside the `omp for` can leave peers at the barrier —
  the same collective-safety class as the RMSNorm `domain_error` issue
  (now fixed via typed `rejectNumeric`, see idea 10).

## Proposed change

1. Validate the complete metadata plane once before the workshare starts (or at
   view/publication admission), so an invalid publication is rejected before
   any worker enters.
2. Back it with an authentic immutable-publication proof: prepare, wire receive,
   slot reuse and migration must all prove generation/ownership, not a pointer
   cache or a shape-only check (admitted spans alone do not establish capacity
   or leases).
3. Keep a "fail fast and fatal" diagnostic (consistent with
   `rms_numeric::rejectNumeric`, `[[noreturn]] noexcept`) instead of a throw
   inside the team.

## Exactness / risk

- Removing the per-block check changes no arithmetic; the validation outcome is
  unchanged (reject before work vs throw mid-work).
- The publication/generation proof is the correctness-critical part; it must
  cover slot reuse and migration, otherwise staleness could be hidden.

## Validation

- Mutation controls: a metadata word with reserved bits must still fail, at
  admission rather than inside the loop; all-format/encoding sweeps (M 1..N,
  workers 1..20) and byte auth against serial.

## Scope

ARM NEON tile only; the x86 `CPUNativeVNNIMultiScaleKernels.h` checks are a
separate (shared) path with the same pattern but different owners.

## Addon implementation (2026-11-09) — f058a2953 part of R1 09cba9963

- Native weight bytes behind CPUNativeVNNIPublishedBytes generation counter;
  CPUNativeVNNIMetadataSeal validates metadata once per publication; the
  per-encoding coverage table (== old in-loop rules, padding included) is in
  the commit body; all 6 producers seal; overlay slot acquisition withdraws
  admission until the arrival reseals; NEON run path noexcept (static_asserts);
  writer-policy sanitizer (narrow named allowlist) + negative controls pass.
- M1 core-free (preliminary, not authoritative): MetadataSeal 6/6,
  WeightView 4/4, PreparedProjection 8/8, ProjectionBatch 10/10, NEONExecution
  10/10, NumericalContract 5/5, writer negatives 5/5; run<> 44045→37434
  instructions. Syntax-only: gcc-16 ARM + clang x86_64-AVX512. Pending:
  core-linked tests (.80) + WSL x86 gate. x86 loadScales checks untouched.

## Reviewer P1 #2 (2026-11-09, R1 @ 09cba9963) — stale test fixture

- CpuExpertSlotPool.AcquisitionWithdrawsAdmissionUntilArrival fills the whole
  native stream with byte(epoch): epoch1 → IQ3_S 0x0101=257, epoch2 →
  0x0202=514 — both violate ≤15 (not only epoch3's 0x0303) — so the new seal
  correctly rejects at epoch1 while the test expected success for 1/2.
- The PRODUCTION seal is right (in-loop-only validation previously let
  invalid metadata through until compute); the FIXTURE must be fixed: seed
  valid secondary words for accepted epochs (epoch-distinct in-range values
  OK), corrupt a single coefficient only in the rejected epoch. Needs the
  .80 core-linked run (not covered by core-free/M1 suites).

- FIXED at R1 tip 26d74e718 (2026-11-09): fixture-only (seal unchanged) —
  accepted epochs seed 64 in-range epoch-distinct secondary words
  (epoch + column%3, max 5); payload stays byte(epoch); ONLY epoch 3 corrupts
  one word (0x0010, mid unit, col 17) and must fail; epochs 1/2 admit, keep
  root + seal on republish. Core-linked run (.80) pending;
  V2_Integration_CPUNativeVNNISlotReuseAdmission is the cell.

## Reviewer final (SOURCE109, 09cba9963 vs ea5402a54) — P2s into R1

- P2a: MetadataSeal::rebound PUBLIC + extent-only check → can rebound a valid
  seal to a corrupt same-sized stream (fromPublished admits w/o scan); make
  private/friend to the deep-copy owner or require an unforgeable copy
  receipt. Public Tile::admitted bypasses requireExecutable/geometry (div0 for
  GPUAligned K) → confine to the admitted batch builder. API hardening (no
  observed exploit).
- P2b: nonfinite-weight death test must prove child nonpublication: SIGABRT +
  escaped-exception/terminate discriminators, child-visible/shared output
  witness + write-before-reject negative; Rows1/2/4; single diagnostic stdio.
- Wording: 'every whole-head integer reduction' (multi-block combine in FP32).
- Positive: seal rules per-lane coverage correct; Q16 bound sound (whole-head
  calls incl. :5964 guarded; 5661/5983 reduce ≤ block_elems, FP32 combine);
  no FP32 tree/clamp change; comments substantive.
- R1 CLOSE GATES: successor native byte/geometry/team paths (.80), core-linked
  slot/movement/serialization (.80), x86 WSL Unit/preflight, resource/economy
  evidence. No merge/execution grant from reviewer.

## Hardened R1 5c89f981c (2026-11-09) — P2s closed

- P2a: MetadataSeal::rebound PRIVATE (friend: CPUNativeVNNIPublishedBytes,
  after its deep-copy); Tile::admitted/executeAdmitted PRIVATE (friend:
  CPUNativeNEONProjectionBatch — admits features/publication/policy/K-tree/
  spans before forming the team); concept checks w/ positive controls pin both.
- P2b: NonfiniteWeightIsFatalBeforeAnySharedNumeratorWrite
  (V2_Integration_NEONAttentionStorageAdmission) — MAP_SHARED numerator,
  Rows1/2/4 × NaN/±Inf × every row, SIGABRT w/ diagnostic (terminate handler
  exits 77 → escaped exception/terminate fails), parent byte-check +
  write-then-reject negative, single stdio per diagnostic.
- Wording: 'every whole-head integer reduction' (multi-block ≤ block_elems,
  FP32 combine) in commit + source; 746e694a2 body untouched (SHA stability).
- M1 core-free: Seal 6/6, WeightView 4/4, PreparedProjection 8/8,
  ProjectionBatch 10/10, NEONExecution 10/10, NEONAttentionMath 8/8; syntax
  (gcc-16 ARM, core-linked slot, x86 Seal) pass.
- OWED (booked): .80 Integration NEON_I8MM full run + core-linked
  slot/movement/serialization; WSL x86 Unit+preflight; economy evidence
  (IQ3_S/IQ4_XS R1-vs-ea5402a54 M1+grouped ABBA; 35B decode ABBA).

## Interim .80 run + seal semantics refinement (2026-11-09)

- Run (a) @46c9ffc58 (Integration NEON_I8MM, gcc-14, all targets): Unit
  960/963, Preflight 2303/2309, focused 82/83; failures = 2 known base + 3
  target-selection-known + NEW-MINE (MoEOverlayPhysicalResidencyFabric,
  MoEOverlayLivePreparedWeightBytes, MoEPipelineStagePhysicalResidency,
  MoEPipelineCPUHostMaintenance) — all from the TEST FIXTURE.
- 2a301e392 (fixture): makeSourceExpertAtLayer filled streams with
  seed+byte·13 (reserved bits) and never admitted sources before CPU service
  measurement → now masks coefficients into domains + admits.
- e0d2a0fd3 (PRODUCTION seal semantics, reviewer attention): ExpertOverlay
  movement transfers LIVE columns only → partial 64-column chunk padding keeps
  slot content; seal now admits every LIVE column (column < N), padding NOT
  admitted; vector lanes past N computed independently + discarded.
  Coverage: reserved bits in last LIVE column rejected; arbitrary padding
  accepted. Proof: 0xffff padding → published bytes identical (IQ2_S/IQ3_S/
  IQ4_XS). x86 per-block checks unchanged. Zero-fill alternative REJECTED
  (allocation/first-touch change).
- 4 'outside-diff' BUILD failures = classes fixed by Main's 302318f1e (branch
  predates it) → LANE REBASES onto 302318f1e before final R1 close.
- Reviewer SOURCE111 recheck requested on e0d2a0fd3.

- R1 REBASED onto 302318f1e (no conflicts), tip **b645ca7d1** (force-pushed).
  SOURCE111 recheck = b645ca7d1. .80 close run restarted at b645ca7d1
  (incremental; Unit/preflight/focused incl. full MoEOverlay/MoEPipeline).
  Perf window (b): base = 302318f1e (confirmed) — IQ3_S/IQ4_XS gemv threads
  1/10/20, M=1+4/15, 3 shapes ABBA; 35B decode MTP off/dyn ABBA saved plan.

## SOURCE111 P1 (2026-11-09, b645ca7d1) — cross-ISA contract mismatch

- Seal accepts poisoned padding; production x86 consumers still REJECT it:
  loadScales512/256 test every loaded IQ3/IQ4 lane (258/263, 283/288);
  CPUNativeVNNIGemv.h checks all four 64-col chunk vectors (1535–1538/
  2402–2405); IQ2 dispatch<16> :2329 defaults columns=64 + checks every
  metadata vector (MultiScaleKernels 623–625). Accepted partial-N arrival can
  throw inside x86 execution.
- Direction (binding): COMMON CONTRACT — producer writes benign padding
  (0x00, in-domain for all encodings) AT PUBLICATION in the same write that
  places live columns (per-arrival; NO allocation/first-touch change).
  Seal: live-only w/ producer guarantee, or belt-check padding==benign (lane
  choice). x86 focused control (poisoned-padding partial-chunk arrival fails
  BOTH admissions). NEON proof must exercise REAL mixed live/padding vectors
  (N%4=1/2/3, chunk tails, grouped I8MM/SDOT/ASIMD, poisoned ADJACENT lanes
  actually loaded, output guards) — N=8/Rows1 proof didn't load cols 8..63.
  Old pre-seal checks were NOT strictly live-only for N%4!=0 (withdrawn).

## SOURCE111 FINAL (2026-11-09) — amended-contract requirements (binding)

- HOLD on P1 (shared-seal/x86-consumer): SOURCE110 approval NOT inherited
  (different all-lane contract). Fix direction CONFIRMED with requirements:
  1) publication-time normalization inside the owned rewrite generation,
     after live writes, before sealing/reachability; descriptor-owned
     planes/extents; live-only communication retained + extents asserted.
  2) tests DISTINGUISH raw poison rejection vs normalized-arrival acceptance.
  3) READER contract proved (x86 AVX2/512 IQ3/IQ4 loaders + IQ2 dispatch see
     benign padding — WSL x86 control).
  4) no new allocation inherently; owner evidence for unchanged
     first-touch/NUMA/economy (.80/WSL).
  5) P2 proof amplitude: mixed live/padding vectors, final-chunk cases, all
     families (grouped I8MM/SDOT/ASIMD) + policies, poisoned adjacent lanes
     actually loaded, output guards, focused preflight. Fix stale 'padding
     lanes included' prose in seal test/CMake.
- Positives: fabric fixture + post-write publication OK; live loop covers N
  bound incl. chunk last live word; 4fb400ecb closes the Q16 precision notes.
- Amend → separate successor SHA for re-review; owner/runtime/WSL/economy
  evidence still required.

## SOURCE111 amendment — 661fc9284 (2026-11-09, parent b645ca7d1, base 302318f1e)

- CONTRACT: padding defined by the PRODUCER — normalizePaddingMetadata zeroes
  secondary words of columns N%64..63 of the final chunk in every block;
  extents from descriptor + asserted; writes through unsealedGenerationBytes
  (owned generation; FATAL on a sealed one); no new allocation/first-touch.
- Arrival publish (admitCPUNativeVNNIPublication) normalizes THEN seals;
  movement remains live-only. Seal back to ALL LANES (belt).
- Tests distinguish: raw poisoned padding REJECTED; normalized arrival
  ACCEPTED, every other byte unchanged.
- Regressions: core-free PaddingNormalization; NEON TailPaddingContract
  (N%4=1/2/3 + chunk tails × every family × ASIMD/SDOT/SMMLA × Rows1..4 ×
  both policies — poisoned LOADED adjacent lanes rejected before writes,
  benign padding vs bytes/oracle/guards); x86
  PartialChunkPaddingIsNormalizedForX86Readers (_X86 preflight: poisoned
  padding fails seal AND x86 consumers (IQ2_S dispatch<8/16>,
  loadScales256/512); normalized owner passes both). M1 green; x86 syntax
  only (WSL = proof). SOURCE112 recheck requested.
- .80 run at 661fc9284 in progress (incremental; BUILD=0 on prior tip →
  302318f1e cured outside-diff failures). Window (b) base = 302318f1e vs
  661fc9284 after.

## SOURCE112 preliminary (2026-11-09, 661fc9284) — two guard/proof items

- Normalization fixes the x86 acceptance mismatch for VALID layouts (positive).
- (1) normalizePaddingMetadata geometry guard computes (N+63) in SIGNED int
  (:233) — unsealed fixture N=INT_MAX overflows before precise rejection; never
  compares blocks_per_row vs ceil(K/32) → changing K alone passes. FIX:
  canonical checked structural geometry (widened/checked products,
  positive/source/encoding admission) BEFORE mutable bytes; malformed N/K
  controls proving NO mutation.
- (2) x86 negative combines readers sequentially (IQ2 throws @dispatch<8> before
  <16>; IQ3/IQ4 @loadScales256 before 512) → EXPECT_THROW(consume()) doesn't
  falsify missing AVX512 checks. FIX: invoke/assert EACH compiled reader
  separately (dispatch<8>/<16>, loadScales256/512) on poisoned + normalized.
- Reviewer still inspecting NEON family/policy tests + arrival ordering; final
  scope follows. Successor re-check = separately identified.

## SOURCE112 addendum (2026-11-09) — two more

- (3) UNKNOWN-ENCODING 0-STRIDE CASE: unknown encoding, is_asymmetric=true,
  N=1, N_padded=64, blocks_per_row=1, data_stride=interleaved_block_stride=0,
  EMPTY storage → prepared*Stride(unknown)=0 → all comparisons pass (0-byte
  extent), then normalization forms data()+256 and memsets past null/empty.
  FIX: reject unsupported source/encoding + validate canonical structure
  BEFORE any mutable span/pointer arithmetic; no executable metadata-admitted
  WeightView before sealing. Same fixture as a regression (reject, no mutation).
- (4) COVERAGE: verifyTailPadding focused matrix = contiguous Rows1/2 +
  ordered Rows1/2/3/4 (NOT Rows1..4 × both policies as claimed) → add
  CONTIGUOUS Rows3/4 (preferred) or narrow the claim.

## SOURCE112 FINAL (2026-11-09) — fix scope consolidated

- HOLD on normalization-admission P1 (sent items): checked canonical structural
  admission BEFORE writes; no executable unsealed view; invalid N/K/source/
  encoding/storage coverage + unchanged-byte asserts.
- Proof corrections: per-reader x86 poisoned/normalized; contiguous Rows3/4;
  poison-rejection AND positive-execution = COMPLEMENTARY proofs, separate
  fixtures (not one normalized-arrival trace).
- Ordering ACK (positive): poll ready-before-completion.ready; ready/claimed
  precede arrival.publish; normalization+seal before reachable engine-field
  install; seals not rewritten; writer-policy gate recognizes
  unsealedGenerationBytes; core-free test proves only intended secondary-padding
  intervals change on valid layouts; 3 focused registrations present. No
  arithmetic/transfer-extent change; normalizer no allocation. First-touch/
  NUMA/economy unchanged = OWNER evidence (not claimed).
- Housekeeping: remove duplicate @throws logic_error in the touched admission.

## SOURCE112 successor — 035d1d948 (2026-11-09, parent 661fc9284)

- (a) normalizePaddingMetadata geometry from
  CPUNativeVNNIWeightView::publishedGeometry (canonical admitStructure: source/
  encoding/positive widened geometry/exact stored extent; plain geometry only,
  no readable/executable view); owner fields compared before any mutable span/
  pointer arithmetic. Regression
  PaddingNormalizationRejectsMalformedOwnersWithoutMutation: N=INT_MAX/N=0,
  K-alone/K<0, unknown source/encoding, foreign blocks_per_row/N_padded, short
  storage, empty unknown-encoding zero-stride asymmetric owner — bytes
  unchanged; ASan/UBSan clean M1.
- (b) x86 TWO owners (raw/normalized); per-reader IQ2_S dispatch<8>/<16>,
  loadScales256/512 on every block's padding vector — reject raw+accept
  normalized, seal agrees (clang-x86 syntax; WSL proves).
- (c) verifyTailPadding Rows1..4 × contiguous + ordered. (d) complementary
  separate-fixture proofs. (g) duplicate @throws removed. (f) measured-only.
- SOURCE113 re-check queued. .80: 661fc9284 Unit 961/963 (2 known-base
  suspect); then incremental rerun at 035d1d948 + window (b) after.

## SOURCE113 final (2026-11-09, 035d1d948) — R1 STATIC GATE PASSES

- Narrow source ACK, no new P1; SOURCE112 normalization P1 source-closed.
  R1's agreed no-P1 static-review gate passes at 035d1d948. Not
  native/x86/resource/economy/integration/merge qualification.
- Acked: view returns plain geometry only (admitStructure private, checked
  source/encoding/policy, widened N/K, padded-ABI overflow, checked
  extent+alignment); no unsealed executable view escapes; cached layout fields
  compared before the mutable borrow; bounds from canonical geometry; only the
  final chunk's secondary padding cleared; generation ownership +
  normalization→seal→reachable-publication ordering intact; no arithmetic or
  transfer-extent delta. Regression covers the malformed list incl. the exact
  empty owner; retains bytes compared. x86 per-reader 8/16-lane raw+normalized
  closes the AVX2-masks-AVX512 finding; NEON matrix Rows1..4 × both policies;
  complementary fixtures documented.
- NONBLOCKING precision note: Test__CPUNativeVNNI_GEMV.cpp:194–202 does NOT
  prove IQ2 rejection at EVERY K block (both blocks poisoned, dispatch starts
  at 0 → first bad block throws); IQ3/IQ4 have explicit per-block asserts →
  independently poison each IQ2 block or narrow the claim (test precision
  only, no re-review).
- R1 CLOSE = the remaining evidence gates: .80 rerun @035d1d948, WSL x86,
  window (b) economy; then R2 (rebased on successor + C2c).

## R1 evidence progress (2026-11-09) — green modulo base

- IQ2 precision item (035d1d948 → **d1f1e8e34**, test-only): each IQ2_S K block
  poisoned alone; dispatch<8> AND dispatch<16> must each reject it. WSL target.
- .80 owner evidence @661fc9284 (Integration NEON_I8MM, gcc-14, base
  302318f1e): Build BUILD=0; Unit 961/963 (MoEForbiddenDependencyScan +
  TestTargetSelection, known base); Preflight 2308/2311 (Release/Pooled/
  Qwen4Exp target-selection, known base); Focused 207/207 (incl.
  TailPaddingContract, full MoEOverlay/MoEPipeline, slot reuse, seal, padding,
  writer policy; the three fabric/pipeline cells that failed at 46c9ffc58
  now PASS). Owner runs, not authoritative.
- Rerun at d1f1e8e34 in progress (build/r1c.status); Release builds for window
  (b) (302318f1e + 035d1d948; kernel identical to d1f1e8e34) chained after;
  (b) announced when done.
