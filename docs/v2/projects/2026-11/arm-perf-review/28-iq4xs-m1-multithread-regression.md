# Idea 28 — IQ4_XS M=1 multi-thread regression after the metadata seal (R1)

Status: **Open (investigation)**. Found in addon window (b), 2026-11-09.

## Finding

Window (b): base `302318f1e` vs R1 `035d1d948` (Release NEON_I8MM, gcc-14,
production gemv path, ABBA, median of two launches per arm, 31 iters; binding
OMP_PLACES=cores OMP_PROC_BIND=close).

- **IQ4_XS 17408×5120 M=1: +6.6% @1 thread, −4.5% @10, −4.2% @20** (both
  launches consistent).
- IQ4_XS 2048×4096 M=1: +5.0 @1, **−4.1 @10**, +1.4 @20.
- IQ4_XS 512×2048 M=1: +11.7/+11.1/+7.3%.
- Everything else grouped/single-thread: **+1–12%** (IQ3_S 512×2048 M=4
  +9.0/+7.0/+7.6%; IQ4_XS M=4/M=15 positive everywhere; IQ3_S 17408×5120
  M=1 +0.9/+2.7/+2.0% etc.).

Claim boundary (lane, honest): byte-identical; **+3.1% 35B MTP-off decode**
(23.31 → 24.04; p50 41.8/42.2→40.5/40.8 ms), neutral MTP (33.94 vs 34.0,
±0.3% within noise incl. one launch outlier; acceptance 0.99 both), prefill
unchanged (~49.5–50 tok/s). NOT claiming economy for the IQ4_XS M=1 cell.

## Suspected cause (within the R1 diff)

- `noexcept` / `__builtin_unreachable` dispatch change (in-loop throw removal);
- the unchecked admitted accessors changing scheduling in the Rows=1 IQ4_XS
  nibble-source-scale path — under bandwidth pressure (17408×5120, multi-
  thread).

## Investigation plan (bounded, ~30 min)

1. Disassembly diff Rows=1 IQ4_XS nibble-source-scale path (base vs R1) —
   scheduling/inlining + dispatch edge.
2. Isolated confirmation ≥5× 31-iter at 10/20t @17408×5120 (raw+p95); if it
   holds, A/B the two candidates (noexcept dispatch vs admitted accessor).
3. Decision rule: codegen-attributable → FIX IN R1 (small amendment); clears
   as artifact/noise → record + close with the item tracked (R2 watch item).

## Evidence

`.80:build/results/w-r1-20261109T031823Z` + `build/wr1-summary.py`.

## ROOT-CAUSED + FIXED (2026-11-09) — b01d285dc (parent d1f1e8e34)

- CAUSE: R1 tile programs spill vector registers — with UNCHECKED
  admittedScales/admittedMins accessors, gcc hoists later metadata/payload
  loads into the dot program and spills accumulators/scale vectors
  (str q31,[sp]…ldr q25,[sp] once per block in Rows=1 IQ4_XS hot loop).
- Attribution (production gcc-14 -O3; gcc-16 same): base 0 spilling fns/0
  spill insns/43442 insns; R1 45/1154/38392; R1 w/ THROWING accessors 0/0/
  42866; address-dependency-only 40; memory-barrier-after-dots 31; hoisted
  strides 43; FIX (admitted accessor + FATAL NORETURN COLD bound check)
  0/0/38548. I8MM envelope: base 0, R1 48, fix 0. Compact kernels never
  spilled. MECHANISM: the old throwing check was the scheduling boundary
  stopping speculative hoisting; fatal noreturn provides it without
  exceptions — run path stays noexcept, instruction savings retained,
  in-loop reserved-bit test stays removed, arithmetic unchanged, M1 bytes
  pass.
- REGRESSION (maxim 22/24): CPUNativeTileRegisterEvidence.py — build-time
  probe of every tile (Rows1..4 × ASIMD/SDOT/SMMLA × policy × encoding) at
  -O3; ANY q/v stack traffic in run/pairBlock fails the build.
  V2_{Unit,Integration}_CPUNativeNEONTileRegisters re-verify the retained
  proof; adversarial tests V2_*_CPUNativeTileRegisterEvidence. Checker
  rejects d1f1e8e34 (120/176 spill), accepts base + fix (0/176).
- Reviewer informed (fatal accessor = small semantic delta; optional look).
- Pending: .80 build+gate at b01d285dc + 3-arm confirmation runs (base /
  R1-spilling / fix, ≥5×31, IQ4_XS+IQ3_S, 17408×5120+2048×4096, M=1/4,
  10/20t, ~45 min).

## Register-evidence gate hardening (SOURCE114 preliminary, 2026-11-09)

- Production fatal accessor + SIGABRT/terminate-77 regression = COHERENT
  (positive). Gate must upgrade before claiming 'complete' (else diagnostic):
  1) header-closure authentication: hash/bind the ACTUAL transitive
     compilation inputs (compile_commands-derived or manifest) + compiler/
     config identity; verify() rejects STALE production headers.
  2) detector width: catch stack-relative vector AND scalar FP (s/d) spills
     through ANY base (sp or fp, constant offsets); DISTINGUISH ABI
     save/restore prologue/epilogue from program spills (exclude only the
     former); include s/d widths.
  3) inventory: expected symbol/body inventory from every explicit
     instantiation (+ pairBlock), inline/call-closure accounting, partial-
     program negatives (mutate one body → reject).

## SOURCE114 FINAL (2026-11-09) — fatal accessor ACK; gate on HOLD

- Narrow source ACK for the fatal-accessor delta + focused regression
  (bounds reject before cast/pointer arithmetic; valid addresses unchanged;
  fputs+abort noexcept cold/noinline; no FP tree/dispatch/transfer change;
  fixture checks all 6 IQ3_S units × both accessors + all 8 bad-coordinate/
  accessor combos SIGABRT+diagnostic term→77; preflight registration exists).
  Direct-accessor regression — NOT a worker-team execution or rollback proof.
- REGISTER-EVIDENCE GATE: HOLD (tooling successor required, separately
  identified): exact compilation dep closure (omitted deps confirmed incl.
  tensors/NEONFP16.h + kernels/common/MoEProjectionNumericalContract.h) +
  RECORD the command EXECUTED; header-change negatives; classify-or-reject
  fp/SP-derived-base + s/d + ABI-save distinction; expected emitted-program
  inventory (rows×family×policy×noncompact encoding) + body/callee closure +
  partial/inline/outlined/empty-body negatives; never change production
  allocation to fit the scanner.
- Owner evidence (scheduling/0-176/M1 bytes/.80/3-arm) remains owner evidence.

## Update (2026-11-09, widened detector) — grouped confirmed; M=1 OPEN

- GROUPED regression confirmed + fixed: R3 SDOT 52→463→52 (base→R1→fix);
  R4 SDOT 70→666→~70; R2 30→24→30 (ASIMD); Rows=1 = 0 in all three on Linux
  gcc-14. The fix is justified by the grouped-tile spill regression.
- M=1 ATTRIBUTION NOT CONFIRMED on Linux gcc-14 (earlier spill = Darwin gcc
  output). IQ4_XS M=1 −4% cause OPEN: 3-arm run decides whether the fix
  changes M=1; else investigate separately (bind/admission per call, code
  alignment). Ratchet gate approved (idea 29): Rows=1 zero; Rows2–4 per-program
  budgets keyed by compiler id+major + envelope; unknown → fail closed.

## Tooling successor 7abfc0dc1 (2026-11-09) — Source115 scope

- The b01d285dc gate FAILED .80 Integration on Linux gcc-14: pre-existing
  grouped spills addressed through [sp] there (114 programs), x29 on Darwin —
  old [sp]-only gate would have BROKEN CI. Superseded.
- Detection: all FP/SIMD widths (b/h/s/d/q/v) via sp, x29, or add/sub/mov-
  derived aliases (retired on reassignment); ABI saves d8–d15 classified
  separately (1 save, ≤1 restore/return).
- Inventory: probe inventory string; every expected rows×family×policy×
  dispatched-encoding/map/minimum emitted exactly once, non-empty body;
  duplicate/missing/empty/missing-inventory rejected.
- Callee closure: local + tail callees audited; external = violation except
  never-returning fatal + abort.
- Policy (approved ratchet): Rows=1 ZERO; Rows2–4 per-program budgets
  (CPUNativeTileRegisterBudget.json) keyed by CMake compiler id + major +
  envelope; unknown/stale fail closed; qualified GNU-14 + AppleClang-21 ×
  NEON/NEON_DOTPROD/NEON_I8MM; budget totals gcc-14 1029/1176/1274, AppleClang
  1429/2017/2185; census action for future budgets.
- Binding: -M closure per-file hashes (incl. NEONFP16.h,
  MoEProjectionNumericalContract.h), executed command+directory, target
  command digest, compiler version, checker/envelope/budget hashes; Ninja
  depfile (CMP0116); Mac checks (touch NEONFP16.h regenerates; edit
  MoEProjection → verify fails until rebuild).
- Tests: 13 negative cases. SOURCE115 forwarded. .80 run at 7abfc0dc1.

## SOURCE115 preliminary (2026-11-09) — four pathological paths to close

- (1) unknown callees cost1 paid from budgets (key2:0:0:0:0:0 budget4 can pass
  an uninspectable `bl helper; ret`); blr/br ignored. Structural/closure
  failures must be FATAL independent of allowance.
- (2) alias analysis textual: conditional alias (`mov x9,sp; cbz; mov x9,x1;
  ldr q0,[x9]`) reads stack on the taken edge but reports 0; `[sp,x9]` and
  `add x9,sp,#1,lsl #12` unmatched; unmatched forms continued over, not
  rejected → CFG-aware or fail-closed; never continue-over.
- (3) d8–d15 excused by counts alone: no save-before-clobber/slot/epilogue
  proof; in-loop/unmatched could pass as ABI saves → prove or reject.
- (4) include closure uses ORIGINAL flags, compile adds -O3/-fno-sanitize=all/
  -fno-lto → recorded -M set may differ from the executed closure → bind the
  EXACT executed flags' closure.
- Reviewer continues inventory/binding/depfile; ratchet = reported-approved
  (not zero-spill/economy qualification); Linux-RED + Darwin-false-clean
  history separate.

## SOURCE115 FINAL (2026-11-09) — SOURCE116 tooling scope (HOLD)

- Ratchet = reported-approved policy, NOT zero-spill/economy approval.
- P1a structural failures fatal independent of spill counts (uninspectable
  callee as cost1 within budget; blr/br uninspected; fatal exception substring
  → exact no-return identity).
- P1b CFG + interprocedural provenance; [sp,x9]/shifted-imm handled or
  reject; alias retirement only on matched defs; callee traversal preserves
  provenance (caller sp-derived x0 + callee [x0] store = spill); ABI exemption
  only with save-before-clobber/slot/lifetime/epilogue proof (count ceilings
  rejected).
- P1c inventory from admitted ENVELOPE (not same assembly string); reject
  unexpected/unaccounted programs (drop-Rows1-with-symbols fails); bodies need
  real instructions; no circular test derivation.
- P1d closure = executed compilation's (exact effective regime; logical include
  resolution preserves symlink identity; pre+post input hashing).
- P2a verify() validates observed compiler identity against configured policy.
- P2b declare ALL outputs/byproducts (proof.s); success JSON LAST.
- Scope: probe header stale under ratchet; SDK-independence not claimed.
  Owner evidence stays owner; Linux-RED/Darwin-false-clean stand.

## .80 gates GREEN @7abfc0dc1 + 3-arm window running (2026-11-09)

- BUILD=0 (witness: 176 programs, Rows=1 zero, grouped within budget);
  Unit 963/965 (2 known base); Preflight 2311/2314 (3 known target-selection);
  Focused 212/212 (fatal-access bound, tile-registers Unit+Integration,
  evidence). 3-arm window (302318f1e/035d1d948/b01d285dc, Release NEON_I8MM,
  IQ4_XS+IQ3_S 17408×5120+2048×4096, M=1/4, 10/20t, 5 interleaved reps) running.
- SOURCE116 in progress (CFG/provenance, structural-fatal, exact no-return
  identity, interprocedural provenance, ABI-save proof, envelope inventory) —
  real GNU-14 counts unchanged; AppleClang −14/envelope (old checker
  double-counted callee-saved reloads) → RE-CENSUS + update budget JSON when
  the tooling lands.

## RESOLVED (2026-11-09) — b01d285dc; idea 29 stays open

- 3-arm window (5 interleaved reps, Release NEON_I8MM gcc-14, .80):
  17408×5120 M=1: 10t base 1014.5 (1011–1020) / R1 1060.2 (+4.5%) / fix
  1009.7 (−0.5%); 20t 536.8 / 558.5 (+4.0%) / 543.9 (+1.3%) — R1 spreads
  tight, non-overlapping; fix overlaps/touches base. 2048×4096 M=1 earlier
  −4% did NOT reproduce (R1 0.966/0.955 of base; fix inside noise) → noise.
  Other cells base-level or faster.
- MECHANISM (honest): Rows=1 ZERO spills in all three on gcc-14 → M=1
  recovery NOT explained by spills; likely the throw-path codegen
  (scheduling/layout) in the decode loop removed by the cold noreturn fatal —
  evidence = timing recovery + zero-spill census only.
- STATUS: idea 28 RESOLVED with b01d285dc; idea 29 open (pre-existing grouped
  spills). SOURCE116 tooling successor submitted (f6be6b510).

## SOURCE116 owner evidence (f6be6b510) — GNU-14 .80 E2E green

- Witness PASS 176 programs (Rows=1 zero; grouped within budget, total 1274
  == committed GNU-14 budget); proof binds 242 inputs; recorded identity
  GNU-14 == configured; 4/4 tests (TileRegisterEvidence + NEONTileRegisters,
  Unit+Integration); lifecycle: removing proof.s re-ran exactly the witness
  edge, next dry run no witness work (Ninja -n CONFIGURE_DEPENDS glob
  re-check = pre-existing, unrelated).
- Owner evidence set: Mac AppleClang E2E; .80 GNU-14 E2E; census 9 GNU + 9
  AppleClang assemblies, zero structural errors; 27 adversarial tests.
- R1: complete from the lane (SOURCE116 under review; WSL x86 gate pending).

## SOURCE116 FINAL (2026-11-09) — tooling HOLD; successor scope (SOURCE117)

- P1a interprocedural: no return/arg/stack-memory summaries + caller-frame
  ownership (x0–x18 cleared on call; x8 indirect-result + caller-stack args
  omitted; callee-visible memory effects absent) → summaries or structural
  rejection; outside budgets.
- P1b operand semantics: BFI/MOVK partial defs (implicit inputs); partial-load
  SP reconstitution (byte/bit ranges or reject); SIMD lane-aware ABI proof
  (LD1 d[1] high-lane writes; LD1R/LD2 don't inherit LDR semantics) → proven
  opcode/lane/layout only.
- P1c grammar: retain global-label statements; explicit admitted
  instruction/directive grammar (`.byte`/`.incbin` code-generating forms);
  compute() fail-closed admission.
- P1d dependency normalization: filesystem-lookup semantics BEFORE hashing
  (normpath after symlink resolution; decoy `/p/h.h` vs real `/q/h.h`
  control).
- P2a abort property: nonreturning ≠ abort — require the declared abort
  property or reject unproved cycles (no-abort + mixed loop/abort controls).
- Positives kept; never change production allocation for the checker;
  owner-evidence framing stands.

## SOURCE117 successor e6afa66e8 (2026-11-09) — closures

- P1a structural rejection (only admitted call = exact fatal noreturn symbol;
  all other boundaries fail; census confirms tiles call nothing else).
- P1b merge defs read dest; 32-bit view of stack addr fails; slots keep
  offsets (partial/narrower/sign-ext/FP reloads fail; partial overwrites +
  untracked/unknown-offset stores fail); ABI exemption = contiguous
  whole-register d/q str/stp/ldr/ldp at exact slot (ld1 d[1]/l{1}r/2/1.1d/st1/
  s-views excluded).
- P1c global-label statements kept; closed grammar; non-code directives only
  in .text (.byte/.word/.inst/.incbin/.zero/.skip/.fill/.rept fail); data-label
  branches + fall-through/across-section fail.
- P1d no lexical normalization (symlink-before-.. resolution; hash at resolved
  file; depfile resolved; decoy-vs-real control passes/fails correctly).
- P2a proven-noreturn = acyclic + every path ends in bl abort (spin/mixed/
  loop+FP/return/tail/blr/ordinary call/.byte disprove; two-path abort proves).
- Owner: 18 census assemblies zero structural errors + identical counts
  (budget JSON unchanged); 33 tests; Mac AppleClang E2E green (WeightView.h
  touch re-runs edge). GNU-14 .80 E2E queued (5-min).

- e6afa66e8 GNU-14 E2E GREEN (arm-sparks, .80): witness PASS 176 programs;
  4/4 tests; proof GNU-14 NEON_I8MM 242 inputs, total 1274. SOURCE117 now has
  GNU-14 + AppleClang end-to-end evidence. Lifecycle flag resolved: .80 tree's
  CONFIGURE_DEPENDS glob re-runs CMake every invocation → `ninja -n` shows only
  manifest regen; the REAL build after rm proof.s re-ran the witness edge
  (PASS line + "Generating proof.json, proof.s" = build-only output); Mac tree
  (stable manifest) lists the edge directly; proof.s = declared OUTPUT +
  DEPENDS → lifecycle holds.

## SOURCE117 FINAL (2026-11-09) — successor scope (SOURCE118)

- P1a indexed-load provenance (unknown-address loads accepted; dest NOT_STACK
  with live pointer slots → half-reconstruction zero-counted); tainted joins
  by union lose nonstack alternatives → MAY/MUST slot lattice.
- P1b CSINC/CSINV/CSNEG transform-or-reject; width-vs-extent + Apple lane
  arrangements (.s .1d etc.) for ALL slot-protecting accesses; Apple + ELF
  overlap controls.
- P1c alignment-directive OPERAND validation (fill bytes executable);
  section-at-global-label; dialect bound to real format declarations; per-op
  FORM admission (operand-less add fails).
- P2a diagnostic proof under admitted grammar + exact-exception set
  (noreturn_symbols direct-abort alignment).
- P2b probe-source spelling preserved (abspath collapse false-rejection).
- Progress kept; never alter production allocation; owner evidence separate.

## SOURCE118 successor c0be497e4 (2026-11-09)

- P1a may/must slot sets + nonstack alternatives; unknown-offset + live
  pointer slots fail; heap-escape fails.
- P1b csinc/csinv/csneg/cinc not admitted (csel joins); exact footprints
  (widths/narrow/sign-ext/structure per opcode+count+lane/arrangement, ELF or
  Apple suffix); unknown footprints fail; both-dialect controls (st1.s 4B
  partial = spill + pointer alive; 16B retires).
- P1c admitted-FORM inventory (dispatched per compiler witnesses; operand-less
  add/svc/smmla fail; docs not 'closed grammar'); .p2align only w/o fill;
  section-in-force at label; dialect bound to compiler identity + format lines.
- P2a diagnostic = admitted forms + fwrite/fputs only + acyclic + every path
  aborts; tiles call ONLY exact fatal symbol (direct abort = structural fail).
- P2b probe-path spelling preserved (resolve-as-kernel; decoy/real controls).
- QUALIFICATION: AppleClang-21 prints SMMLA NO-OPERANDS (-S, 480 in I8MM
  witness) → AppleClang-21 NEON_I8MM REMOVED from budgets; Mac I8MM build
  FAILS CLOSED (no receipt; reason in _doc). GNU-14 all + AppleClang NEON/
  NEON_DOTPROD re-audit clean, budgets unchanged. 40 tests; Mac NEON_DOTPROD
  E2E green. GNU-14 .80 E2E queued (non-urgent).

## SOURCE118 FINAL (2026-11-09) — successor scope (SOURCE119)

- P1 whole-witness assembler state: reject unmodelled conditional/macro/
  include/symbol-rebinding controls across preamble + data sections +
  non-audited bodies (`.if 0` split/`.endif`-in-rodata; `.macro fmov→str q0`)
  or bind actual expansion — whole-file, structural, never budgetable.
- P2 relative path authority: resolve against the EXECUTED compile directory
  (verify vs build cwd mismatch) or reject relative sources pre-compile;
  differing cwd test with real/decoy.
- Precision: T:154–155 ≠ composed heap-escape trace (narrow claim or add it).
- Qualification boundary: AppleClang-21/NEON_I8MM evidence must be RECOVERED
  (suggested: objdump/llvm-objdump of the probe OBJECT — full operands both
  compilers) — fail-closed is honest current state, not a green exclusion;
  never alter production allocation.

## SOURCE119 census FULLY GREEN (2026-11-09; sealed K14-SOURCE119-CENSUS-20261109)

- Witness PASS 176 (single-row zero; grouped within budget); ctest 4/4; proof
  GNU-14 NEON_I8MM 242 inputs/1274, GNU objdump 2.42; lifecycle after proof.o
  removal = 1 (PASS re-emitted). Census 9/9 cells (Debug/Release/Integration ×
  NEON/NEON_DOTPROD/NEON_I8MM): single=0, grouped = budget EXACTLY
  (1029/1176/1274), mismatches=0. Last GNU-14 gap closed — the requalified
  object-disassembly gate has its complete owner evidence (AppleClang Mac E2E
  + GNU-14 .80). Reviewer SOURCE119 (source) pending; .80 free.

## SOURCE119 FINAL (2026-11-09) — successor scope (SOURCE120)

- P1a complete instruction coverage: bind admitted ranges + symbol
  entries/extents to the object; reject gaps/prefix/tail/overlap/duplicate/
  out-of-order/cross-section; consistent positive fixtures + deletion/
  extent/address controls (hash-binding an incomplete rendering ≠ proof).
- P1b trusted callees by BINDING: external-import-only libc exceptions; reject
  defined/rebound/alias trusted symbols; ELF + Mach-O controls + undefined-
  import positives; proven_noreturn must bind the selected bodies.
- P1c assembler-input freshness: .include/.incbin consumed by the assembler
  outside -M/-MD → bind into stability/rebuild/verify closure or reject
  unbound cases; mutation control alongside C++ header controls.
- Precision: 'lo12 on stack base FAILS' = memory operands only (ADD generalizes
  to UNKNOWN_STACK) — narrow claim or enforce/test the broader rule.
