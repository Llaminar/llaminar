<!-- Provenance: reviewer ruling #166 (2026-11-10), delegated by the operator.
     Text below is committed VERBATIM as the standing rule. -->

### Standing rule: installed NEON tile register gate

**Claim level.** The installed-set register contract is a mandatory, fail-closed change-admission gate, not advisory evidence. Reworks, rebases, compiler changes and checker changes inherit it; ordinary re-certification does not reopen this decision.

**One authority and explicit scope.** Use the existing CPUNativeTileRegisterEvidence.py machinery and its canonical census. CPUNativeTileRegisterBudget.json owns the numeric limits; CPUNativeNEONTileRegisterProbe.cpp, the checker and V2CPUArchitectureTests.cmake own the installed inventory and compilation contract. Cover every declared compiler/ISA/configuration cell and every enumerated program, not just the cells that ran or passed. The gate measures the checker's per-program FP/SIMD stack-transfer instruction sites, excluding proven ABI saves/restores. It does not claim zero general-register spills, final-application code identity, runtime correctness or economy.

Performance-only forced/experimental variants remain diagnostic while unreachable in production. Default or user-selectable production admission requires their promotion into the canonical coverage and a complete pass first. Calling a production route experimental does not exempt it. Do not delete a required cell, program or architecture to clear a failure.

**Pass contract.** Every single-row program must have zero classified spills. Every grouped program must remain at or below its committed per-program ceiling. Every required cell must pass; reductions elsewhere cannot offset a failing key. Structural failures, unsupported instructions/calls, incomplete coverage, unknown identities and malformed or missing proof are never budgetable.

**Bound evidence.** A qualifying census consists of the existing canonical bound build/verify evidence for every required cell. The stand-alone census subcommand is diagnostic: it does not apply budgets or publish the complete build/verify receipt. Its exit status, a tree stamp, a summary marked GREEN, patch equality or equal budgets alone cannot satisfy this gate.

Validate the actual final configured candidate, including its complete dependency and command identities, compiler/assembler, object, disassembler, checker, envelope, inventory and budget. Any changed bound input requires fresh canonical evidence for the affected cells. Reuse unchanged cells only when the existing verifier validates their complete binding against that candidate. Checker composition and policy edits invalidate the bindings they change. Never splice incompatible epochs or relabel an old success after a failed collection. Preserve failure evidence; incomplete collection publishes no current pass.

**Operational consequence.** A classified spill/budget violation is a register-contract failure. Missing, stale, incompatible or infrastructure-failed evidence leaves the candidate unqualified; it is not by itself proof of a spill regression. Both withhold promotion, merge admission and qualification of the affected installed change until resolved. Attribution determines repair ownership, not permission to pass.

This is not a repository-wide work freeze: independent work retaining a valid covered state may proceed and reuse valid proof. A candidate containing the failing installed state cannot bypass the gate by calling its newest diff unrelated. Split or revert the offending change, fix it, or use the narrowly permitted budget amendment below. Other required gates remain in force.

**Downward ratchet.** Compare against the pre-change committed ceilings; do not let collection rewrite its own acceptance limits. A complete admitted census that lowers a grouped key's worst count must lower that key's committed ceiling. Where configurations share a budget key, use the maximum count across all required configurations sharing it, not the most favorable cell. This is a ceiling ratchet, not a separate tracker-only last-run baseline. Disclose per-cell changes even when they remain within a shared ceiling.

Automation may propose monotone reductions. Adopt them as an explicit source diff through ordinary review, with final proof bound to the resulting policy. Tightening needs no special exception approval; unattended rebaselining is not allowed.

**Only escape hatch: an explicit grouped-budget amendment.** Before acceptance, obtain a named register-gate reviewer's written ACK and normal integration-owner approval for the exact grouped keys and finite old-to-new ceilings. Retain the original failure, exact source/tool/configuration identities, bound per-program evidence, the reason for the increase, its disassembly attribution, and the applicable correctness/economy evidence. Explain why the trade-off is justified; recertification cost or a desire to turn the check green is not justification.

Commit the named change to the existing budget authority and verify the complete required census against the final source/checker/policy combination. A pass against a proposed loosening does not approve that loosening. Record acceptance as a reviewed budget amendment, never as non-regression against the old ceiling. Approval covers only that exact amendment; later increases require new approval.

No such amendment may waive single-row zero, structural completeness, binding/freshness, required coverage or another correctness/economy contract. No skip, anonymous reserve, parser blind spot, copied receipt or generic override is an escape hatch. Checker repairs require their focused controls and the existing complete checker gates before requalification.

The tracker records this standing rule, evidence references and explicit decisions. It is not a second measurement engine or live threshold ledger. Passing this gate does not replace any other production admission gate.

### #175 successor complete: d4cdcd998 (arm-addon, 2026-11-11 ~02:43Z)

`codex/arm-s4-nibble-witness` @ d4cdcd998 on origin (9 commits on e8866ec04:
5 re-pinned patches + InstalledWidth pin + #169 fix-back + comment fix +
2ff599ec0 reconciler hardening + 38ed21507 reflow + d4cdcd998 =
coordinator regression/registration).

**d4cdcd998** closes the #175 successor (checked-in device-free coordinator
regression + preflight registration):
1. `test_cpu_native_tile_register_gate.py` drives the ACTUAL shell
   coordinator end-to-end against a synthetic candidate tree (stub CMake
   registers the same 14 CTest alias names as echo tests; two proof
   targets as controlled stubs; real checker/budget + canned production
   header → reconciliation/retention/publication run for real).
   - Positive: 9-cell happy path → gate=PASS, publication=authenticated,
     manifest + per-cell accept digests.
   - Negatives (one failure class each, #174): coordinator-owned setting
     refusal (both spellings, exit 2, nothing created); stale receipt dir
     replaced (no masquerade); stub build failure → FAIL(build); stub
     ctest failure → FAIL(ctest); unregistered alias → FAIL(not-passed);
     missing retained artifact → FAIL(missing/retention); malformed proof
     → FAIL(reconcile-crashed), never PASS-with-empty-identity; wrong
     proof identity → FAIL(reconcile); MID-RUN DRIFT (alias mutates the
     shared production header through the same python -c command ctest
     runs) → PUBLICATION REFUSED ("production header changed under the
     epoch").
2. Coordinator hardening (driver-only): per-cell accept.digest.json
   verified at publication before the manifest; compile_commands entries
   must reference the candidate header; refusal-pattern reflow.
3. Preflight registration: V2_(Unit|Integration)_CPUNativeTileRegisterGate
   Coordinator (core-free device-free; Integration one in
   ProductionTestPreflight) via the existing metadata-suite loop.

FOCUSED EVIDENCE (AppleClang-21): regression 10/10 (~62s device-free);
hardened coordinator on the real tree 9/9 PASS publication=authenticated
45-digest manifest; CMake syntax sanity.
UNVERIFIED-AT-LANE (→ Main's single full campaign): 18×14 canonical
receipts at the final pin; portability matrix; full preflight; Integration
execution of the new registration on Linux/GNU-14.
NEXT: await Main's post-B12 placement-fix tip → mechanical re-rebase →
reviewer exact-source ACK at the final pin → NO FF until Main's 18×14
current-binding acceptance.

### Reviewer verdict #175 successor — HOLD at d4cdcd998; consolidated #191 (RG1–RG4)

- **RG1 (cleanup/write/copy completeness):** initial sha substitutions,
  mkdir/rm/recreate, summary writes, per-cell mkdir/status writes and
  tee/publication are not explicitly checked; set-e switches on only after
  the first reconciliation (behavior depends on the earlier cell); the
  final publication substitution under active set-e can exit BEFORE
  pub_rc/failure report; the unused phase helper retains unchecked
  writes/copy. The new run checks (command/exit records, retain-check cp)
  are good partial fixes but not cleanup/publication completeness. NEED:
  actual shell failure injection for required infrastructure actions — not
  successful stale-marker replacement.
- **RG2 (configured-project identity):** the filter misses attached
  -S/-B/-GNinja and split '-D' 'CMAKE_BUILD_TYPE=...'; cache preload can
  override owned variables; validation happens AFTER build/inventory and
  does not inspect actual CMAKE_HOME_DIRECTORY/CMAKE_CACHEFILE_DIR or
  generator — containment of entries alone is NOT configured-project
  identity. NEED: seal owned configuration BEFORE building; test
  redirect/alternate project within the same root, stale persistent build
  dir, and bypass spellings.
- **RG3 (accepted-closure staleness):** per-cell copies are hashed only
  AFTER copying (authenticating whatever landed there, not the bytes the
  canonical checkers accepted); the final check hashes only
  CPUNativeNEONPreparedProjection.h, not the complete source dependencies/
  compiler/SDK/checker/envelope epoch — earlier accepted originals can go
  stale after their checks without touching that one header. NEED: use
  original binding authorities for the protected accepted closure or a
  final canonical revalidation, plus a verifiable retained mapping — no
  receipt rewrite/second analyzer.
- **RG4 [P2] (compile-database shape):** final publication requires a
  compile_commands entry whose `file` contains CPUNativeNEONPrepared-
  Projection — an INCLUDED HEADER, not the compiled TU; the canonical
  register target owns CPUNativeNEONTileRegisterProbe.cpp/custom proof
  compile; the synthetic stub fabricates the header as compile entry,
  concealing incompatibility with real compile databases. NEED:
  authenticate the header via canonical dependency closure, not invented
  TU presence.
- TESTS: the actual shell is driven and Unit/Integration preflight aliases
  registered (confirmed). But the ten controls don't force cleanup/write/
  copy failure, configuration redirect, post-check/pre-copy corruption,
  non-header dependency/tool drift, or real compile-database shape;
  --cmake/--ctest are parsed but ignored (driver uses PATH); active Ninja
  is not resolved/sealed by the driver; build/ctest omit --parallel.
  CLOSE THESE FOCUSED CONTRACTS; no lane-wide matrix requested. No
  execution here; owner 10/10/9/9 unauthenticated. Final rebase
  exact-diff review + Main-owned 18×14/full-collection gate remain
  separate; no FF/source ACK.

### #191 fix-back landed: ab860378b (arm-addon, 2026-11-11 ~03:27Z) — IN REVIEW

FF from d4cdcd998; 10 commits total; driver-only. RG1: every infra action
self-checked (mkdir/cleanup/summary/status/appends/verdicts; failure →
exit 2 REFUSED, nothing published); ROOT CAUSE found by the injection test
— set-e left ON after reconciliation silently aborted the publication
substitution (9 PASS cells, no verdict); set +e restored, every rc checked.
Owned STUB_INFRA knobs force both infrastructure failures. RG2: attached +
split override spellings refused; post-configure no-op re-configure
(regeneration fails) + cache CMAKE_HOME_DIRECTORY/source-dir/generator =
configured-project identity. RG3: accept.digest.json schema 2 seals the
COMPLETE closure (artifacts + all common inputs); publication verifies
sealed input digests — post-acceptance drift refuses. RG4: compile DB must
REFERENCE the header (file or command), not name it as compiled TU; stub
no longer fabricates header-as-TU. Regression now 12 tests OK device-free;
real tree 9/9 PASS publication=authenticated (45 artifacts, 6 sealed
inputs); --cmake/--ctest consumed, --ninja sealed via CMAKE_MAKE_PROGRAM.
Hold unchanged: campaign + re-pin await Main's source-handover base.
