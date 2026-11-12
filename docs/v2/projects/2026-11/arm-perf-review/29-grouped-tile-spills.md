# Idea 29 — Pre-existing grouped-tile FP/SIMD spills on ARM (Ratchet follow-up)

Status: **Open (perf item)**. Found 2026-11-09 by widening the tile-register
detector (SOURCE114 direction).

## Finding

Production gcc-14 on .80 (probe at NEON_I8MM; cells = programs with spills /
spill instructions, out of 16 per family):

| Build | R2 ASIMD/SDOT/I8MM | R3 ASIMD/SDOT/I8MM | R4 ASIMD/SDOT/I8MM |
|---|---|---|---|
| base 302318f1e | 14/30, 8/16, 2/6 | 14/46, 14/52, 16/52 | 16/947, 16/70, 16/40 |
| R1 035d1d948 | 14/28, 12/24, 2/6 | 14/158, **14/463**, 16/58 | 16/966, **16/666**, 16/48 |
| fix b01d285dc | 14/30, 8/16, 2/10 | 14/52, 14/65, 16/54 | 16/947, 16/66, 14/34 |

- The grouped row tiles (Rows 2–4) spilled in the BASE — the worst is **R4
  ASIMD ≈ 950 spill instructions**. The earlier [sp]-only census missed these
  (gcc addresses them through x29).
- The R1 regression was real in grouped SDOT (R3 52→463, R4 70→666); the fix
  brings both back to base level.
- Rows=1 is ZERO in all three builds on Linux gcc-14 (the M=1 spill seen
  earlier was Darwin gcc output — see idea 28 for the open M=1 investigation).

## Gate design (ratchet, approved 2026-11-09)

- Rows=1 (decode): ZERO FP/SIMD spills enforced (hard rule).
- Rows 2–4: committed per-program RATCHET BUDGETS keyed by compiler identity
  (id + major version) + ISA envelope; values = today's counts (fix ≈ base);
  over budget fails; unknown compiler identities FAIL CLOSED. Never change
  production allocation to fit the scanner.

## Follow-up

This is a real perf item: reduce/eliminate the grouped-tile spills (R4 ASIMD
~950) — C4 (wider panels, addon lane) touches the same tiles; coordinate when
C4 lands.

### B9-STRENGTHENING FOLD LANDED (station session, ~23:08Z): 108bce2e — 64/64 OK

Independent station implementation of the B9 method (register-evidence
classifier: post-index general load `ldr x,mem,imm` + unscaled halfword
`ldurh w,mem` admitted in the tile-witness grammar; post-index beside a
scaled offset/with writeback/from-register stays unclassifiable), commit
108bce2eab8d9860a82456b7ff9aafe0a163f09e (tree d71a3615a; parent = exact
tip 825fe8deb; 2 files, 31+/1−).
VALIDATION (station-native, compiler-free, the registered cells' own
commands): RED baseline 63 tests w/ exactly the 3 deferred controls
failing → AFTER **analysis shard 64/64 OK** (the folded criterion) +
binding shards 4/4 (25 tests); py_compile clean; no production/checker/
registration/inventory change; no perf claims; hooksPath unset recorded.
NOTE: f265d2950 (addon's B9 composition) not in the station object DB —
independent implementation vs the branch's own 3 controls; reviewer can
range-diff. Evidence: /home/tdsynnex/repo/llaminar-perf-b9/artifacts/
b9-strengthening/ (baseline-red/ + after-method/ logs, receipt SHA256s);
bundle ffbbb8a8… + patch 935c18a6…. PUSH authorized (FF from 825fe8deb
only); branch cleanup for b9-strengthening = post-review.

### B9 PUSHED (station, ~23:10Z): 825fe8deb → 108bce2e (FF, verified)

codex/arm-b9-strengthening advanced exactly as authorized: pre-push
ls-remote = 825fe8deb, post-push = 108bce2eab… (fast-forward, no force).
The B9-strengthening branch is now implemented + validated (64/64) +
pushed; reviewer range-diff vs the addon's f265d2950 composition remains
available if desired.

### FMLAL station co-validation: dependency transport solved locally

Watcher r2 exit 2 as predicted: the codex lineage's configure fetches
FOUR pinned third-party deps before any target (pcre2 10.49 + utf8proc
2.11.3 tarballs @ src/v2/CMakeLists.txt:1361/1364; ftxui v5.0.0 + libfort
v0.4.2 git tags @ :1385/:1402) — the no-network container cannot. NOTE:
with GPU languages off, OneDNN PASSED on this tree ('V2: BLAS backend:
OneDNN' — the idea14 lock blocker did NOT reproduce). Fix in flight:
hash-verified local provision (SHA256s match the pinned URL_HASH
constants exactly; shallow pinned-tag clones for ftxui/libfort) +
FETCHCONTENT_SOURCE_DIR overrides at repo-local copies (ignored
artifacts dir; shared external/ untouched; network-none container;
versions/hashes unchanged — ONLY the transport is local). Provisioned
rerun launching; exit + JUnit + SHAs to follow.

### B9 VERDICT (reviewer): SOURCE HOLD at 108bce2e — one blocking omission

f265d2950 was ALREADY LOCAL to the reviewer (read-only range-diff, no
fetch/execution). **BLOCKING: the independent composition admits the two
forms but OMITS f265's base/transfer overlap refusal + its three focused
negatives.** :1209-1212 goes from post-index-shape validation straight
to base_values — nothing rejects a writeback base that is ALSO a
transferred GP register; the loaded destination is assigned at :1245
then :1262-1268 overwrites it with the advanced OLD base (selecting one
interpretation of architecturally constrained-unpredictable behavior).
'ldr x1,[x1],#8' classifies CLEAN instead of structural failure.
Witness: str d8,[sp,#40]; mov x10,sp; ldr x10,[x10],#40; ldr d8,[x10];
ret — the model invents x10=sp+40 and ACCEPTS the d8 restore (static
predicted false-clean). FIX: restore the fail-closed overlap invariant
(canonical gp/gp_view comparison BEFORE transfer/update) + the three
alias controls (post-index ldr / pre-index ldr / post-index ldp) from
f265; keep the new grammar + strengthening controls. The 'post-index
combined with !' test does NOT cover destination/base aliasing.
POSITIVE: non-overlapping post-index ldr semantics correct; whole-X
slot propagation present; ldurh exact-2-byte/not-ABI-eligible; new
controls meaningful. No production/budget/inventory delta.
DISPOSITION: station 64/64 stays attributed (the controls omit the
failing invariant — green does not transfer f265's ACK); fix = port
invariant + controls; B4 round-5 unchanged (Linux/JUnit handoff).

### B9 FROZEN (Main scoping accepted, ~23:53Z): register-gate apparatus under the NOGO

Main's reading: the B9 file IS the register-evidence checker
(CPUNativeTileRegisterEvidence.py) = register-gate apparatus under the
operator's NOGO scope (#166/#167/#191 descoped) → Main will NOT apply
B9 to PR #43 without literal operator authorization text, even after a
clean verdict.
ACTIONS TAKEN: station instructed to STOP (no overlap-invariant port);
the reviewer's HOLD verdict stands as FINAL; codex/arm-b9-strengthening
@ 108bce2e stays pushed as FROZEN UNMERGED TOOLING (the #58 precedent).
COORDINATOR ERROR OWNED: the B9 fold was dispatched from the standing
queue list without re-checking its scope against the register-gate
NOGO — the scope-check is now a standing part of the dispatch step for
anything touching the register-evidence checker. No further
station/reviewer time on B9 unless the operator provides literal
authorization.

### PR #43 REBASE IN FLIGHT (Main): new base d1fd0a077

Main is rebasing PR #43 onto d1fd0a077 (operator asked Main to resolve
upstream conflicts); force-push once validated; new tip to be announced.
Lane branches on the old lineage (FMLAL 0b53fa7a4, B4 52fec643f, idea16
a6aee9bac) rebase at handover per the anchor-refresh rule; coordinator
routes the new base when announced.

---

## **B9 TOPIC CLOSED (operator decision: "leave it benched and close this topic")**

FINAL STATE: codex/arm-b9-strengthening @ 108bce2e stays pushed as
FROZEN UNMERGED TOOLING (the #58 precedent) — the tile-witness grammar
admits the two new forms; the base/transfer overlap refusal + the three
f265 alias controls are NOT ported (the reviewer's HOLD verdict stands
as the final word). NO further work by any lane; no re-verdict is
coming; B9 is REMOVED from the collection-candidate inventory and will
NOT be applied to PR #43 now or later without fresh explicit operator
authorization. The topic is CLOSED, not parked as open tooling (the
NOGO-means-CLOSE discipline). The station's reusable assets from the
session (worktree-local OneDNN install + legacy-transport v7 tooling)
survive for future station builds.
