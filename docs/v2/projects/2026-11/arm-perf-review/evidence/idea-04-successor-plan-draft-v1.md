# Idea-04 successor implementation plan (draft v1 — execution gated on the M=1 window)

Prepared 2026-11-11 at `bb893cd79` (verified carry). This supersedes the
amplification claims in the first interpretation guide: a deeper source read
shows the C4/R2 land ALREADY implements candidate 2 for the production M=1
leaves. The plan below is written against the true residual set, with an
honest decision tree for the window outcome.

## 0. What is already implemented (do NOT re-propose)

At `bb893cd79`, `CPUNativeNEONPreparedProjection.h`:

- **Panel path (S=2/4, DotProd leaf)** — nibble + expanded encodings:
  `activationBlocks<Ops>(block)` loads every row's 32-byte Q8_1 payload ONCE
  per block into two `int8x16_t` registers (C3 seam `ActivationBlock{low,
  high}`), and every strip reads its group by lane via
  `Ops::dotLane<Group>` = `vdotq_laneq_s32(acc, weights, source, Group%4)`
  (register operand, no memory access). ASIMD fallback
  (`vdupq_laneq_s32` + `dotFourColumnsASIMD`) also reads from registers.
- **Q6_K panel (S=2)**: `activationGroup(row, block, group)` reads the
  group's 4 bytes once per (row, group), hoisted out of the strip loop —
  over 8 groups that is exactly 32 bytes = the payload, each byte read once.
  NOT an amplification (8 scalar loads vs 2 vector loads is an L1-throughput
  nuance, not a bandwidth one).
- **Metadata hoist**: `nativeQuantisedFP16ScaleToFP32(a.d)` is computed once
  per block per row, above the strip loop. Per-strip metadata that remains is
  the WEIGHT scale (`admittedScales(... + column)`), which is inherently
  per-column data.
- The register witness enforces Rows=1 zero spills at these widths (4 strips
  expanded, 2 strips nibble/Q6_K/minimum) with grouped budgets only down.

**Consequence**: the production M=1 decode leaf (Rows=1 DotProd, panels) reads
each activation byte exactly once. The baseline's Q8_K 10-thread regression
(2.96×, measured 2026-11-07) PREDATES the C4 panel land (2026-11-10). The
window at bb893cd79 therefore tests whether the regression SURVIVES the
already-implemented hoist — and if it does, the reload hypothesis is
disconfirmed for the production leaf and the plan shifts to the alternatives
in §4.

## 1. Residual surface A — the S=1 strip walk (run() else-branch)

Paths still on the strip walk: the ASIMD leaf (Rows=1, no DotProd), SMMLA
row-pair tiles (Rows>1 I8MM), and shared-row tiles. There, `consume()` memcpy's
the 4-byte activation word per (row, block, group, COLUMN) — the same word is
reloaded columns/4 times per block. THIS is the live per-column amplification,
on non-production-at-M=1 leaves.

**Restructure (block-outer / columns-inner, order-exact):**

1. Invert the strip-walk loop nest to `for block { blocks[row] =
   Ops::activationBlock(activation_[row][block].qs); for column { ... } }`.
   Per-column block/group order is unchanged (each column's accumulator chain
   sees blocks in the same order — the exactness invariant), and the weight
   stream becomes linear in the interleaved layout (`unit + Group*256 +
   column*4` for ascending columns within one block unit) instead of
   re-walking the same unit per column.
2. Replace `consume()`'s scalar `memcpy` + broadcast with
   `Ops::template dotLane<Group>(dots, weights, blocks[row])` — the same
   seam the panels already use; the ASIMD fallback keeps register feeding
   via `vdupq_laneq_s32`.
3. `activation_scale` hoisted per (row, block) as in the panel path.
4. Panel() vs the restructured walk: the walk keeps `#pragma clang fp
   contract(off)/reassociate(off)` and the per-column program unchanged; only
   the A-operand source moves from memory to registers.

**Register-pressure prediction (Rows=1, ASIMD leaf, S=1):** live set per
block step = accumulator f32 (4B) + blocks (32B = 2 q-regs) + weights q-reg
+ decoded/temp (≤2 q-regs) + scale/min vectors (≤2) ≈ 7–8 of 32 q-regs —
no spill risk at any column count because the loop is NOT unrolled across
columns (columns iterate serially; per-iteration live set is bounded). The
tile-register witness runs anyway (Rows=1 zero spills, budgets only down).
For the SMMLA row-pair path: `activationPair` already fuses two rows into
one SMMLA A operand per (pair, group); the hoist replaces two 8-byte reads
with one 32-byte block load per row pair + in-register lane assembly — live
set +4 q-regs for two rows' blocks; SMMLA tiles are Rows=2/4 so the witness
budgets apply (they must stay only-down; the tournament re-runs).

## 2. Residual surface B — Q6_K panel scalar loads → block loads (marginal)

Unify Q6_K's `activationGroup` per-group scalar loads with
`activationBlocks` + a Q6_K-aware `dotLane` (8 lane extracts from the block
pair instead of 8 scalar loads). Expected effect: L1 load-port relief only;
bandwidth already 1×. Implement ONLY if the window implicates Q6_K
specifically (its curve regresses while expanded encodings do not).

## 3. Oracle, tournament, evidence ladder (unchanged disciplines)

- **Byte-auth oracle**: the existing serial reference is the oracle —
  every encoding × Rows × M (1..16/31/33 tails) × N tails × workers 1..20,
  byte-identical outputs vs serial (the restructure changes no arithmetic;
  the FP32 per-block program, contract(off)/reassociate(off), and the
  partition tree are preserved verbatim). `PanelTailsMatchSerial` and the
  exact-batch contract (CPUExecutionExactBatch.inc) run as preflights.
- **Tournament integration**: the installed tournament
  (`LLAMINAR_PANEL_POLICY=native|expert`) re-runs per encoding × Rows × M ×
  workers for any touched leaf; the successor must win or tie with budgets
  only down; the witness census re-runs (0 new/0 gone).
- **Evidence ladder**: M1 core-free (portability project, the 18-cell
  contract) + x86 syntax sweep → .80 focused via arm-perf booking →
  engine E2E byte-identity + economy per A1 → handover to Main for the one
  campaign.

## 4. Window decision tree (the gate)

- **Regression gone at bb893cd79** (Q8_K 8→10 ≤ 0, knees ≥ 8–10 across
  encodings): measured-and-negative — the C4 hoist already fixed it;
  idea-04 closes fully; §1/§2 are NOT implemented (no speculative work).
- **Regression persists, uniformly across encodings**: the reload hypothesis
  is DISCONFIRMED for the production leaf (activation already 1×). Successor
  shifts to the weight-stream/K-walk suspects, chosen by the CSV shape:
  (i) knee ≤ 4–8 with flat gops top → serial K-walk latency bound → the
  partition-shape successor (partitions_ tuning at M=1, order-exact, oracle
  unchanged); (ii) gops declining with width beyond the knee → weight-stream
  bandwidth → weight-layout/prefetch successor (layout change = new witness
  census + packer proof, higher risk, plan first); (iii) knee at 8–10 with
  cliff → scheduling/placement → route to idea 05/26 owners with the data.
- **Q6_K-specific regression only**: §2 (marginal, cheap, same oracle).

## 5. Sequencing

Window (booked) → summary.py decision table → if §1 or §2 triggers:
implement on a Lane-2 branch off bb893cd79 (codex/<desc>), commit-first,
M1 core-free + syntax, then the ladder in §3. If measured-and-negative:
record in the tracker (existing artifacts only) and close idea-04.

Source line references at bb893cd79: run() 523–660, consume() 355–368,
consumePair() 377–398, activationGroup() 665–671, activationBlocks()
674–687, panel() 706–800+, dotLane/activationBlock in
CPUNativeNEONMultiScaleOps.h 62–128, panelStrips() 182–200.
