# Idea 08 — ARM prepared-weight layout: scoped plan (NOT implementation)

Status: PLAN for coordinator review. ABI-bearing, deliberately last; this plan
argues a two-phase path: a cheap measurement package first, an ABI change only
behind measured numbers.

## Outline (the five decisions the plan asks arm-perf/Main to bless)

1. Split idea 08 into two independent sub-changes: (A) dead-byte elimination
   (the 128 B `comp_int16` x86 compensation section in ARM-prepared strides),
   (B) ARM-native compact encodings (nibble payloads stay 3/4-bit; `vqtbl`
   decode) for the expanded formats, highest value IQ3_S (9.5 → ~3.6 bpw).
2. ABI + migration handled by one mechanism: the prepared-layout identity moves
   into the prepared cache/wire/BOM keys, versioned per CPU instruction family;
   mixed-version ranks are refused at admission, persistent prepared artifacts
   are invalidated (canonical migration = re-prepare from GGUF; GGUF stays the
   interchange format).
3. Phase A (measurement-only, no ABI) reports the real numbers: exact BOM
   audit of today's layouts per format and model, plus a diagnostic
   (non-installed, rule-166-conformant) compact-decode kernel tournament which
   is all that is needed to bound the IQ3_S hypothesis.
4. Phase B (ABI) is a conditional GO, gated on measured thresholds
   (engine decode ≥ +5% on a real model, or UMA capacity gain ≥ 5 GiB on the
   35B/128 GB GB10 pool), and on idea 16's FFN/LM-head attribution remaining
   weight-bandwidth-bound.
5. Effort ≈ one R2-scale lane (larger than C4, smaller than R1+register-gate),
   much of the qualification harness already exists (seal, width/panel byte
   fixtures, register gate, A1 engine ABBA); the one genuinely new kernel piece
   is the compact IQ3_S decode (vqtbl4 + source scale).

## What the change actually is

Today the NEON path reuses the x86 VNNI 64x32 prepared layout unchanged:
per (N-chunk, K-block) unit the stride carries a 128-byte `comp_int16`
compensation section used only by the x86 `u8xs8` path
(`sum((qa+128)*qw) - 128*sum(qw)`), which ARM signed-SDOT never reads.
Consequences from the tracker's own audit:

| Encoding | Stride (B / N-chunk / K-block) | Dead share |
|---|---|---|
| NibbleLUT (Q4_0, IQ4_NL) | 1280 | 10.0% |
| ExpandedInt8 (incl. rotated) | 2304 | 5.56% |
| IQ3SExpandedSourceScale | ~9.5 bpw vs 3.44 bpw source | largest single lever |
| Q6KNativeDualScale / CompactMultiScale | own metadata | (optionally harmonized) |

Sub-change A: remove the dead section from the ARM layout for every encoding
where the review found no ARM reader; per-encoding mutation tests prove the
zero-reader claim before each removal.

Sub-change B: keep nibble/LICENSE-grid payloads packed. Q4-family nibbles are
already packed (NibbleLUT). The expanded cases (IQ3_S source-scale, rotated
ExpandedInt8, IQ5/IQ6 families) currently materialize decoded/expanded bytes
(prepared-bpw 9.5 for IQ3_S vs 3.44 in the GGUF). An ARM-native block keeps
the raw 3-bit indices (or 4-bit LUT indices), decodes with `vqtbl` (barrel
LUTs, no per-element ALU widening chain), and applies scale/base in FP32
during accumulation — the same arithmetic consumers, different bytes.

## ABI break and migration story

- One new typed identity: prepared-layout version, belonging to the CPU
  instruction family (NEON vs AVX2/AVX512 never share prepared bytes in one
  mesh). It enters: `preparedEncodingForSource` + stride/footprint + BOM +
  prepared-weight cache keys + wire/slot/movement identity + evidence hashes.
- Same-version cross-rank movement: unchanged (move operations carry prepared
  bytes between equal-identity ranks; the seal still guards liveness).
- Mixed-version meshes: fail fast at rank admission (startup publication
  carries the layout identity; mismatched ranks refuse with the migration
  instruction instead of decoding garbage).
- Persistent prepared artifacts (disk caches, corpora): keyed by layout
  version; a version bump invalidates and re-prepares from GGUF — the GGUF
  is already the single interchange format, so there is no byte-level
  converter to write. Replay fixtures record inputs, not layouts; they are
  unaffected apart from the fixture that exercises the new viewer.
- The x86 path is untouched (its layout keeps the compensation section; the
  two families version independently).

## Projected gains (hypotheses + the measurement that would prove each)

| Hypothesis | Where the win comes from | Bound / expectation | Phase-A measurement shape |
|---|---|---|---|
| A: dead bytes | Stride shrinks 5.6–10%; prepared bytes are the per-token weight traffic at M=1 | If weight-bandwidth-bound: ~+5–10% decode tok/s on Q4/Q8-class models | Exact BOM audit first (static); then kernel tournament with a diagnostic compact layout, one candidate per launch |
| B: compact IQ3_S | 35B UD-IQ3_S FFN+router prepared bytes drop ~60% | Weight-bound decode could plausibly gain +20–60% engine decode on the 35B; M=4 verifier rows may partially give back ALU-side | Same tournament, both M=1 and M=15 grouped rows; then the standard engine A1 ABBA |
| Capacity (idea 2) | GB10 is a 121 GB UMA pool shared CPU+GPU; peak 35B RSS 35.9 GiB today | ~10% prepared-size cut frees GBs for KV/context or a bigger resident model; for compact IQ3_S the cut is far larger | plan/capacity admission audit on .80: same model with fixed budget, report freed bytes → admitted context/KV delta |

Floor of credibility: R2's A1 engine runs on the real 35B/27B models exist
already, so "prepared-vs-GGUF resident ratios are unmeasured" in the tracker
is stale — the audit is runnable today with a Release build and the BOM.

## Cost to build and qualify

- Packer/view/footprint: new stride math + (for B) two new decode walks.
  The 64-column chunk / 32-row block grid and per-block arithmetic stay
  untouched, so the C4 panel programs apply unchanged — only their indexing
  arithmetic and metadata loads move.
- Seal: the 64-lane padding-normalization contract must be re-specified per
  encoding (padding region moves with the layout); the generation-bound seal
  and its fatal path are unchanged machinery.
- Tests/gates, all existing and familiar to this lane:
  byte-auth fixtures per format (PanelWidths/Tails-style sweeps cover any
  new encoding automatically — they iterate the whole catalog),
  metadata-seal coverage + x86-reader padding tests, BOM/footprint tests,
  wire-codec + migration tests, engine A1 per-format byte-identity + economy.
- Register gate: new encodings add programs to the canonical inventory; each
  needs a committed budget (single-row zero spills, grouped ceilings), and the
  full 18-cell canonical re-campaign runs through the NEW coordinator + the
  18x14 rule-166/167 receipts. Budgets only downward.

## GO/NO-GO

- **Phase A: GO now.** No ABI, no layout change; one lane-week. Produces the
  BOM audit and a diagnostic compact-decode tournament bound, and feeds from
  tooling this lane already owns (fixture + tournament + A1 harness).
- **Phase B: conditional GO.** Proceed only if Phase A (plus idea 16's
  attribution when its window lands) shows:
  1. FFN/router weight traffic stays the binding limit at 35B decode (43% +
     35% of the step today), and
  2. the dead-byte or compact-tournament bound clears ≥ +5% engine decode
     (A) or ≥ +15% projected 35B decode (B), with the M=4 ALU check passing.
- **NO-GO triggers:** attribution shows scheduler/latency (not bandwidth)
  binds the step; the M=4 verifier ALU cost eats the bandwidth win; the
  mixed-version/migration machinery surfaces a cross-lane cost (wire codec,
  disk cache) larger than the win.

## Sequencing

Deliberately after the register-gate + coordinator land (this delivery), so
Phase B programs have their hard gate from day one. Idea 2 consumes the
capacity result; idea 16's attribution is the decision input for Phase B's
gate; idea 17's bandwidth-gap analysis shares the microbench data.
