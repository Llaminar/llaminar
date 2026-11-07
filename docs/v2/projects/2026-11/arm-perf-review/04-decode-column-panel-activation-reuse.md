# Idea 4 — Wider decode column panel + activation reuse

Status: **Open** (inner kernel unchanged since `ebe8da586`; the workshare
improvements in `33690f9b1`/`408ff90e5` address the outer task structure, not
the tile K traversal).

## Context

`CPUNativeNEONPreparedProjection.h::run()` (~:254):

- Outer loop over 4-column strips; inner loop over K blocks (32 values) and 8
  weight groups.
- `consume()` (~:219) reloads/broadcasts the 4-byte activation word **per row,
  per group, per column strip**, so the same 32-byte `Q8_1Block` payload is
  touched up to 16 times per block. The activation scale conversion
  (`nativeVNNIFP16ScaleToFP32(a.d)`) and `sum_qs` are recomputed per strip.
- For M=1 there is one int32 dot chain and one f32 accumulator per strip that
  are live at any time; the strip walks K serially, so latency of the dot
  chain (not issue width) bounds decode. (The "three registers" shorthand in the
  review is not a complete ISA/PMU finding — weights, LUT/planes, factors,
  corrections and reduced partials also live.)

Candidate changes:

1. Wider N panels (e.g. 16–32 columns) with several independent accumulator
   chains when M is small. Each column keeps its own fixed K/block/partition
   program, so arithmetic per column is unchanged; the work per byte re-read
   rises. The existing `#pragma GCC unroll 1` on Q6_K exists only to stop
   hoisting eight decoded groups into a spilling Rows=4 live set — it is not
   evidence against Rows=1 wider panels.
2. Load the 32-byte activation payload once per block into two registers and
   feed lane-indexed dot (`vdotq_laneq_s32`/ASIMD equivalent); reuse the
   metadata (scale, `sum_qs`) per panel. Note `Q8_1` has metadata in addition
   to the 32-byte payload, and Q6_K has low/high scales whose per-block FP32
   order must survive.
3. Both ASIMD (baseline) and DotProd variants need real implementations; byte
   auth per encoding.

## Exactness / risk

- Low but real: any change must keep each column's exact block/group order and
  per-block FP32 rounding (backend-native FMA vs GPU-aligned round per block).
- The Q6_K low/high scaling path and `WithMinimum` correction path are the
  sensitive ones.
- No current A/B bandwidth evidence establishes the winner; this needs the
  harness (idea 1) before and after.

## Validation

- Panel tournaments per Rows × encoding (Q4_K/Q5_K/Q6_K/Q8_0/IQ2_S/IQ3_S/
  IQ4_XS/MXFP4), M 1..16/31/33, workers 1..20, byte-auth vs serial, then
  fused bundle and engine confirmation.

## Coordination

Must be one shared change with the PR #35 MXFP4 lane (`NibbleMap::MXFP4` uses
the same tile); certify all `NibbleMap`/scale representations on the combined
source.
