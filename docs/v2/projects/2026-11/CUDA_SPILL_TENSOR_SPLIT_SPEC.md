# CUDA spill rework — tensor prefill producer/consumer split (design spec, rev 2)

Status: **spec revision 2 for review** (rev 1 rejected: arithmetic errors, invalid
register accounting, incomplete fragment ABI, barrier/timeline gap, and the
kRows-dependent V-shift parity gap — all addressed below).
Scope: compute capability **12.x only**; all other targets bit-identical (host
CC dispatch; the kernel variant joins graph-capture identity). TARGETS: sm_120
and sm_121 are distinct targets — every number below is measured per target, and
compiler-resource limits are bound per target (no cross-SM assumptions).

## 1. Corrected geometry (reviewer point 1)

Owner split per 16-row group: **1 producer warp + 2 consumer warps** (each
consumer owns one 128-column half of the group). CTA = 4 groups = **64 physical
rows**, 12 warps = **384 threads**.

| Role | Warps | Owns |
|---|---|---|
| Producer | 4 (one per group) | rows `g*16..g*16+15`, complete QK dot + softmax path, per-tile maxima/denominators (running state), `probability_shift`, `pa` fragments. **No output accumulator.** |
| Consumer A | 4 (one per group) | same rows; output columns `c*8 + lid*2 (+1)`, c = 0..15 (cols 0..127) |
| Consumer B | 4 (one per group) | same rows; output columns c = 16..31 (cols 128..255) |

Bijective output map: every output element (row, column) has exactly one final
writer (consumer A or B by column half); `target[n*8 + lid*2 (+1)]` indexing is
unchanged. Sole maxima/denominator publisher: the group's **producer** (only the
producer's warps evaluate the quad shfl reductions; consumers read published
scalars — never recombine).

Full output coverage: 4 groups × 16 rows = 64 rows; grid `gridDim.y =
ceil(M/64)` covers the full M. (Rev-1 error fixed: 8 producer warps only cover
128 rows with **16** consumer warps — 24 warps/768 threads; rejected below in §2.)

## 2. Register accounting (reviewer point 2)

CUDA register allocation is kernel-wide (no per-warp budget). CTA = 384 threads,
1 CTA/SM: per-thread ceiling = `floor(65536/384) = 170` (verify per target with
`-v`; ptxas allocation granularity applies, e.g. per-warp quantized allocation —
state the measured value not the theoretical floor).

| Role | Peak live estimate (to measure) | Fits 170? |
|---|---|---|
| Producer | QK/softmax path only: score (8 f32), pa (4×b32), dimension-loop addressing, quad-reduction state, published-state staging ≈ **accept ≤ 170** (no acc) | measure |
| Consumer | acc 64 f32 + PV-mma window (b 2×b32, h/half, value 4 f32, ldmatrix addresses ≈ 40–60) ≈ **≈ 124–150** | measure |
| Kernel-wide worst case | max(producer, consumer) — both roles coexist in one kernel ⇒ ceiling 170 applies to both | must be ≤ 170 |

Rejected alternatives: 8 groups (8P+16C = 768 threads → ceiling 85, producer
does not fit); 16 warps (8P+8C with full-width consumers → no acc reduction).
If the measured consumer peak exceeds the ceiling, the fallback is **not** an
automatic 4-way split: it is a separate offline design candidate with its own
ownership/BOM (reviewer point 5: no automatic failure fallback).

## 3. Fragment identity + shared BOM (reviewer point 3)

### 3.1 Current fragment/reduction identity (from source, kKeys=16, kWidth=256)

- Per producer lane: `score[kKeys/8][4]` = `[2][4]`; element `score[n][d]` is the
  **complete** QK dot (all 256 dims, accumulated via the `#pragma unroll 1`
  dimension loop) for `(row, key)` where `d<2 ⇒ row0, d≥2 ⇒ row1` and
  `key = key_start + n*8 + 2*lid + (d&1)`. Full dot, not a partial.
- Per (row-pair, n): `maximum0/1` then quad `shfl_xor` 1,2 → per-row-pair max
  (4-lane quad = same gid, varying lid = different keys). Denominator:
  `sum0/sum1` via `shfl_sync(score[group][col], source_lane, 4)` summed over
  source_lane 0..3 — the canonical per-tile denominator, no scalar P/V.
- `pa[kKeys/16][4]` = `[1][4]` per lane: `pack(p0,p1)`, `pack(p2,p3)` —
  FP16 pair halves for (row0,key pair) / (row1,key pair); consumed by the PV
  mma in the **same lane-quad layout** (`mma.m16n8k16(pa[j], b)`).
- Per tile scalars: `shift`/`down` (V half scale), `up`
  (`(127+shift)<<23`), `next0/1`, `alpha0/1`, `probability_shift0/1`,
  `probability_up0/1`; per element `inverse_power = 2^(shift - prob_shift)`.

### 3.2 Published ABI (per group, per K/V stage, per tile) — exact values

| Item | Size | Publisher → consumer |
|---|---|---|
| `down` (half2 bits) | 4 B | producer → both consumers (uniform) |
| `pa[1][4]` per producer lane | 512 B / group | producer lane L → consumer lanes with the same quad position (direct mirror; both consumer warps of the group read the identical buffer) |
| per row-pair (gid 0..7): `alpha0/1`, `inverse_power0/1` | 8 × 16 B | producer (quad reduction results) → both consumers |
| (`next0/1`, `probability_shift0/1`, `probability_up0/1` remain producer-internal) | — | — |

Allowed shared total (12.x variant): Q operand `64×256×b16 = 32 KiB`; K/V
slots `2 × (2×16×256×b16) = 32 KiB`; pa buffers `4 groups × 512 B × 2 = 4 KiB`;
row-pair state `4 × 8 × 16 B × 2 = 1 KiB`; ranges/misc 64 B ⇒ **≈ 69 KiB +
alignment** at 384 threads. Dynamic shared is requested via the existing
`extern __shared__` interface; the per-target opt-in limit is verified against
**each** 12.x target's `cudaDeviceProp.sharedMemPerBlockOptin` (no assumed 227
KiB — sm_120 and sm_121 are measured separately), then the static+dynamic
combination and the `cudaFuncAttributeMaxDynamicSharedMemorySize` are set
explicitly per target.

Bank layout: pa buffer stride padded so the 4-word per-lane rows land on
distinct banks for the two consumer warps' simultaneous reads; provide the
concrete stride and prove no same-bank pair across the 8 groups' readers with
the actual addresses (bank = `(addr/4) % 32`), one uniform barrier per tile
(§4). "Even banks" is not a proof — the proof is the explicit stride/address
table (to be produced with the measurement pass).

## 4. Tile-by-tile timeline (reviewer point 4)

`P` = pa/state slot, `KV` = K/V stage (2 slots), both double-buffered.

```
PROLOGUE      all: issue(stage0) [copy K/V s0, publish ranges]  waitAllCopies; __syncthreads()
STEADY s      producer: QK/softmax on KV[s%2] -> publish P[s%2]
              __syncthreads()
              consumers: read P[s%2], ldmatrix V[KV[s%2]], mma, acc
              all (tile+1 exists): issue(stage t+1) into KV[(s+1)%2]   [copy overlaps consumer compute]
              waitAllCopies; __syncthreads()   [slot reuse: KV[(s+1)%2] written only after
                                                 consumers finished reading KV[s%2]; P[(s+1)%2]
                                                 written only after consumers finished P[s%2]]
DRAIN         last tile: consumers finish; __syncthreads() (partition boundary as today)
PARTITION END existing final write per partition; barrier behavior unchanged
```

Slot-lifetime proof obligations: (a) KV[(s+1)%2] overwrite happens after
consumers' ldmatrix of KV[s%2] (the existing two-slot schedule, preserved);
(b) P[(s+1)%2] producer write happens after the `__syncthreads()` following
consumers' read of P[s%2] — this is the added publication barrier; (c) producer
never runs ahead of the consumer of the same stage by more than one slot.
Masked work: every CTA barrier remains converged; all warps execute every
`__syncthreads()`; per-row inactivity only guards arithmetic (same `key_start >
query` guard), never barriers or MMA participation.

## 5. Parity: the kRows/`shift` gap (reviewer point 5) — explicit resolution

The 12.x geometry changes `kRows` 128 → 64, hence `first_live`, `end`, the
per-stage `largest`, `shift` and `down`/`inverse_power` **can change** for
boundary tiles. Direct contamination would change output bits. Resolution:
keep the **logical tile/range/partition decisions independent of physical
ownership**:

1. **Scores/probabilities are row-local and CTA-independent**: `score`,
   `probability_shift` derive only from (row, key) data and the per-CTA range
   (`largest`) **only** through `shift`/`down`/`inverse_power`.
2. **Power-of-two exactness**: `down = 2^(15-shift)`, `inverse_power =
   2^(shift-prob_shift)`, `up = 2^shift`; all FP16 packing rounds the mantissa
   with exponent-invariant RN (round-to-nearest-even) **while the scaled value
   stays in the normal FP16 range**; the elementwise FMA order and argument
   values are unchanged. Any shift value that keeps every scaled operand in
   the normal range therefore yields **bit-identical** outputs.
3. **Make the shift CTA-independent by construction** (12.x variant): replace
   the data-dependent `largest` selection with the **format-derived bound**
   `largest_split = 0x7BFF` (max normal half exponent field). This is a valid
   headroom bound for every subset CTA (data max ≤ format max), so:
   - the chosen `shift` is a compile-time constant for the variant,
   - `V*down ≤ 1` in normal range (no overflow),
   - `p*up` unchanged (score-based, not range-based),
   - every scaled operand stays normal ⇒ exact ⇒ **byte-identical** to the
     original for every tile, including boundaries — no tile depends on the
     physical row extent of its CTA.
   The proof is mechanical: monotone `shift(largest)`, bound ≥ data max,
   exponent-invariant rounding, and the existing compensation identities;
   the implementation validates it with the parity/E2E gates + a direct
   before/after output-bytes comparison on identical inputs (all geometries
   and boundary tiles), plus captured byte/replay tests (reset/dirty/partial)
   registered in `ProductionTestPreflight`.
4. `issue()`/ranges remain the CTA-wide copy publication (unchanged roles; the
   `ranges` buffer is still produced — the range value is certified for the
   diagnostic path, while the 12.x shift uses the format bound).

## 6. Constraints

- No relaxed byte contract, no allowlists, no arch omission, no disabled
  candidate, no relabeling; NATIVE_QUANTISED candidate gate (nonzero local memory
  rejected) applies to the VNNI kernels; the attention kernel must also show
  **zero** ptxas stack/spills and zero lmem accesses in the hot loops.
- Evidence per target (sm_120, sm_121, plus each shipped non-12.x target as
  control — unchanged schedule): ptxas -v (regs/stack/spills), annotated SASS,
  source+geometry+compiler/flags binding, ncu occupancy, unprofiled Release
  timings; artifacts linked outside tracked dirs.
- Register estimate table above is a design estimate; measured values replace
  it; if measured consumer peak > 170 → stop, report, do NOT fall back
  silently (4-way split is a separate spec).

## 7. Open items

- [ ] Measure per-role peaks/shares with `-v` per target; produce the
      bank-stride table and the opt-in shared limit per target.
- [ ] Confirm exact fragment/lane quad mapping during implementation against
      `mmaFragmentRow/Column` (the ABI above is from source reading; the
      measured table must match).
- [ ] BK64/BK256-ordered M/N-ownership specs: separate documents (no
      inheritance from this spec or the BK256 safe-rolls).
