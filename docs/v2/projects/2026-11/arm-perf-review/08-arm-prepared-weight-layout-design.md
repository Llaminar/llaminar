# Idea 08 — compact ARM prepared-weight layout: design

Status: **Design complete — reframed after the sealed tournament result;
implementation gated on an operator capacity-value call** (arm-idea08,
operator-authorized 2026-11-12T20:58Z). Pinned to `d5e04fdce` (PR #43 tip;
this lane's read-only base).
All `file:line` citations resolve at that commit unless prefixed with another
SHA. **§10 is the tiered-residency addendum (gate context, design-only,
2026-11-12T23:37Z operator request) — no implementation authorization;
§10.1 is corrected by §10.10 (premise error, seq 51) and §8 carries the
queued Phase-1 entry items 0–2 (reviewer spine findings).**

Inputs of record:

- **Corrected census (sealed ~20:03Z, K14-I08A-CENSUS-5C2CEF47 / SUMMARY-v2):**
  the 35B (Qwen3.6-35B-A3B-MTP-UD-IQ3_S) prepared quantised weights are
  **26.62 GB predicted vs 15.34 GB source = 1.73×**. Carrier:
  **CompactMultiScale `ffn_gate/up_exps` (80 tensors, 40/41 layers):
  6.88→16.11 GB = 2.34×, +9.2 GB**, and **NibbleLUTSourceScale `ffn_down_exps`
  (37 tensors): 5.28→6.83 GB (+1.55 GB)**. Q6KNativeDualScale is genuinely
  native (1.06×); IQ3SExpandedSourceScale (blk.39) is negligible. Estimator-based
  (predicted labeling; `preparedFootprintForFormat` byte-equality is the tested
  production invariant and the normative anchor). Observed per-tensor
  enumeration remains a post-merge debug seam — not built here.
- **Tournament seal (K14-I08A-TOURNEY-3B109793C-20261112, 9/9 green,
  reviewer-ACK-pinned bench, first valid timing)**: for the IQ3_S family,
  decoding from a compact source-scale layout (4.25 bpw) is **7.0–8.6×
  slower** than reading the pre-expanded prepared grid (9.5 bpw) across
  ASIMD/DotProd/I8MM at M=1/4/15 (e.g. I8MM M1: 4.02 µs expanded vs 34.8 µs
  compact; M15: 34.6 vs 271.0), despite the compact form reading 2.2× fewer
  bytes. **Decode at these shapes is dequant-compute-bound, not
  bandwidth-bound.** The "compact the layout to win decode bandwidth"
  premise is measured-and-negative for the IQ3_S family; this design claims
  capacity only, never decode speed (§1.5, §2.3).
- **Repaired tournament bench (reviewer ACK @ `3b109793c`,
  `codex/arm-idea16-tournament-repair`)**: the per-invocation observability
  standard this design's bench plan follows (§7).

Companion idea doc: `08-arm-prepared-weight-layout.md` (same directory). The
review-head file names (`CPUNativeVNNI*.h`) were consolidated before
`d5e04fdce` into the `CPUNativeQuantised*` names cited here.

---

## 1. What the expansion bytes actually are

### 1.1 The prepared unit

Every CPU NativeQuantised projection is stored as one interleaved stream of
**units**: 64 output columns × one 32-value K block
(`CPUNativeQuantisedWindowPacking.inc:61-64`, `CPUNativeQuantisedWeightView.h:445-448`).
`data_stride` = payload bytes of a unit; `interleaved_block_stride` = total unit
bytes including inline metadata. Both come from exactly two constexpr functions
(`CPUNativeQuantisedPreparedFootprint.h:69-110`) — the single layout authority —
and every consumer re-derives or validates against them (§6).

Unit = 2048 logical weights, so stride in bytes ÷ 256 = bits per weight.

### 1.2 Section anatomy per encoding (canonical layout, at `d5e04fdce`)

Writes: `CPUNativeQuantisedWindowPacking.inc:168-213` (superblock/native path),
`CPUNativeQuantisedWeightPacker.h:360-400` (`packExpandedInt8Direct`),
`CPUNativeQuantisedWeightPacker.h:150-330` (deferred repack). Reads:
`CPUNativeQuantisedWeightView.h:317-364` (`chunkScales`/`chunkMins`/
`admittedScales`/`admittedMins`/`chunkEffectiveMins`), `interleavedB`
(`:372-381`).

| Encoding | Unit sections (canonical) | Stride | bpw |
|---|---|---|---|
| NibbleLUT (Q4_0/IQ4_NL/Q4_1) | `[0,1024)` nibble payload (16 B/col, all live) · `[1024,1152)` **int16 compensation ×64** · `[1152,1280)` fp16 scale ×64 · `[1280,1408)` fp16 min ×64 (asym only) | 1280 sym / 1408 asym | 5.0 / 5.5 |
| NibbleLUTSourceScale (IQ4_XS) | `[0,1024)` nibble payload · `[1024,1152)` **comp ×64** · `[1152,1280)` fp16 base ×64 · `[1280,1408)` raw coefficient ×64 | 1408 fixed | 5.5 |
| ExpandedInt8 | `[0,2048)` int8 payload · `[2048,2176)` **comp ×64** · scales · mins (asym) | 2304 sym / 2432 asym | 9.0 / 9.5 |
| IQ3SExpandedSourceScale (IQ3_S) | `[0,2048)` decoded grid · **comp** · half base ×64 · raw nibble ×64 | 2432 fixed | 9.5 |
| Q6KNativeDualScale | `[0,1024)` payload · `[1024,1536)` high-bit planes (512 B) · scales `[1536,1664)` · signed coefficients `[1664,1792)` | 1792 | 7.0 |
| CompactMultiScale (Q3_K/Q2_K/IQ2_S/IQ2_XS/IQ1_M) | `[0,1024)` **fixed 4-group payload region** · primary fp16 `[1024,1152)` · secondary `[1152,1280)` · emins uint32 ×64 `[1280,1536)` | 1536 | 6.0 |

### 1.3 The dead bytes, family by family

**(A) CMS payload-region padding.** The CMS payload region is a *fixed*
`data_stride = 1024` (4 groups × 256 B) regardless of the source payload width
(`preparedDataStride`, `CPUNativeQuantisedPreparedFootprint.h:69-76`). The
writer places `payload_bytes × 64` live bytes at `(byte/4)*256 + column*4 +
byte%4` (`WindowPacking.inc:169-170`); the rest of the region is zero ("Unused
bytes are zero, not another ledger", `PreparedFootprint.h:106-107`). Live
payload per unit by codebook (`NativeQuantisedFormatInfo.h:198-210` catalog;
payload bytes per 32-value block):

| Codebook | Format | payload_bytes | live payload | dead padding |
|---|---|---|---|---|
| 9 | Q3_K | 12 | 768 B | 256 B |
| 10 | Q2_K | 8 | 512 B | 512 B |
| 13 | IQ2_S | 9 | 576 B | 448 B |
| 14 | IQ2_XS | 9 | 576 B | 448 B |
| 17 | IQ1_M | 6 | 384 B | 640 B |

**(B) CMS emins section.** 256 B per unit of `uint32` effective minima at
`data_stride + 256` (`WindowPacking.inc:173`, view `:357-364`). Only codebook
10 (Q2_K) reads it (`MultiScaleKernels.h:668` `chunkEffectiveMins`, gated by
`codebook_id == 10` in the scalar reference `:310-316`; `has_emins` is true
only for Q2_K in the catalog). For Q3_K, IQ2_S, IQ2_XS, IQ1_M the entire
256 B is zero-filled dead weight.

**(C) Compensation section, ARM-dead.** The 128 B int16 compensation vector
exists for x86 `u8xs8` arithmetic: `sum((qa+128)*qw) − 128*sum(qw)`. It is
written for every non-CMS/non-Q6K encoding (`WindowPacking.inc:197`,
`WeightPacker.h:620-655`), and read by x86 AVX2/AVX-512 GEMV/GEMM
(`CPUNativeAVX2Gemv.h:173-178,325-330,492-496` etc.;
`CPUNativeQuantisedGemv.h:1589,1792,2143,2455,2840,3635,3882,4121,4573,4729,4881,5042`).
The NEON projection never reads it: ARM SDOT/SMMLA compute signed×signed dots
directly, and the asymmetric correction uses the **activation's** `sum_qs`
(`CPUNativeNEONPreparedProjection.h:631-637` `accumulateCorrectedSingleScale-
BlockNEON` with `a.sum_qs`, panel form `:827-833`), not the weight
compensation. The idea-08
review's mutation-test plan (no NEON reader of comp per encoding) is confirmed
structurally at this head. Dead cost: NibbleLUT 1280 B stride → 10.0% dead;
NibbleLUTSourceScale 1408 B → 9.1% dead; ExpandedInt8 2304 B → 5.6% dead.

**(D) Scale/secondary replication.** CMS and NLSS store primary (and
secondary) words **per column per 32-value block** (`chunkScales(chunk, kb)`,
`WindowPacking.inc:171-172,210-212`), while the source formats store one
factor set per 256-value superblock (8 blocks). E.g. IQ4_XS source: 1088 B/unit
(`native_block_bytes_for_format` = 136 B per 256 values,
`WeightPacker.h:76-100`) vs 1408 B prepared: +128 B dead comp, +112 B base
replication (2 B per 64 columns per block vs 2 B per 8 blocks), +80 B
coefficient widening (6-bit source scale → uint16).

### 1.4 Per-family expansion table (unit = 2048 weights)

Source bytes per unit = `native_block_bytes_for_format × 8` for superblock
formats (256-value blocks), `× 64` for 32-value blocks:

| Format (cb) | Source B/unit | Prepared B/unit | Expansion | Anatomy |
|---|---|---|---|---|
| **IQ2_S (13)** | 656 | **1536** | **2.34×** | 448 pad + 256 dead emins + 128×2 scale repl. |
| IQ2_XS (14) | 592 | 1536 | 2.59× | same |
| Q2_K (10) | 672 | 1536 | 2.29× | 512 pad (emins live) + scale repl. |
| Q3_K (9) | 880 | 1536 | 1.75× | 256 pad + dead emins |
| IQ1_M (17) | 448 | 1536 | 3.43× | 640 pad + dead emins |
| **IQ4_XS (4+s)** | 1088 | **1408** | **1.29×** | 128 comp + 192 repl. |
| Q4_0/IQ4_NL (0/4) | 1152 | 1280 | 1.11× | 128 comp |
| Q4_1 (5) | 1280 | 1408 | 1.10× | 128 comp |
| Q8_0 (19) | 2176 | 2304 | 1.06× | 128 comp |
| Q6_K (8) | 1680 | 1792 | 1.07× | ~native; none addressed here |

The census carrier ratio (2.34×) identifies the 35B gate/up experts as
IQ2_S-dominant, exactly matching the 1536/656 unit arithmetic — the census and
the packer agree without any observed seam.

### 1.5 The tournament result: what it refutes, and what it does not touch

The sealed tournament asked a **decode-program question**: for IQ3_S, is it
cheaper to (A) read pre-decoded grid bytes (9.5 bpw — the production
IQ3SExpandedSourceScale layout) or (B) decode the compact source payload
inline per invocation (4.25 bpw — scalar gathers from the 512-entry grid
plus the sign program, "the irreducibly scalar part on this ISA", bench
header @ `3b109793c`)? Measured answer: (B) is 7.0–8.6× slower per
invocation; decode at these shapes is **dequant-compute-bound**, so the
2.2× byte reduction buys nothing. Two consequences, kept strictly separate:

1. **The measured price does not attach to this design.** This design never
   changes a decode program. The CMS family's production kernel *already*
   decodes inline from the native payload — an existing production choice,
   retained unchanged: IQ2_S/IQ2_XS quarters assemble a 10-bit grid index
   from `word(0)/word(1)` and fetch each eight-byte grid entry with four
   scalar `vld1q_lane_u64` lane loads plus `vuzp` transposes
   (`CPUNativeNEONMultiScaleOps.h:203-222` `gridPair`, driven from
   `CPUNativeQuantisedMultiScaleKernels.h:405-420` and `:476-492`);
   Q3_K/Q2_K use the shift/mask linear decoder (`:340-346`);
   NibbleLUT/NibbleLUTSourceScale decode with one `vqtbl1q` per group
   against the 16-entry IQ4 LUT (`CPUNativeNEONPreparedProjection.h:206-212`
   `nibbles()` — the table fits one 16-byte register, categorically cheaper
   than the 512-entry grid gathers that priced the tournament's compact
   arm); ExpandedInt8/IQ3_S read pre-decoded bytes (the tournament's
   *winning* arm). Compaction here removes bytes those exact programs
   **never load** (§2.1 proves the read set equals the retained bytes per
   family — the reviewer's static pass finds this argument structurally
   sound, seq 53) and adds zero dequant arithmetic. **That is the static
   argument; it is not the performance guarantee. The binding guarantee is
   the §7 measured gate on both ISAs.**
2. **The tournament's bandwidth finding does transfer — as a ceiling on
   claims, not a cost.** CMS inline decode is the same class of per-block
   dequant work (scalar grid lane loads + transposes) the tournament
   measured as dominant. If decode is dequant-bound, removing unread bytes
   is *expected* to be speed-neutral — a hypothesis, not a construction
   guarantee: stride/cache/prefetcher perturbations are real and are
   measured, never argued away. This design therefore claims **capacity,
   not speed**; its performance gate is non-regression under the operator's
   same-or-better bar (§7, §8).

Dead, explicitly: any proposal that trades pre-expanded bytes for inline
decode (the tournament's shape) — measured-and-negative, sealed, not
revisited here. The "~42% reduction as the decode-bandwidth prize" framing
of the original lane brief is retired. The surviving scope is exactly the
post-tournament classes: **(a)** dead-byte trimming inside the current
layouts with the access pattern unchanged — this design; **(b)** the
capacity value of that trim where admission is memory-bound, quantified
honestly in §2.3 (no currently-measured topology is blocked by the 1.73×
expansion; the value is residency headroom); **(c)** nothing that slows the
expanded decode path — no decode program changes anywhere in this design.
If a future capacity emergency ever demanded tournament-style source-byte
residency, the sealed 7.0–8.6× decode price would be the stated, accepted
cost of that separate decision — not of this one.

---

## 2. The compact ARM layout (per family)

Design principle: **compaction changes byte placement and unit extent only**.
Payload bits, metadata words, decode expressions, scale programs, partition
trees, and every FP32 edge are byte-identical to the canonical layout. The
readers' addressing is already `data_stride`-relative (§1.2), so the compact
layout is expressed almost entirely through the two stride functions plus one
view predicate.

### 2.1 Compact unit sections

| Encoding | Compact sections | Stride | bpw |
|---|---|---|---|
| NibbleLUT | payload 1024 · scales +0 · mins +128 (asym) — **comp dropped** | 1152 sym / 1280 asym | 4.5 / 5.0 |
| NibbleLUTSourceScale | payload 1024 · base +0 · coefficient +128 — **comp dropped** | 1280 | 5.0 |
| ExpandedInt8 | payload 2048 · scales +0 · mins +128 (asym) — **comp dropped** | 2176 sym / 2304 asym | 8.5 / 9.0 |
| IQ3SExpandedSourceScale | payload 2048 · base +0 · nibble +128 | 2304 | 9.0 |
| Q6KNativeDualScale | **unchanged** (already native; no comp, no padding) | 1792 | 7.0 |
| CMS Q3_K | payload `ceil(12/4)×256 = 768` · primary +0 · secondary +128 | 1024 | 4.0 |
| CMS Q2_K | payload 512 · primary +0 · secondary +128 · emins +256 | 1024 | 4.0 |
| CMS IQ2_S / IQ2_XS | payload `ceil(9/4)×256 = 768` · primary +0 · secondary +128 | 1024 | 4.0 |
| CMS IQ1_M | payload 512 · primary +0 · secondary +128 | 768 | 3.0 |

Rules, in one place:

1. **Compensation section: absent.** `usesInlineCompensation()` is false for
   every encoding under the compact layout (it is already false for Q6K and
   CMS; `CPUNativeQuantisedWeightView.h:304-305`).
2. **CMS payload region: tight.** `data_stride = ceil(payload_bytes/4) × 256`
   (the 4-byte group interleave granularity is retained so NEON `vld1q` lane
   loads keep their exact addresses — `decodeLinear`'s `word(Group/4)`
   (`MultiScaleKernels.h:340-346`), IQ2_S quarters `word(0)/word(1)/
   sign_byte 5+Q → word(1)/word(2)` (`:405-420`), IQ1_M signs at absolute
   `unit+256` (`:675`), all stay in-range by construction).
3. **CMS emins: Q2_K only** (`has_emins` from the format catalog — the sole
   live consumer).
4. **Q6K: byte-identical.** Its readers use absolute offsets
   (`unit+1024` planes, `NEONPreparedProjection.h:442` and the `q6Group`
   helper `:438-452`; scales/coeffs via
   `admittedScales/admittedMins` at `data_stride+0/+128` with
   `data_stride = 1536`), which the compact layout leaves untouched.

The view/packer offsets that change are exactly the ones already dual-formed:
`chunkScales = data_stride + (usesInlineCompensation() ? 128 : 0)` and
`chunkMins = data_stride + (… ? 256 : 128)` (`WeightView.h:317,330,340,349`);
`chunkEffectiveMins = data_stride + 256` is layout-invariant. The compact
layout therefore *reuses the CMS/Q6K addressing form* for the comp-bearing
encodings. `interleavedB`'s group bound is `data_stride/256` (`WeightView.h:372-381`)
— tight CMS data strides keep every legal group in range.

**Read-set proof (the zero-arithmetic-change invariant, per family).** The
retained bytes are exactly the bytes the production kernels load; the
removed bytes are provably never loaded:

| Family | Production read set within the unit | Removed bytes ever loaded? |
|---|---|---|
| CMS Q3_K (9) | payload words 0-2 (`decodeLinear`: `word(Group/4)` for G<8 plus `word(2)` high bits, `MultiScaleKernels.h:340-352`) · primary · secondary | padding beyond 768 B: no; emins: no (Q2_K-only reader `:668`) |
| CMS Q2_K (10) | payload words 0-1 · primary · secondary · **emins** (`:668-672`) | padding beyond 512 B: no |
| CMS IQ2_S/IQ2_XS (13/14) | payload words 0-2 (index low `word(0)`, index bits + signs `word(1)`, 9th byte via `word(2)` for Q=3, `:405-420/:476-492`) · primary · secondary | padding beyond 768 B: no; emins: no |
| CMS IQ1_M (17) | payload words 0-1 (signs at `unit+256` absolute, `:675`) · primary · secondary | padding beyond 512 B: no; emins: no |
| NibbleLUT / NLSS | payload groups 0-3 (`unit + group*256 + column*4`, `NEONPreparedProjection.h:366-372`) · scales · mins/coeff | comp `[data_stride, data_stride+128)`: no NEON reader (§1.2 C) |
| ExpandedInt8 / IQ3S | payload groups 0-7 · scales · mins/base/nibble | comp: no NEON reader |
| Q6K (8) | payload + planes `[0,1536)` · scales · coefficients | nothing removed |

Every retained load keeps its address expression and value; only the unit
extent and the metadata section base (comp removal) change. This is the
structural reason the tournament's 7.0–8.6× price cannot attach (§1.5).

### 2.2 What is intentionally NOT compacted in L1 (V1) — and the L2 in-plan
headroom

The seq-51 premise correction (§10.10) re-classified the access-pattern
items: the CMS/NLSS hot paths already decode their native payload inline,
so tightening toward source bytes is a **load-shape change, not a program
change** — the tournament's 7.0–8.6× never attaches to it. They are
therefore **in-plan as L2** with their own measurement gates, no longer
out-of-scope:

- **L2-a — per-unit scale planes → superblock dedup (in-plan, Stage C).**
  Primary/secondary stay per column per block in L1. Superblock dedup
  (store the shared base once per 8 units) would cut another ~0.7–1.0 bpw
  for IQ2_S/IQ4_XS — but it is **a new physical identity, not a layout
  variant with a flatter stride**: the shared base changes what a unit's
  bytes mean, and it carries concrete geometry obligations — **K-origin
  alignment** (a unit's shared base depends on its superblock's position
  in K, so a K-windowed or partial root cannot address units
  independently), **partial-superblock handling** (K extents and
  intersections not aligned to 8-unit superblocks), **tail geometry** (the
  final superblock of a row), and **wire/manifest geometry** (the tier
  wire's per-column compact form presumes per-unit planes; §10's movement
  math assumed per-unit scales). L2 enters the plan as Stage C with its
  own measured non-regression case (§7-class bench + byte-equality) and
  its own identity/version work; describing it as "a 656-byte stride" is
  wrong — 656 is a byte total, not a geometry.
- **L2-b — IQ2_S secondary width (in-plan).** The coefficient is a byte
  (`≤ 255`, `native_quantised_iq2s_scale::valid`, seal rule
  `WeightView.h:139`, check sites `:82,:110,:533`) currently stored as
  uint16. Byte-packing saves 64 B/unit (6%) but adds a per-codebook layout
  branch through seal/reader/packer and widens loads — same L2 gate.
- **L2-c — IQ2_S 9th payload byte (in-plan).** The ceil-to-group padding of
  192 B/unit could become a separate 64 B byte-plane section (832 B/unit) —
  a different payload addressing family for the tail byte; same L2 gate.
  The L1 group form keeps one uniform payload addressing family.

L1 (this design) remains exactly the dead-byte trim with the access pattern
unchanged; L2 items are sequenced behind L1's own gate and are not part of
the Stage-A/A0–A4 chain (§8).

### 2.3 Sizing and the honest capacity claim (35B, census carrier split)

- CMS gate/up: 16.11 GB × 1024/1536 = **10.74 GB** (−5.37 GB; IQ1_M-shaped
  tensors would fall to 768, none identified in the census).
- NLSS down: 6.83 GB × 1280/1408 = **6.21 GB** (−0.62 GB).
- IQ3S/ExpandedInt8 outliers (blk.39): ≈ −0.03 GB.
- Q6K (output.weight, token_embd, 3 down tensors): unchanged.
- **Total: 26.62 → ≈ 20.6 GB prepared quantised (−22.6%, −6.0 GB)**; plan-BOM
  total 34.9 → ≈ 28.9 GB. **Capacity arithmetic (corrected, seq 55):** CMS
  units 1536→1024 B = **−33% bytes ⇒ +50% units per admitted byte**; NLSS
  1408→1280 = −9% ⇒ +10%; expert-routed totals 23.9→17.9 GB = −25% ⇒
  **≈ +34–35% resident routed experts per admitted tier**. (An earlier
  draft's “−6.0 GB ≈ 26% more resident experts” wrongly quoted the
  prepared-byte reduction as the capacity increase.) The remaining
  gap to "source-sized" (15.34 GB) is scale-plane replication plus the
  retained group granularity — the L2 in-plan headroom catalogued in §2.2.

**What this capacity is worth — quantified against the coordinator's
question ("which topologies does 26.6 GB block that 20.6 GB admits?"):**

- **No currently-measured topology is admission-blocked by the 1.73×
  expansion.** The GB10 (128 GB UMA) hosts the 35B at a 34.9 GB plan BOM
  with ample headroom; the same holds for the measured serving cells. The
  trim does not unblock any deployment that fails today.
- The value is **residency headroom on memory-bound axes**:
  (i) ExpertOverlay CPU/expert tiers are admitted by bytes
  (`PhysicalMemoryAuthority`, `--expert-tier … memory-mb /
  max-experts-per-layer`): −6.0 GB (−22.6% of prepared; −25% of
  expert-routed) ⇒ **≈ +34–35% more routed experts resident per admitted
  tier at the same byte budget** (§2.3 arithmetic). How headroom converts
  into resident experts and movement is decided by `PhysicalMemoryAuthority`
  admission, migration policy and live demand — **no change in migration
  count or residency is guaranteed by the layout alone**; this is a
  capacity/steady-state effect, **not** a per-token speedup claim; (ii) KV/context headroom (+6 GB) on UMA hosts
  where KV, weights and GPU workspaces share one pool (idea 2's
  live-admission gap is the complementary piece); (iii) multi-model
  packing headroom.
- **Speed expectation: neutral — as a gated hypothesis, not a construction
  guarantee.** Per §1.5, CMS/NLSS decode is dequant-compute-bound inline
  work; removing unread bytes should not remove arithmetic, but
  stride/cache effects are measured, never assumed away. The binding gate
  is same-or-better measured performance (§7/§8); anything slower is
  rejected and the lane reverts.

Because the prize is capacity rather than a blocked-topology unlock, the
lane's expected value is materially lower than the original brief assumed;
§8 places an explicit operator decision gate before any byte-changing
phase, and §9 states the close/continue assessment.

The LM-head census finding (Q6 445 MB × 2 read per token at M=1) is untouched
by this design — it belongs to the idea-16/B4 LM-head lane.

---

## 3. Per-ISA layout identity — x86 bytes unchanged by the host selector;
performance by gate

A typed layout dimension, not a runtime mode:

```cpp
enum class CPUNativeQuantisedLayout : std::uint8_t
{
    InterleavedCanonical = 0,   ///< current bytes: inline compensation, fixed CMS payload region
    InterleavedCompactA64 = 1,  ///< §2: no compensation, tight CMS payload, Q2_K-only emins
};

#if defined(__aarch64__)
inline constexpr CPUNativeQuantisedLayout kHostPreparedLayout =
    CPUNativeQuantisedLayout::InterleavedCompactA64;
#else
inline constexpr CPUNativeQuantisedLayout kHostPreparedLayout =
    CPUNativeQuantisedLayout::InterleavedCanonical;
#endif
```

- **Production packers** (`packWeightsCPUNativeQuantised`,
  `initializeNativeWindowOutput`/`appendNativeWeightGroups`,
  `packExpandedInt8Direct`, deferred repack) derive the layout from
  `kHostPreparedLayout`. An x86 build physically produces canonical bytes;
  no caller can select otherwise — there is no parameter to misuse.
- **Production execution admission** (`CPUNativeNEONProjectionTile::
  requireExecutable` / the x86 GEMV/GEMM admission, `PackedWeights::
  readView`) requires the stream's layout to equal `kHostPreparedLayout`.
  An ARM build cannot execute canonical bytes; an x86 build cannot execute
  compact bytes. Retirement is complete per-ISA; no fallback exists.
- **Estimators take an explicit layout.** `preparedFootprintForFormat(source,
  rotation, layout = kHostPreparedLayout)` — homogeneous serving (planner on
  the serving ISA) is unchanged by default; cross-ISA planning (an x86
  discovery root pricing an ARM CPU tier) passes the target endpoint's layout
  from the typed inventory. The compact tables are inert constexpr pricing
  data on x86 — they never select bytes there.
- **The archive/wire records the layout** (§7) so mismatches are diagnosed,
  never silently reinterpreted.

Operator bar ("x86 performance unchanged"): **byte identity** is selector-
based — every x86 byte-producing and byte-consuming path serves the same
bytes with the same strides (`kHostPreparedLayout` never selects compact
off AArch64; the only x86-visible changes are the added enum, the widened
signatures, and the archive/wire version field), and the x86 byte-equality
suites prove it. Same bytes and same arithmetic are **not** a codegen or
performance identity claim — those are carried by the gates below. The **performance** half of the bar is
gate-based, not construction-based: x86 cells in the §7 bench and the
canonical CI matrix carry it, and any measurable x86 regression reverts the
change.

---

## 4. The full ABI chain that moves in one piece

Every site below touches `preparedDataStride` / `preparedInterleavedBlockStride`
/ section offsets / the stride-derived extent. The change lands as one atomic
branch state; no partial ABI is ever published.

**Layout authority**
1. `kernels/cpu/gemm/CPUNativeQuantisedPreparedFootprint.h` — layout enum,
   host constant, layout-aware `preparedDataStride(layout, encoding, source)`
   and `preparedInterleavedBlockStride(layout, encoding, source)` (canonical
   overloads remain as the x86 authority and the dispatcher's canonical arm).
2. New typed `PreparedUnitSections` (payload extent, has-comp, scales/mins/
   emins offsets, stride) computed by one function; packer, view, seal and the
   wire kernels consume it instead of re-deriving offsets
   (replaces the duplicated `+128/+256` literals at `WindowPacking.inc:171-212`,
   `WeightView.h:280,317-364`, `WeightPacker.h:620-655`).

**Producer**
3. `CPUNativeQuantisedWindowPacking.inc` — `initializeNativeWindowOutput`
   (:55-73) stride/layout selection; `appendNativeWeightGroups` (:88-214)
   layout-conditional comp/scales/mins placement (CMS branch :168-174 is
   already offset-relative and comp-free).
4. `CPUNativeQuantisedWeightPacker.h` — `packExpandedInt8Direct` (:198-412,
   comp/scale/min placement :620-655); deferred `repackNativeBlocksToInterleaved`
   (:120-330); rotation path interleaving (:700+).

**Owner / view / seal**
5. `CPUNativeQuantisedPackedWeights.h` — descriptor gains `layout`;
   `readView` (:207-223) extent/geometry checks; `normalizePaddingMetadata`
   (:232-290) secondary-offset; `preparedFootprint()` (:184-196) carries
   layout.
6. `CPUNativeQuantisedWeightView.h` — `admitStructure` (:422-465) validates
   extent = chunks×blocks×`stride(layout, …)`; `usesInlineCompensation()`
   (:304-305) becomes layout-aware; `CPUNativeQuantisedMetadataSeal::validate`
   (:502-527) scans the metadata plane under the section offsets.
7. `CPUNativeQuantisedPublishedBytes.h` — sealed container identity.

**Execution kernels**
8. `CPUNativeNEONPreparedProjection.h` — no addressing changes beyond the
   view predicate (§2.1); `requireExecutable` (:100-137) adds the host-layout
   admission.
9. `CPUNativeQuantisedMultiScaleKernels.h` — unchanged (all addressing is
   `data_stride`-relative or payload-relative; scalar reference :288-330 works
   through accessors).
10. x86 `CPUNativeAVX2Gemv.h` / `CPUNativeQuantisedGemv.h` /
    `CPUNativeQuantisedGemmKernel.h` — unchanged (host layout = canonical);
    admission rejects non-canonical streams.
11. `CPUQ8ResidualCorrection.h:74-75` — stride validation via the
    layout-aware function.

**Estimators / BOM (lockstep, per the lane brief)**
12. `CPUNativeQuantisedEncoding.h` — `preparedFootprintForFormat` overloads
    (:75-101) gain the layout parameter; `NativeQuantisedPreparedFootprint`
    gains a `layout` field so BOMs carry identity.
13. `CPUWeightPreparationMemory.cpp:45-52` — preparation pricing.
14. `CPUProjectionWorkspaceContract.cpp:37` — workspace contract footprint.
15. `execution/moe/ExpertPreparedMemoryGeometry.h:182` — expert-tier
    prepared-memory geometry (layout = target tier's ISA).

**Cache keys / archive**
16. `kernels/PackedWeightsSerialization.h` — `PACKED_WEIGHTS_VERSION` 7→8
    (:43-57) with a `prepared_layout` header field; v7 = canonical
    interpretation. Loader constructs the descriptor and the owner's
    host-layout check rejects cross-layout archives with a precise
    diagnostic (ARM: "v7/canonical prepared weights are retired on AArch64;
    re-prepare from source").

**Movement / wire (MoE overlay)**
17. `execution/moe/ExpertTierWeightStream.cpp` — manifest gains
    `cpu_layout`; validation (:290-320, :560-615) re-derives strides through
    the layout-aware functions; receiving-side materialization writes the
    host-layout resident unit.
18. `execution/moe/ExpertTierWeightDeviceLayout.h` —
    `ExpertTierWeightCPUStorageLayout` gains the layout field; `valid()`
    (:131-160) checks strides per layout.
19. `execution/moe/CpuExpertSlotPool.cpp:148-187`,
    `MoEOverlayPhysicalResidencyFabric.cpp:911,3271-3272`,
    `MoEOverlayRemoteProjectionProtocol.cpp:206,473,591` — slot geometry,
    residency-fabric stride equality, remote-projection root validation.
20. `kernels/cuda/repack/CUDAExpertTierWeightKernels.cu` and
    `kernels/rocm/repack/ROCmExpertTierWeightKernels.hip` — the CMS
    conversion kernels are already scalar-parameterized
    (`compactMultiScaleUnitsCUDAUnit`, `CUDAExpertTierWeightKernels.cu:160-210`:
    payload at `(byte/4)*(4*columns)`, metadata at
    `(cpu_data_stride/64)*columns` + `2*columns` + `4*columns`) and work for
    the compact layout through the manifest scalars. The nibble/expanded
    conversion kernels (:239-277, :334-389; comp computation :257-268,
    :371-377) currently write the CPU compensation vector and must become
    layout-conditional (skip comp, shift scales to `data_stride+0`). Backend
    symmetry: CUDA and ROCm move together.

**Test surface (updated with the layouts, not after)**
21. `tests/v2/utils/NativeQuantisedPreparedByteFixture.h`,
    `Test__PackedWeightsSerialization.cpp`, `Test__NativeQuantisedFormatInfo.cpp`,
    `Test__CPUNativeQuantisedWeightView.cpp`, `Test__CPUNativeQuantisedMetadataSeal.cpp`,
    `Test__CPUNativeNEONPreparedProjection.cpp`, `Test__CPUNativeNEONExecution.cpp`,
    the parity/format sweeps (`Test__CPUNativeQuantised_GEMV.cpp`,
    `Test__CPUNativeQuantisedWeightPacking.cpp`,
    `CPUNativeQuantisedExecutionExactPolicy.inc`), the expert-tier wire
    contracts (`ExpertTierWeightWireTestOracle.h`,
    `ExpertTierCapturedPackedTransferContract.inc`,
    `ExpertTierRectangularRemoteContract.inc`, the CUDA/ROCm rectangular
    contracts, `Test__ExpertTierWeightStream.cpp`,
    `Test__MoEOverlay*`), and the Perf__ sweeps that read the stream.

Nothing in this chain may ship partially: a stream, archive, slot, seal or
estimator that disagrees with the packer's layout is a fatal inconsistency by
construction (typed field + single authority + admission checks), not a
repaired-after-the-fact condition.

---

## 5. Correctness contract — storage compaction must not change arithmetic

1. **The arithmetic identity is the current production program.** Same payload
   bytes (bit-identical), same metadata words (bit-identical primary/
   secondary/emins values), same decode expressions, same INT32 dot order,
   same FP32 scale program and partition trees. The only permitted
   differences are *where* those bytes live inside the unit and the unit's
   total extent.
2. **Proof obligations per phase (all before any performance claim):**
   - Unit-level: for every cataloged source format and both layouts, packing
     the same synthetic source block produces accessors whose payload/
     metadata words are bit-identical (`interleavedB`, `chunkScales`,
     `chunkMins`, `chunkEffectiveMins` reads).
   - Execution-level: compact-layout production kernels vs the layout-neutral
     scalar reference (`scalarReferenceBlock`,
     `MultiScaleKernels.h:286-334`) for every encoding/codebook, every
     admitted NEON family (ASIMD/DotProd/I8MM incl. the SMMLA row-pair walk),
     M ∈ {1,4,15}-shaped row tiles, symmetric/asymmetric, expert and
     BackendNative policies — byte-identical FP32 outputs.
   - Cross-layout equivalence (diagnostic, performance namespace): compact
     arm vs a bench-local canonical-arm construction of the same source
     weights, byte-identical outputs per sample (the mirrored form of the
     ACK'd tournament bench, whose "prepared" arm was production and compact
     arm was bench-local; post-landing the roles swap because canonical is
     retired on AArch64 — §9).
   - Migration/wire: archive and tier-stream round-trips per layout;
     cross-layout rejection diagnostics (§7).
3. **Stop-and-report conditions.** If any compact stride would force a
   reader addressing change beyond offset selection (e.g. a payload-relative
   address that no longer fits the tight region), or if any format's
   arithmetic cannot be preserved bit-exactly under the compact placement,
   the lane stops and reports — that is a different lane (arithmetic change).
   Nothing of that kind was found during this design pass; all CMS/nibble/
   expanded addressing is `data_stride`-relative or payload-relative
   (verified §2.1).
4. **Serial-row byte equivalence / MTP verifier parity** is untouched: the
   grouped verifier path consumes the same views; compact storage changes no
   reduction order. The all-format/backend sweep continues to certify it.

---

## 6. Migration, versioning, and retirement

1. **Archive (`PackedWeightsSerialization.h`)**: v8 adds `prepared_layout`.
   Exactly **v7 is handled explicitly**: v7 predates the field and its bytes
   are canonical, so a v7 archive is admitted as canonical with the same
   versioned semantics it has today. **Pre-v7 archives stay rejected exactly
   as now**: exactly v7 is admitted, globally — the archive version gate is
   not per-format (`PackedWeightsSerialization.h:325-327`, `:540-544`). On
   AArch64, loading v7 (or v8-canonical) CPU prepared weights fails admission
   with a precise re-prepare diagnostic; on x86 they load as today (canonical
   is x86's *current* layout, not a replaced one). No on-load conversion, no
   dual-reader period.
2. **Tier-stream manifest / remote-projection protocol**: layout field added
   to the manifest identity and hash inputs; receivers re-derive their own
   layout's strides and reject mismatches. The **CPU↔GPU** conversion carries
   a direction-precise layout contract: GPU→CPU demotion selects the
   **destination** CPU endpoint's layout, CPU→GPU promotion selects the
   **source** CPU layout (manifest names it; conversion kernels are
   parameterized by the layout scalars) — this conversion is required and
   part of A2. It is not a CPU↔CPU repacker
   (§6.3: none exists, none authorized).
3. **Cross-ISA CPU tier movement** (x86 rank ↔ AArch64 rank exchanging
   prepared CPU experts) is a typed rejection **as direct prepared-byte
   movement** — no hidden repack path, no fallback; the rejection is
   registered as a focused regression. **No CPU↔CPU conversion lane exists
   in this design and none is authorized by it**; if cross-ISA movement is
   ever required, a CPU↔CPU repacker is a separate, explicitly commissioned
   operator decision with its own economics. This is distinct from §6.2:
   the **CPU↔GPU conversion is endpoint-layout-selected and IS part of the
   built tier ABI** (A2 work). Mixed-ISA clusters keep working for every
   topology that does not move prepared CPU weights across ISAs.
4. **Cache keys**: the prepared store's caches are keyed by binding/device
   and are **not** re-keyed by the archive version bump. Canonical bytes are
   unchanged through Stage A, so existing prepared caches remain valid
   without invalidation. The layout participates in **admission identity**
   (the descriptor comparison of §6.5), not in a version-derived cache-key
   space. Estimator byte-equality preflight (the normative anchor) runs
   per-ISA: on AArch64 it certifies compact strides; on x86 canonical
   strides.
5. **Invalid states unrepresentable**: the layout is a typed descriptor
   field; the packer has no layout argument to get wrong; admission compares
   descriptor layout, strides and extent against one authority; the seal
   binds the layout into the byte-range identity (`admits()` compares the
   full descriptor, `WeightView.h:168-182`).
6. **Complete retirement**: after the AArch64 compact layout is activated,
   no AArch64 production path can materialize, admit, execute or move
   canonical-layout CPU prepared weights **for its own residency**. Cross-ISA
   CPU movement remains the §6.3 typed rejection — there is no CPU↔CPU
   repacker in this design; the endpoint-layout conversion that does exist
   is CPU↔GPU (§6.2, A2). The canonical packer branches for comp-bearing
   encodings remain compiled on AArch64 only where shared constexpr/table
   code requires them (pricing); they are unreachable from production
   entries.

---

## 7. Bench plan (per-invocation observability standard @ `3b109793c`)

Bench = **non-installed diagnostic**, performance namespace
(`tests/v2/performance/kernels/cpu/native_quantised/`), never
`tests/v2/unit`. It follows the reviewer-ACK'd tournament standard
(`Perf__CPUCompactIQ3SDecode.cpp` @ `3b109793c`). The question it answers is
**non-regression, not speedup**: per §1.5 the compact layout must time
within noise of canonical (it executes the identical decode program over
strictly fewer resident bytes); a compact arm measurably slower than the
canonical arm is a lane defect and blocks landing.

1. **Two arms, one microkernel**: production compact-layout NEON projection
   (the installed path, post-landing) vs a bench-local canonical-layout arm
   built from the same source weights (the retirement mirror of the ACK'd
   bench). Both arms share the production decode/dot/scale programs
   (`gridPair`/`decodeLinear`/`nibbles()`, `NEONMultiScaleOps::dot`/
   `pairDot`, `accumulateSingleScaleBlockNEON`/`…Corrected…`, the IQ2_S
   coefficient program) and differ only in their weight-vector source and
   stride/addressing — so any timing difference isolates the layout change
   itself (unit stride and section offsets), nothing else.
2. **Per-invocation observability**: every timed invocation's live output
   span is dirtied with a volatile quiet-NaN sentinel before the clock
   (poison-then-verify); an instruction-free object-bound escape
   `asm volatile("" : : "r"(span), "r"(bytes) : "memory")` at **both** clock
   edges binds dirty/escape/walker/scan to the same object (compiler-level
   retention, `:649-650` at `3b109793c`); fences remain alongside.
3. **Untimed validation chain, before any reuse**: sentinel freshness scan
   (a surviving sentinel = the invocation did not write), per-sample byte
   comparison arm-vs-arm per row, and comparison against a byte-exact
   **sample-0 reference**. No digest as acceptance authority (FNV
   cancellation attack closed by the reviewer); memcmp only as diagnostics.
   All scans/checks/comparisons stay outside the timed interval.
4. **Coverage**: census carrier families first — IQ2_S (gate/up shape),
   IQ4_XS (down shape), plus Q2_K/Q3_K/IQ2_XS/IQ1_M and Q6_K (control,
   expected stride-identical and ratio exactly 1.0); ISA cells NEON /
   NEON_DOTPROD / NEON_I8MM; row shapes M=1, M=4, M=15 (grouped tails,
   DotProd row / SMMLA pairs mirroring the production tile split). Timing
   reports both arms' per-sample time and the ratio, with per-sample byte
   comparisons gating every timed sample; the acceptance reading is
   ratio ≈ 1 within measured noise (canonical-vs-canonical self-run sets
   the noise band), not ratio < 1.
5. **Host runs** (`.79/.80 sparks) are requested through the arm-perf
   coordinator — never direct. Local Mac runs the same bench for correctness
   (never-green-first-run RED of this bench = this lane's defect, fixed
   before any host request).
6. **E2E/in-model validation** (post-merge, coordinator-scheduled): census
   model decode A/B at fixed placement/policy; PerfStats counters prove the
   compact path ran. Acceptance = same-or-better decode throughput and
   latency (operator bar); the −6.0 GB resident footprint is reported
   alongside as the capacity deliverable. A measured E2E regression reverts
   the layout change.

## 8. Phase plan (this branch; Unit green each phase; push per checkpoint)

- **Phase 0 — typed layout spine, zero byte change.** Layout enum, host
  constant, layout-aware stride functions (canonical arms delegate to the
  existing tables), `PreparedUnitSections` single-authority refactor,
  descriptor/layout threading through owner/view/seal/packer/stream with
  `InterleavedCanonical` everywhere. All existing tests stay green on both
  ISAs (x86 byte fixtures unchanged; ARM still canonical). New device-free
  Unit tests: layout/stride/section table contract per encoding+source.
  **Shipped at `d3e936fbd` (297/297 NEON portability; tile-register proof
  PASS after the metadata-delta hot-path fix). Known latent gap, reviewer-
  found post-push (Main-confirmed): view/seal admission divides by the
  derived stride before rejecting an unimplemented layout — entry item 0
  below fixes it before compact becomes real.**
- **OPERATOR GATE — capacity value vs ABI cost.** After Phase 0 the lane
  reports and waits: the tournament removed the decode-speed prize (§1.5),
  no measured topology is admission-blocked (§2.3), and the remaining
  deliverable is −6.0 GB resident prepared bytes (−22.6%; ≈ +34–35%
  resident routed experts) at zero arithmetic risk, for the price of the
  full ABI chain (§4) and its review.
  **GATE LIFTED 2026-11-13T00:36Z (seq 55)** — the consolidated plan is
  cleared by Main + coordinator; the operator flow is satisfied (details
  settled → reviewer pass, #212 folded in → plan decided → start). Single
  writer on this branch.
- **Consolidated stage plan (seq 55, reviewer-accepted structure).** The
  production selector stays **canonical until the whole
  consumer/admission chain is layout-aware** — activation is Stage B, not
  part of Stage A. **Stage A = layout-aware chain, ZERO byte change:**
  `kHostPreparedLayout` stays canonical; the compact tables become real
  pure-data entries exercisable through an explicit layout argument in
  unit tests only.
  - **A0 — S1 fail-closed + doc corrections** (one checkpoint push): entry
    item 0 below, plus the doc corrections folded into the same push
    (gate-based claims only; capacity math −33%/+50%/≈+35%; §6.3/§6.6
    typed-conversion framing; §2.2 L2 in-plan; §10.10).
  - **A1 — section-presence authority everywhere**, derived from
    `PreparedUnitSections`, never from the encoding alone (Main-verified
    citations): entry items 1 and 2 below, extended to every producer
    (direct int8, deferred, rotation), every copier (gather/scatter/direct),
    the absent-emins accessors (`WeightView.h:367-371`,
    `PackedWeights.h:446-451` — fail explicitly), and the host reference
    conversion (`ExpertTierWeightStream.cpp:1344-1356` / `:1554-1565`).
  - **A2 — endpoint-layout GPU conversion**; **A3 — identity/transport/wire
    v8**; **A4 — planning/inventory/PMA**. Same dispatch pattern, after
    A0+A1; A3's detail is the former Phase-2 bullet below.
  - **A2 fixup (reviewer's consolidated set, seq 85; landed as one push on
    top of `85ddd8572`).** Five items beyond the checkpoint body:
    (1) **Q6 high-plane origin** is a NAMED subplane quotient
    (`kQ6LowPlaneBytesPerColumn = 32*4/8 = 16`), static_assert-tied to the
    packer geometry in both kernel files — the high plane begins at the
    LOW-NIBBLE payload end inside the 24-column-byte payload, never at the
    payload-end section boundary (Stream.cpp `:779/:794` + the CPU oracle
    remain the reference); a device-free partial-unit regression
    (`PartialUnitQ6TransfersCarryTheHighPlaneInsideThePayload`) pins the
    section math (payload 24 / scales 24 / mins 26 column-bytes, 16+8=24)
    and both-direction host-reference round trips for N∈{12,37}.
    (2) **API migration is total**: `makeGpuToCpuExpertTierWeightStreamManifest`
    carries a REQUIRED `destination_layout` at every call site (37 sites,
    repo-wide grep, no default argument, no compatibility overload), and
    the host reference `gpuToCpuExpertPackedReference` takes the
    destination layout explicitly (20 sites) so the reference never
    silently canonicalizes a non-canonical destination.
    (3) **Presence honesty**: the non-CMS secondary metadata accesses in
    the CUDA/ROCm kernels remain conditional on the source format's
    compile-time encoding facts (asymmetry/secondary presence/codebook),
    with plane-derived OFFSETS; the earlier wording "every site
    plane-driven" overstated this and is corrected here. What is now
    enforced — the fail-closed option — is admission: both layout
    validators (`ExpertTierWeightDeviceLayout::valid()` and the CPU
    storage layout's) require the section tables to satisfy the encoding's
    metadata presence contract EXACTLY (compensation iff not
    Q6-dual-scale/CMS; secondary iff asymmetric or
    NLSS/IQ3S/Q6/CMS; emins iff CMS), so a layout that drops a plane a
    kernel requires, or invents one it forbids, is rejected before launch
    — the kernels never skip silently. Canonical satisfies this
    tautologically; compact tables are bound to the same section
    semantics.
    (4) **Field-tamper controls**: each of the five derived wire-plane
    fields, mutated independently on an otherwise-valid device layout,
    fails validity (re-derivation mismatch); the carried-layout rejection
    control is labelled known-but-unimplemented (compact tables are not
    admitted yet), not a two-valid-layouts disagreement.
    (5) **Q6 native registration**:
    `PartialQ6UnitsCarryTheHighPlaneInsideThePayloadBothDirections`
    (CUDA+ROCm rectangle contracts; live widths 12/37 + 64 full-unit
    control, both directions, three dirty replays with alternating chunk
    widths, guarded live extents against 0xa5, live-row-exact moved-byte
    counts) registered per backend as
    `V2_Integration_${backend}_ExpertTierPartialQ6PayloadPlane` in
    `V2ExpertTierRectangularWeightTests.cmake`. CUDA runs natively in the
    GB10 boundary build; the ROCm registration compiles in PR CI.
  - **A3 landed (seq 94 dispatch, one push on top of `433e84632`): the
    layout enters every versioned identity.** (1) **Tier stream v4**:
    `cpu_endpoint_layout` is a named, hashed manifest field (canonical is
    byte zero; the seal covers the byte and pins the reserved tail), and
    receivers keep re-deriving both strides from the carried layout's
    section tables; a re-sealed compact stream authenticates its hash and
    still fails on the unimplemented tables — no hash and no stride
    equality substitutes for layout identity. (2) **Remote projection
    ABI 5**: the transaction identity carries the layout in the hash
    (`hashIdentity`), the manifest inherits the tier stream's sealed byte
    instead of dropping it, `remoteCpuProjectionDeviceLayout` derives the
    CPU layout from the carried byte instead of forcing canonical,
    `baseManifest` derives wire strides from the sealed identity layout
    (unimplemented layouts yield zero strides that validity rejects), and
    `requireCanonicalCpuRoot` requires the CPU root to BE the sealed
    layout — no CPU↔CPU repacker, mismatch is a typed rejection. (3)
    **PackedWeights archive v8**: the header's `prepared_layout`
    (promoted from reserved) is sealed into the versioned identity;
    receivers decode it and re-derive both strides from its section
    tables before adopting any payload; exactly v7 remains admissible as
    an explicitly canonical archive; v6 and earlier reject globally, not
    per-format. (4) **Carried identity offsets landed**: the copier's Q6
    payload subplane quotient is named and static_assert-tied
    (`kQ6LowPlaneBytesPerColumn`, shared semantics with the CUDA/ROCm
    kernels — 16 and the 1024 bitplane window derive from it); the CMS
    host demotion addresses scales/secondary/emins through the receiving
    layout's absolute section offsets (no `+0/+128/+256` family
    arithmetic); the Q6 reference moves both dual-scale words and the
    bitplane window through `cpu_sections`/`output_sections` (no
    1536/1664/1024 literals); `chunkScales`/`chunkMins`/`chunkComp`
    derive from `preparedSections()` with an explicit absence guard on
    `chunkMins` (an absent secondary section is a typed rejection, never
    a family-guessed offset); `requireCompatibleCpuWeights` includes the
    prepared layout in the compatibility set. Focused regressions:
    `V2_Unit_PackedWeightsArchiveLayoutIdentity` (v8 seal, exactly-v7
    explicit, global pre-v7 reject, unknown byte, stride tamper, compact-
    byte unadmitted, direct-descriptor path) registered Unit +
    ProductionTestPreflight; `RemoteProjectionIdentitySealsEndpointLayout
    AndRejectsSubstitution` in the distributed-residency protocol suite
    (carry, derive, unknown-byte gate, re-sealed compact rejection);
    `OwnerMetadataAccessorsDeriveFromSectionsAndGuardAbsence` in the
    layout suite. Compact section tables remain INVALID and unadmitted:
    their stride tables are the next step (pure-data entries + explicit
    layout argument in tests), unlocked by this identity binding but
    held for the A2 pin ACK per Main's constraint.
  - **BINDING FOR A3 (reviewer, seq 85 — precondition for any compact
    table to become valid).** The unhashed `reserved_identity[0]` carrier
    is safe today ONLY because no second valid layout exists; once compact
    tables are admitted, an equal-stride Q6 (or any layout sharing
    canonical strides) passes validity WITHOUT a hash change even with the
    canonical selector — so the layout must enter EVERY versioned identity
    before compact tables become valid: (a) tier-stream v3 → version bump
    carrying the layout in the sealed identity; (b) the remote-projection
    manifest (`MoEOverlayRemoteProjectionProtocol.cpp:709-727`) currently
    DROPS the layout, and `remoteCpuProjectionDeviceLayout`
    (`:890-894`) forces canonical — both become layout-carrying; (c) the
    `PackedWeights` archive promotes to v8 with the layout in the hashed
    identity. **Sequencing rule: compact section tables stay INVALID and
    unadmitted until A3's identity binding lands; only after that may
    tests introduce compact tables via the explicit layout argument.**
  - **Stage B — activation** (the production selector moves to compact on
    AArch64; packer compact placement `kHostPreparedLayout`-driven;
  view/seal/NEON admission require it): held behind the reviewer's
  **Gate A**, fed by Stage-A per-checkpoint evidence.
- **Phase 1 — entry item 0 (reviewer spine finding, Main-confirmed
  2026-11-13, seq 45): fail-closed admission before any division.** At
  `d3e936fbd`, `admitStructure` (`CPUNativeQuantisedWeightView.h:456-459`)
  takes the stride from the layout-aware `preparedInterleavedBlockStride`
  and immediately divides by it in the extent overflow guard; an
  unimplemented layout returns the fail-closed ZERO stride
  (`preparedUnitSections` compact arm, `CPUNativeQuantisedPreparedFootprint.h`),
  so a compact descriptor reaches **division by zero** instead of a precise
  rejection, and `MetadataSeal::validate` (`:539-542`) reaches the same
  path. Latent in Phase 0 (no production caller selects compact; the packer
  fixes `kHostPreparedLayout`), and the Phase-0
  `UnimplementedLayoutFailsClosed` test covers tables/footprints only, not
  view/seal admission. Fix BEFORE compact is real: explicit
  unimplemented-layout/zero-stride rejection in view admission and the
  seal, placed **before the overflow guard**, plus a view/seal-level
  regression. **Generalized invariant: every future layout must fail closed
  the same way — a layout without implemented section tables is rejected at
  admission with a precise diagnostic; the zero stride is a sentinel for
  pricing/admission tables, never a value a division may consume.**
- **Phase 1 — entry item 1 (reviewer #212, Main-confirmed in source, seq
  53): tier-stream section copy is encoding-only.**
  `ExpertTierWeightStream.cpp:755-762` (`copyCPUEncodedColumn`): for every
  CMS encoding it copies two u16 planes plus one u32 emins plane at
  `data_stride + 4×columns` on **encoding-only** logic. With IQ2_S compact
  (payload 768, stride 1024) the emins access begins outside the trimmed
  unit; full-unit memcpy paths can hide it, but the partial/rectangular
  paths use this helper. Fix: derive plane presence/offsets from the
  layout's `PreparedUnitSections` — emins is present only when
  `has_emins` (Q2_K sources).
- **Phase 1 — entry item 2 (reviewer #212, Main-confirmed in source, seq
  53): packer writes are section-derived but not section-guarded.**
  `CPUNativeQuantisedWindowPacking.inc` — the CMS emins write (at
  `sections.emins_offset`, currently unconditional for CMS) and the non-CMS
  compensation write (at `sections.compensation_offset`, unconditional for
  nibble/expanded) are safe under canonical Phase 0, but must be
  **presence-guarded** from `PreparedUnitSections` before compact
  activation: compact IQ2_S has no emins section and compact
  nibble/expanded have no compensation section — an unguarded write lands
  at offset 0 inside the payload.
- **Focused regressions (entry items 1+2):** (a) a mixed-stride
  partial/rectangular tier copy with IQ2_S and Q2_K units side by side;
  (b) packer section-presence checks.
- **Design invariant (reviewer's, adopted): every writer and copier derives
  section presence and offsets from `PreparedUnitSections` — no
  encoding-only assumption survives.**
- Reviewer status note (seq 53): the NEON compute read-set argument for the
  Phase-1 compact layout is structurally sound (static pass; final report
  pending, separating these activation gaps from the Phase-0 division
  defect).
  - **Stage C — L2 access-pattern items (§2.2)**: each L2 item is a **new
    physical identity** with the superblock-shared-base,
    K-origin/partial-superblock/tail and wire-geometry obligations spelled
    out in §2.2 — never a flat stride change — each behind its own
    measurement gate and identity/version work. All capacity percentages
    quoted for L2 (§2.3, §10.10) are idealized predictions, not admitted
    counts.
- **Stage B activation detail (former Phase 1 body).** Compact stride
  tables become production-selected; packer compact placement
  (`kHostPreparedLayout`-driven); view/seal/NEON admission require the
  host layout; estimator overloads with layout (host default + explicit
  target). ARM-local Unit: accessor bit-identity vs canonical for every
  format; execution vs scalar reference byte-exact for every
  encoding/family/tile. x86: untouched bytes, all green. Sequenced behind
  Gate A.
- **A3 detail (former Phase 2) — archive/wire/movement versioning.** Archive v8 + layout field
  + rejection tests; tier-stream/remote-projection/slot-pool/residency
  fabric layout identity; CUDA+ROCm conversion kernels layout-conditional
  comp write (CMS kernels already parameterized); cross-layout rejection
  regressions; estimator/BOM byte-equality per-ISA in preflight.
- **Carried offsets — LANDED in A3 (seq 94):** the copier payload group
  ordinal offsets, the CMS host demotion relative metadata arithmetic, the
  Q6 reference 1536/1664 constants, and the `PackedWeights`
  `chunkScales/chunkMins/chunkComp` arithmetic plus the owner `chunkMins`
  absence guard are section-offset-driven (or guarded) now — see the A3
  record above. `ResidencyFabric::requireCompatibleCpuWeights` includes
  the prepared layout in its compatibility set. Still carried: the
  absent-emins trimmed-unit control for the tier-copy regression, which
  becomes representable when the compact stride tables exist as pure-data
  entries (test-only via the explicit layout argument, after the A2 pin
  ACK per Main's constraint).
- **Gate-A evidence (former Phase 3) — bench + focused preflight
  regressions.** The §7 bench in the
  performance namespace (non-regression reading); focused regressions
  registered in `ProductionTestPreflight` for the ABI invariants (layout
  identity mismatch, archive version rejection, estimator-byte-equality
  under compact); PerfStats counters for the compact path (which layout a
  projection executed).
- **Post-merge (Main integration)**: rebase onto the merged develop tip
  (reviewer ACKs do not transfer across rebase), E2E decode A/B on the
  census model via sparks (same-or-better gate; regression ⇒ revert),
  observed per-tensor enumeration seam if commissioned.

## 9. Risks / open questions for the coordinator

1. **Lane-value assessment (the coordinator's close flag).** My reading:
  **do not close, but re-baseline expectations.** The design is sound and
  the tournament negative does not attach to it (§1.5: zero decode-program
  change; the read-set argument passed the reviewer's static pass, seq 53),
  so the static case for neutrality is strong — but the performance bar is
  enforced by the §7/§8 measured gate, not by construction, and its
  deliverable is now purely
  capacity/residency (−6.0 GB, no blocked topology today), against a real
  ABI-chain cost. That trade is an operator call, hence the §8 gate. If the
  operator weighs only decode performance, the lane should close at the
  gate; if expert-tier residency/KV headroom on UMA hosts matters for the
  post-merge roadmap, it proceeds.
2. **Cross-ISA CPU expert tiers** are typed-rejected (§6.3). If a serving
  topology needs x86↔ARM prepared-CPU movement, an explicit conversion lane
  must be commissioned — this lane does not build it.
3. **L2 refinements (§2.2) are in-plan behind their own gates, not part of
  Stage A.** Superblock scale dedup, secondary byte-packing and the 9th
  byte-plane change the access pattern (cross-unit scale addressing /
  byte-widening loads); the seq-51 correction (§10.10) moved them from
  out-of-scope to in-plan L2 because the CMS/NLSS decode programs are
  unchanged — but each enters only with its own measured non-regression
  case, sequenced after L1's gate.
4. **Microarchitectural stride effects.** Changing the unit stride (e.g.
  1536→1024) perturbs DRAM row/bank and prefetcher behavior in principle;
  the program-level read-set identity is the static argument, not a
  performance guarantee. The §7 noise band (canonical self-run) and
  the E2E same-or-better gate are the binding protections; a measurable
  regression reverts the change.
5. **ARM CI surface**: compact-layout tests are ARM-gated like the existing
  NEON tests; the x86 develop gate exercises the canonical tables and the
  shared ABI (versioning, rejection paths, estimators) device-free.
6. **The tournament's own scope note**: its compact arm was a bench-local
  IQ3_S inline-decode candidate, not this design; the seal bounds
  decode-program trades for the IQ3_S family and the bandwidth-bound
  assumption at those shapes — nothing more. This design cites it as the
  reason its claims are capacity-only.
7. **Tiered residency (§10)**: the T2-for-CMS branch was **invalid and is
  corrected (§10.10, Main-found premise error, seq 51)** — the CMS/NLSS
  hot paths already decode their native payload inline, so a source-size
  form for those families is the same program with load-shape changes
  only. Tiering survives solely for expanded-baseline families (≈0.7 GB
  on the 35B — negligible), and the hot-set cardinality measurement is
  **off the critical path**. The remaining §10 content (promotion cost,
  cold-miss options, prefill interaction) stands for the
  expanded-baseline families where applicable.

---

## 10. Addendum (gate context, design-only): tiered residency — compact at
rest, expanded on promotion

Operator question (2026-11-12T23:37Z): **are the byte savings achievable
without ever putting compact bytes in the hot decode path** — i.e. without
the tournament's 7.0–8.6× decode program? The option to size: all experts
stored compact in the tier pool (~source bytes); an expert is expanded into
the existing hot-expert-cache/residency machinery only on promotion (one
expansion pass per expert-layer, off the decode critical path in maintenance
streams); steady-state decode reads only expanded hot bytes; the ABI chain
carries both representations with `PhysicalMemoryAuthority` accounting for
each. This section sizes that option from sealed inputs only (census
K14-I08A-CENSUS-5C2CEF47, tournament K14-I08A-TOURNEY-3B109793C, idea-16
per-step attribution, idea-25 prefill split). No new runs; no implementation.

### 10.1 Three representations, two tier variants (corrected — §10.10)

**Premise correction (Main-found, coordinator source-verified, seq
51, 2026-11-13):** the original table attributed the tournament's program
B (inline grid-decode, 7.0–8.6×) to every source-scale form. Verified in
source at `d5e04fdce`: the CMS hot path (`dotHalfRowPairs` Codebook
13/14/17, `CPUNativeQuantisedMultiScaleKernels.h:399-416` and `:470`)
loads the unit's native grid words and decodes them **in-kernel** via
`Ops::gridPair` — program A already includes the grid decode; the 1536 B
unit is native payload + layout overhead, not decoded weights (§1.5
documented this; the original §10.1 contradicted it). Corrected model —
the source-scale form's program is family-dependent:

| Representation | Bytes (35B) | Decode program | Speed |
|---|---|---|---|
| (a) canonical (today) | 26.62 GB | A per family: CMS/NLSS decode the native payload inline (grid/LUT); IQ3S/ExpandedInt8 families serve pre-decoded planes | X (measured baseline) |
| (b) Phase-1 compact (§2) | 20.6 GB | A unchanged — dead bytes never loaded (§2.1 read-set proof) | ≈X claimed, Phase-3 bench-gated |
| (c) source-scale (~source bytes) | ≈15.3 GB | **inline-decode families (CMS, NLSS)**: same program A, load-shape changes only. **expanded-baseline families are not one class**: IQ3S would need inline grid-decode (program B); Q5_K-class ExpandedInt8 sources would need planar/nibble decode, and Q4_1 is NibbleLUT, not ExpandedInt8; **Q8_0 is the native-int8 source whose compact unit (2176 B) already equals its source bytes — no decode program exists to change. Q8_1 is not in that class: its 36-byte source blocks (`BlockStructures.h:69-76`) do not equal a 2176 B compact unit** | CMS/NLSS: no sealed penalty (the tournament did not price this class). IQ3S: X/7.0–8.6 sealed for that family. Other families: **unmeasured hypotheses; the IQ3_S timing does not transfer across families** |

The tiered option holds (b) or (c) at rest and expands to (a) on promotion:

- **T1 — compact-tier insurance**: tiers hold (b). Cold experts serve at
  ≈X (program A on compact bytes — no miss penalty *if* §2.1 holds).
  Promotion buys canonical hot bytes regardless of the Phase-3 bench
  outcome. If §2.1 holds, promotion buys nothing measurable; if the bench
  refutes it, T1 is the fallback that keeps the −4.5 GB tier trim while
  restoring canonical hot-path speed.
- **T2 — source-scale tiers**: tiers hold (c)-grade bytes (~source).
  **Valid only for expanded-baseline families** — where (c) genuinely
  needs program B or expansion before it serves at speed. For
  inline-decode families (the 35B's CMS gate/up + NLSS down carrier), (c)
  is program A with load-shape changes: no tiering is needed to reach
  ~source bytes on the hot path (§10.10). Cold experts in valid T2 either
  serve through program B or block until expanded; promotion is
  grid-decode + repack. The 7.0–8.6× figure is the **sealed IQ3_S result
  for that family only** — it does not transfer to other expanded families,
  whose source-scale decode programs are unmeasured (and Q8_0/Q8_1 have no
  source-scale decode program at all).

The operator's framing ("~source bytes in the tier pool") is T2. Both
variants reuse the same promotion fabric; they differ in tier format,
cold-serve path, and kernel surface (§10.5, §10.8). **The 35B T2 rows in
§10.3/§10.6/§10.8 are superseded for the CMS/NLSS bulk by §10.10 — they
survive only for the expanded-baseline families (≈0.7 GB on the 35B).**

### 10.2 Per-instance arithmetic (census-anchored)

The 35B (Qwen3.6-35B-A3B) has 41 routed layers × 256 experts = 10 496
expert-layer instances; each instance is gate+up (2 × 2048×512) + down
(2048×512) = 3.15 M weights (idea-16 shapes; the tournament tile is exactly
this shape — `kRows=512`, `kK=2048`, `Perf__CPUCompactIQ3SDecode.cpp:105`).

Per typical instance (CMS gate/up + NLSS down; blk.39's IQ3S gate/up is a
2.5 MB canonical outlier over 256/10 496 instances — negligible):

| | source | Phase-1 compact | canonical expanded |
|---|---|---|---|
| gate/up | 0.67 MB | 1.05 MB | 1.57 MB |
| down | 0.56 MB | 0.66 MB | 0.72 MB |
| **total** | **1.23 MB** | **1.71 MB** | **2.29 MB** |

Model-level anchors (census + §2.3): expert-routed 13.30 / 17.94 / 23.96 GB;
non-expert rest (dense, lm/embd, shared, other — Q6K-native, untouched by
Phase 1) ≈ 2.66 GB; totals 15.34 / 20.60 / 26.62 GB (±0.1 GB family
rounding).

### 10.3 (1) Hot-set sizing — resident bytes as a function of the hot set

The residency machinery has **no fixed hot-set count**: promotion is
demand-economics over histogram windows
(`src/v2/execution/moe/DecodeExpertHistogram.h`; defaults in
`src/v2/execution/moe/DeviceMoERebalancePolicyShared.h:30-84`: ≥64 window
activations to be a candidate, 1300‰ imbalance threshold, ≥50‰ improvement,
≤4 swaps/layer/wave, ≤16 plan entries/wave, 65 536-token payoff horizon;
window 256 tokens growing 1.5× to 4096 — `RuntimeConfig.h:2000+`). The hot
set is whatever persistently clears those bars; `--moe-hot-expert-cache`
adds bounded *replicas* on top (`MoEHotExpertCacheConfig`,
`RuntimeConfig.h:1814`, default `Off`). **The measured hot-set cardinality on
real routing traffic does not exist in any seal — the census carries no
routing histogram. It is the decisive unknown for this option and should be
read as the gate input it is.** Until measured, the honest sizing is the
resident-bytes function itself:

Let h = promoted (expanded) instances per layer, 41 layers, keep-both
accounting (the promoted expert's tier copy stays until demotion):

| h/layer | instances | T2 rest+tiers+hot | T1 (compact tiers) | Phase-1 only | canonical only |
|---|---|---|---|---|---|
| 0 | 0 | 15.96 GB | 20.60 GB | 20.60 GB | 26.62 GB |
| 8 (top-8 working set) | 328 | 16.71 GB | 21.35 GB | 20.60 GB | 26.62 GB |
| 16 | 656 | 17.46 GB | 22.10 GB | 20.60 GB | 26.62 GB |
| 32 | 1 312 | 18.97 GB | 23.61 GB | 20.60 GB | 26.62 GB |
| 49 | 2 009 | 20.60 GB | 24.61 GB | 20.60 GB | 26.62 GB |
| 64 | 2 624 | 21.97 GB | 26.62 GB | 20.60 GB | 26.62 GB |
| 113 | 4 633 | 26.62 GB | — | — | 26.62 GB |

Formulas: T2 keep-both = 15.96 + 0.0939·h GB; T1 keep-both = 20.60 + 0.0939·h;
net-replace (tier slot released on promotion) = 15.96 + 0.0435·h (T2) and
20.60 + 0.0238·h (T1). Self-checks: T2 net-replace at h=256 → 26.7 ≈ canonical;
T1 keep-both at h=64 → 26.62 = canonical.

**Reading:** T2 beats Phase-1-only iff h < ≈49/layer (net-replace: h < ≈107);
T2 beats canonical-only iff h < ≈113 (net-replace: always, down to the
15.96 GB floor). T1 never beats Phase-1-only on bytes — its value is
hot-path insurance, not capacity.

### 10.4 (2) Promotion cost — one expansion pass per expert-layer instance

Grounded in the tournament's sealed timing (its tile **is** the expert
tensor shape, §10.2): compact program-B decode runs at 30.1 Gw/s (M=1 I8MM,
34.8 µs per 1.048 M weights) to 58.0 Gw/s (M=15, 271 µs per 15.7 M) at 20
threads on GB10 (.79, GCC-14, Release). One instance = 3.15 M weights =
three tournament matrices:

- **54–105 µs of 20-thread work** per instance; ≈ **0.27–0.53 ms at the 4
  default maintenance streams** (`kDefaultMigrationExecutionStreams = 4`);
  ≈ 1–2 ms worst-case single-thread. Plus ≈3.6 MB UMA read+write — trivial.
- **Maintenance-token bound:** the cold-serve penalty is ≈ +206–261 µs per
  cold instance read (§10.5), so a 0.3–0.5 ms expansion **repays after ~2–3
  activations** of that expert. The 65 536-token payoff horizon
  (`kDefaultMigrationPayoffHorizonTokens`) is never the binding constraint;
  the binding constraint is the **fabric throttle**: at ≤16 plan
  entries/wave, filling h=32/layer (1 312 instances) takes ≈82 waves ≈
  **25–30 minutes** at the 256-token window (~19 s at 75 ms/step; windows
  grow toward 4096, stretching this). Slot/stream capacity is not the
  throttle (16 entries × 0.4 ms through 5 slots ≈ 1.3 ms of stream work per
  wave); the plan-entry cap is, and it is config, not code.

A promotion is therefore cheap per instance but the hot set fills at
minutes-scale at default fabric — relevant if the operator pictures
burst-fill after a traffic shift.

### 10.5 (3) Cold-miss behavior — honest options

Per-step anchors (idea-16, post-bundle-fix, 35B MTP off, ~75 ms/step):
routed experts ≈ 274 µs/layer (pooled 8-expert workshare) ≈ 34 µs per expert
instance read expanded; GDN 440 µs; LM head 5.7 ms. MTP dynamic ≈ 27 tok/s.

- **Serve-compact-slow (T2 only):** a cold instance read runs program B at
  the sealed ratio: 34 µs → ≈240–295 µs, i.e. **+206–261 µs ≈ +0.3% of the
  75 ms step per cold instance read** (additive in the pooled workshare). A
  step with k cold reads pays ≈ +0.3·k%; the all-cold worst case (328
  reads) is ≈ 2× the step — degraded but serving, no deadlock. **This option
  requires production program-B kernels for every tier encoding family** —
  the tournament built one bench-local family (IQ3_S); the repo's coverage
  totality bar (every codebook, M sweep, backends) makes this a large
  kernel surface, and it is the very program the tournament priced.
- **Serve-compact-slow (T1):** no miss penalty *expected* (compact
  bytes are program-A bytes minus dead bytes; §2.1 is the static argument —
  the Phase-3 bench and E2E gates measure what the argument cannot,
  including stride/cache perturbation). This is the insurance case — the
  penalty exists only if the bench refutes §2.1, in which case it is the
  sealed ratio on the compact-read share.
- **Block-until-expanded:** 0.3–0.5 ms + queueing per instance against 5
  preallocated slots; a burst of newly-hot experts serialises decode behind
  maintenance — a bounded stall risk that also cuts against the repo's
  no-blocking-sync hot-path discipline. Rejected as a default; only
  defensible as an explicit admission policy for a latency-insensitive
  serving class.
- **Verifier rows:** grouped MTP verifier rows are decode work routed to the
  same experts as the accepting step, so steady-state verifier reads hit the
  hot set; only newly-hot experts serve cold (experts ≈ 3.8 ms per grouped
  call would become ≈ 30 ms only if *all* its members were cold — a
  transient, not a steady state).

### 10.6 (4) Combined-with-Phase-1 floor — the 35B theoretical resident minimum

The combined option (dead-trimmed non-hot bulk + compact tiers + expanded
hot set) minimises at the smallest viable hot set. With the top-8 working
set (h=8):

- **T2 net-replace: ≈ 16.3 GB** (15.96 floor + 8×41×(2.29−1.23) MB) — the
  theoretical resident minimum for this option family on the 35B.
- T2 keep-both: ≈ 16.7 GB.
- For comparison: Phase-1-only 20.60 GB; canonical-only 26.62 GB; absolute
  all-source floor (h=0, every read program-B — not a serving config)
  15.96 GB; census source total 15.34 GB (the 0.6 GB gap is the Q6K-native
  non-expert rest, which no variant trims).

So the tiered family's realistic floor is **≈ 16.3–16.7 GB**, i.e. −3.9 to
−4.3 GB beyond Phase-1-only, at h=8. The floor degrades toward Phase-1-only
as h grows (break-even h≈49 keep-both / h≈107 net-replace).

### 10.7 (5) Prefill and grouped-row interaction

Prefill on this model is overhead-dominated, not expert-kernel-bound:
measured marginal 3.01–3.22 ms/token vs captured stage kernels 0.60
ms/token ⇒ ≈80% of prefill wall is bucket-pipeline replay/launch/sync +
fixed intercept (idea-25, head numbers 2026-11-08; the operator's ~91%
host-bound reference is this measurement's direction). Expert streaming is
a slice of the 20% kernel share:

- A 512-token batch touches up to min(256, 8·512) = 256 experts/layer;
  grouped same-expert row sharing (pooled expert FFN, `408ff90e5`) streams
  each touched instance once. Worst-case all-touched: expanded streaming
  ≈ 73 ms of a ≈1.54 s prefill; compact-slow (58 Gw/s) ≈ 567 ms ⇒ **≈ +32%
prefill marginal**. At 64/layer touched (2 624 instance-streams): ≈ **+8%**. So a cold-heavy prefill
  costs single-digit to low-tens percent — prefill's overhead dominance
  absorbs most of the 7.9×.
- **Expand-on-prefill is not viable**: a 512-token batch can touch all
  10 496 instances ≈ 4+ s of expansion work for transient demand. The
  correct interaction is the existing one: prefill activations feed the
  demand histograms (`DecodeExpertHistogram.h`), promotion follows
  *persistent* demand, and transient prefill reads serve compact-slow at
  the bounded cost above (T2) or ≈speed-neutral (T1).

### 10.8 (6) Delta vs Phase-1-only

| | Phase-1 only | T2 tiered (h=8…32) | T1 tiered |
|---|---|---|---|
| Resident bytes (35B) | 20.60 GB | 16.7–19.0 GB | ≥ 20.6 GB (21.3+ at h=8) |
| Beyond Phase 1 | — | **−1.6…−3.9 GB** | none (costs extra) |
| Cold experts per tier byte | 1/1.71 MB | 1/1.23 MB (**+36%**) | 1/1.71 MB |
| Hot-path representation | compact (bench-gated ≈X) | canonical (X by construction) | canonical (X by construction) |
| New kernels | none | program-B production sweep + expansion pass | expansion pass only |
| ABI/PMA | one layout dim | **two representations**, dual accounting, representation-aware migration cost model | two representations, dual accounting |
| Hot-set fill | — | minutes at default fabric | minutes |
| Unknown dependency | §2.1 bench confirmation | **measured hot-set cardinality** + Phase-1 machinery anyway (tiers hold compact bytes) | §2.1 bench outcome |

The honest summary for the gate:

1. **Phase 1 alone** buys −6.0 GB with one representation, zero new kernels,
   and a bench-gated speed-neutrality claim.
2. **Tiering (T2)** buys −1.6 to −3.9 GB *more* (to a ≈16.3–16.7 GB floor)
   and keeps the hot path canonical by construction — but requires Phase 1's
   machinery anyway if tiers hold compact bytes, adds a second
   representation end-to-end (tier store, expansion pass, dual PMA
   accounting, representation-aware migration economics in
   `DeviceMoERebalancePolicyShared.h`, representation identity in
   `ExpertTierWeightStream.h`'s exact-migration-state validation), a
   production program-B kernel sweep for cold serve, and its value pivots on
   an unmeasured hot-set cardinality (break-even h≈49/layer keep-both).
3. **T1** is not a capacity option; it is the insurance variant if the
   Phase-3 bench refutes compact-read neutrality.

### 10.9 What this addendum does not claim

No topology unlock (same as §2.3 — no measured topology is admission-blocked
at 26.62 GB). No decode-speed claim for any variant. All numbers are
sealed-input arithmetic (census, tournament, idea-16/25 head numbers) —
projections, not new measurements; per-instance values carry ±0.1 GB family
rounding. The hot-set cardinality, the Phase-3 bench outcome, and the
representation-aware migration cost model are the three inputs that would
change this table, and none of them exists today. Implementation of anything
in this section remains gated on the operator's CONTINUE/CLOSE decision.

### 10.10 Premise correction (Main-found, coordinator source-verified, seq 51) and its consequences

**The error (in the original §10.1):** the canonical CMS form was treated
as "expanded" and the tournament's program B (inline grid-decode,
7.0–8.6×) was attributed to the source-scale tier form for **all**
families. Source-verified at `d5e04fdce` and re-verified on this branch:
the CMS hot path (`dotHalfRowPairs` Codebook 13/14/17,
`CPUNativeQuantisedMultiScaleKernels.h:399-416` and `:470`) loads the
unit's native grid words and decodes them in-kernel via `Ops::gridPair`;
the packer writes the source's own packed payload bytes verbatim
(`CPUNativeQuantisedWindowPacking.inc`, CMS payload branch). The canonical
1536 B unit is native payload + layout overhead — not decoded weights.
This was already §1.5's argument; the original §10.1 contradicted it. Only
the expanded-baseline families (IQ3SExpandedSourceScale, ExpandedInt8) have
pre-decoded canonical bytes, and only for them does source-scale storage
require program B or expansion.

**Consequences (accepted into the plan):**

1. **T2-for-CMS is invalid.** The 35B's expert carrier (CMS gate/up + NLSS
   down) is entirely inline-decode families: a source-size form is the
   same program A with load-shape changes only. Phase 1 (L1) plus the §2.2
   L2 access-pattern items reaches ~source bytes (≈ 15.8–16 GB potential:
   gate/up 10.74 → ~6.9 GB, down 6.21 → ~5.3 GB) **on the hot path without
   any tiering**, each L2 item behind its own measurement gate. These
   percentages are **idealized byte-arithmetic predictions, not admitted
   counts** — admission derives live quotas through
   `PhysicalMemoryAuthority` per tier at run time (§2.3).
2. **T2 tiering survives only for expanded-baseline families** — on the
   35B: blk.39 IQ3S (0.64 GB) + dense ExpandedInt8 (0.04 GB) ≈ 0.7 GB,
   negligible. The §10.4 promotion-cost and §10.3/§10.6 floor arithmetic
   stands where applicable (those families only); the 35B T2 rows are
   superseded for the CMS/NLSS bulk.
3. **The hot-set cardinality measurement (§10.3's decisive unknown) drops
   off the critical path** — it was T2's dependency.
4. **Mirror option (design record — unmeasured hypothesis):** today's CMS
   hot path runs an inline grid-decode program, the same *class* as the
   tournament's compact arm. That class observation is **not a speed
   claim**: the tournament priced IQ3_S shapes at M=1 only, and the timing
   does not transfer across families — no IQ2_S or Q2_K measurement exists.
   The mirror trade — pre-expand IQ2_S for speed at +bytes (≈ the census's
   +9.2 GB carrier, 6.0 → 9.5 bpw) — is recorded as a hypothesis with a
   **mandatory measurement gate before any claim or plan change**; the
   sealed IQ3_S result motivates the measurement, it does not substitute
   for it. It is a separate lane with its own economics; this lane's bench
   gate stays non-regression for compact-for-capacity.
