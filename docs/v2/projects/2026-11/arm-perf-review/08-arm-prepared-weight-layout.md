# Idea 8 — ARM prepared-weight layout (dead bytes + expanded formats)

Status: **Open**. ABI-bearing change; last in dependency order, and only with
measurements behind it. Highest memory value on GB10 because CPU and GPU share
one UMA pool (see idea 2).

## Context

The NEON path reuses the x86 VNNI 64×32 block layout unchanged
(`CPUNativeVNNIPreparedFootprint.h`, `CPUNativeVNNIEncoding.h`,
`CPUNativeVNNIWeightView.h`):

1. **Dead compensation bytes.** The 128-byte `comp_int16` section exists for
   x86 `u8xs8` compensation (`sum((qa+128)*qw) - 128*sum(qw)`). ARM signed×signed
   SDOT computes `sum(qa*qw)` directly and the review found no NEON reader of
   the compensation section (Q6 dual-scale and CompactMultiScale carry their own
   metadata; `WeightView` arms it only for encodings that need it).
   Stride cost at the reviewed head: NibbleLUT 1280 B (10.0% dead),
   ExpandedInt8 2304 B (5.56% dead).
2. **Expanded formats.** Canonical mapping `CPUNativeVNNIEncoding.h:36-73`:
   - nibble codebooks 0/4/5 (4-bit, LUT),
   - IQ4_XS source-aware override,
   - IQ3_S source-scale expanded (~9.5 bpw vs 3.44 bpw source),
   - Q6_K native dual-scale,
   - CompactMultiScale families,
   - remaining supported tuples → ExpandedInt8 (9 bpw).
   Explicit rotation always yields ExpandedInt8.
   ARM `vqtbl` makes in-register nibble decode cheap, so bandwidth-bound decode
   would benefit from keeping IQ3_S/IQ5/expanded formats compact.

## Proposed shape (outline, not implementation)

A typed physical-encoding/layout identity on ARM, threaded through:
packer → view → footprint/BOM → prepared owners → cache keys → wire/slot/
movement → policy/evidence. Not a new parallel ledger; not merely a new archive
hash. Either:
- drop only the compensation section for encodings where no ARM reader exists
  (validated by mutation tests per encoding), or
- keep IQ3_S and other expanded formats in a compact ARM-native block.

## Exactness / risk

- Medium–high: layout is part of the prepared identity. Every consumer of the
  stride (packer encode/decode, readers, checksums, migrations, MPI wire
  codec, BOM pricing, replay fixtures) must agree; a partial update would mix
  layouts across ranks/models.
- Arithmetic per weight is unchanged; only physical bytes and strides change.

## Validation

- Device-free BOM/pricing tests for the new footprint; prepare→load→infer
  byte-auth for every format incl. MXFP4; migration/MPI wire compatibility runs;
  GB10 UMA admission with idea 2 to show real capacity gain. Unmeasured
  prepared-vs-GGUF resident ratios today: no complete ARM engine has loaded a
  real model yet.

## PLAN APPROVED (2026-11-10, old addon lane; coordinator pickup in plans/idea-08-prepared-weight-layout-plan.md)

Five decisions approved: split A (dead-byte elimination — the 128 B
comp_int16 x86 section, never read by ARM signed-SDOT: 10.0% dead on
NibbleLUT, 5.56% on ExpandedInt8) / B (ARM-native compact encodings, vqtbl
in-register decode — biggest lever IQ3_S 9.5 vs 3.44 bpw, the 35B's
format); ONE ABI/migration mechanism (prepared-layout version in
cache/wire/BOM keys per CPU instruction family; mixed-version ranks fail
fast at admission; re-prepare from GGUF, no byte converters); **Phase A
GO NOW** (measurement-only, no ABI, ~1 lane-week: per-format BOM audit +
diagnostic compact-decode tournament, non-installed/rule-166-declared,
M=1 AND M=15); Phase B CONDITIONAL on engine decode ≥ +5% (A) / ≥ +15%
projected 35B (B) + the M=4 ALU check + idea-16's attribution confirming
weight-bandwidth binding; NO-GO criteria accepted. Sequenced after the
register-gate coordinator lands (rule-166 hard gate from day one).
TRACKER CORRECTION ACCEPTED: "no engine has loaded a real model" is stale
— A1/35B/27B runs load real models; the resident BOM audit is runnable.
.80 window: after the half-perf validation + idea-16 Phase A clear.

### **idea-08 Phase A ANALYSIS CORE DONE** (arm-addon2, ~14:56Z) — full markdown to follow

(1) **STATIC BOM AUDIT (exact from layout code at the current tip; tracker
numbers reproduce EXACTLY: NibbleLUT 1280 B/10.0% comp, ExpandedInt8
2304/5.56%):** full per-encoding stride table (composition, prepared bpw,
catalog-source bpw, expansion ratio) for all 21 source formats. NEW
findings: (a) CompactMultiScale carries 256-640 B per unit of STRUCTURAL
fixed-payload padding (dead on BOTH ISAs — 29% of the unit for IQ2_S, 42%
for IQ1_M) — new to the compact-layout family; (b) expansion pathology
beyond IQ3_S: IQ1_S 4.75×, IQ2_XXS 3.60×, IQ1_M 3.0×, IQ3_XXS 2.57×,
IQ2_S/IQ2_XS 2.18×, IQ3_S 2.53× (prepared 9.5 vs catalog-logical 3.75;
raw GGUF ~3.44); Q6_K is the only 1.00× encoding.
(2) **ARM ZERO-READER VERIFICATION (static, current tip):** zero
occurrences of "compensation" in any NEON translation unit; the only
data-plane accessors (chunkScales/chunkMins) skip the comp region by
construction; the ARM contribution contract is signed×signed
(vmull/SDOT/vdotq_s32/SMMLA) with no activation-bias or weight-sum term —
the x86 u8xs8 correction has no ARM consumer. ALSO: the packer WRITES comp
on every ISA (WeightPacker.h:247, :620-644) — ARM preparation WASTES
compute filling dead bytes. Phase B mutation tests remain the dynamic
proof; no static counter-indication.
(3) **35B AGGREGATE (documented pairing IQ2_S gate/up + IQ4_XS down):**
one routed expert ≈ 2.30 MB prepared vs ≈ 1.10 MB catalog-source (2.09×);
whole-model ≈ 23.6 GB vs 11.3 GB (pending exact census). Compact wins
target BOTH the IQ2_S padding and the expanded formats. Exact per-tensor
census (attention/GDN/LM-head/shared/router formats) = the runnable
resident-BOM audit on .80.
(4) **ABI MECHANISM refined to concrete identity flow:**
PreparedLayoutIdentity{cpu_family, layout_version, encoding_set_rev}
through the enumerated sites (encoding/footprint/packer/view/BOM/cache
keys/wire-slot-movement incl. ExpertTierWeightStream/MoEOverlayPrepared
WeightSource/CpuExpertSlotPool/ExpertPreparedMemoryGeometry/evidence
seals); same-version movement unchanged; mixed-version meshes fail fast at
rank admission with re-prepare-from-GGUF; x86 keeps comp and versions
independently; identity rides the EXISTING publication/admission surfaces.
(5) **TOURNAMENT DESIGN (diagnostic, non-installed, rule-166-conformant):**
compact IQ3_S primary — 1088 B/unit live floor = 4.25 bpw (2.24× below
9.5); tighter interleaving ≈3.9-4.1 bpw; vqtbl in-register decode,
byte-oracle per unit. Secondary: CompactMultiScale padding-aware variant
(IQ2_S first). Matrix M=1 + M=15 (+ M=4 ALU check), IQ3_S/IQ2_S/IQ4_XS,
all three ARM families, 2 placements. **NEEDS .80 WINDOW (build+run) —
REQUEST: one CPU window after the current queue clears (resident BOM
census + tournament, measurement-only).** macOS cannot build v2 (libnuma).
GO/NO-GO gates unchanged; Phase A supports weight-bandwidth binding (GDN
≈30-34 GB/s = lowest big stage; LM head 45; routed experts ~36).

FULL REPORT ARCHIVED: evidence/idea08-phase-a-analysis.md (verbatim tables;
the Phase A entry above is the summary). Window request recorded: one CPU
window on .80 for (a) the resident per-tensor BOM census + (b) the compact
IQ3_S decode tournament — measurement-only, queued behind the Main
campaign.
