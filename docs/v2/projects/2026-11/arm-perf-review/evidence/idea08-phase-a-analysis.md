# Phase A analysis (idea-08 addon session, source-audit at tip bdbb8e370 = bb893cd79 + B1 bench; layout code identical to bb893cd79)

Measurement-free static audit per the approved plan: exact per-encoding BOM
from the current layout code, ARM zero-reader verification for the
compensation section, the ABI-version mechanism refined to concrete
identity flow, and the diagnostic compact-decode tournament design
(non-installed; its RUN needs the .80 window — booking request at the end).

(Verbatim full text received from arm-addon2 seq 14 attachment; the
complete tables and sections are preserved in the git history of this
file at commit time. Key tables summarized in 08-arm-prepared-weight-
layout.md Phase A entry.)

## Key tables (summary)

Unit = 64 output columns × one 32-value K block = 2048 logical weights.
Per-unit layout: [groups×4×64B payload | 128B comp_int16 | 128B scales_fp16
| (+128B mins_fp16 if asymmetric)].

| Encoding | Stride B | comp share | prepared bpw | catalog bpw | expansion |
|---|---|---|---|---|---|
| NibbleLUT sym (Q4_0, IQ4_NL) | 1280 | 10.0% | 5.0 | 4.5 | 1.11× |
| NibbleLUT asym (Q4_1, Q4_K) | 1408 | 9.09% | 5.5 | 5.0 | 1.10× |
| NibbleLUTSourceScale (IQ4_XS) | 1408 | 9.09% | 5.5 | 4.5 | 1.22× |
| ExpandedInt8 sym (Q8_0/Q8_1/Q8_K/Q5_0) | 2304 | 5.56% | 9.0 | 8.5/5.5 | 1.06×/1.64× |
| ExpandedInt8 asym (Q5_1/Q5_K) | 2432 | 5.26% | 9.5 | 6.0/5.5 | 1.58×/1.73× |
| IQ3SExpandedSourceScale (IQ3_S) | 2432 | 5.26% | 9.5 | 3.75 | 2.53× |
| Q6KNativeDualScale (Q6_K) | 1792 | — | 7.0 | 7.0 | 1.00× |
| CompactMultiScale (Q3_K/Q2_K) | 1536 | 256/512 pad | 6.0 | 3.5 | 1.71× |
| CompactMultiScale (IQ2_S/IQ2_XS) | 1536 | 448 pad | 6.0 | 2.75 | 2.18× |
| CompactMultiScale (IQ1_M) | 1536 | 640 pad | 6.0 | 2.0 | 3.00× |
| IQ2_XXS → ExpandedInt8 sym | 2304 | 5.56% | 9.0 | 2.5 | 3.60× |
| IQ3_XXS → ExpandedInt8 sym | 2304 | 5.56% | 9.0 | 3.5 | 2.57× |
| IQ1_S → ExpandedInt8 asym | 2432 | 5.26% | 9.5 | 2.0 | 4.75× |

Two dead-byte classes (previously conflated):
- comp_int16 (128 B/unit, ARM-dead only) — packer WRITES comp on every ISA
  (WeightPacker.h:247, :620-644) → ARM preparation wastes the compute
  filling dead bytes; ARM contract signed×signed, no u8xs8 correction
  consumer; zero "compensation" occurrences in any NEON TU.
- CompactMultiScale fixed-payload padding (dead on BOTH ISAs): 768/512/
  576/576/384 B native payloads in a fixed 1024-B region → 256-640 B
  structural padding = 17-42% of the 1536-B unit — NEW to the BOM;
  addressable by the same compact-layout family (both-ISA gain).

## 35B-A3B aggregate (documented pairing; exact census = window item)
One routed expert ≈ 2.30 MB prepared vs ≈ 1.10 MB catalog-source (2.09×);
whole-model ≈ 23.6 GB prepared vs ≈ 11.3 GB source. Compact wins target
BOTH the IQ2_S gate/up padding and the expanded formats. Exact per-tensor
census = the runnable resident-BOM audit on .80.

## ABI mechanism
PreparedLayoutIdentity { cpu_family, layout_version, encoding_set_rev }
flowing through: preparedEncodingForSource/preparedInterleavedBlockStride,
WeightPacker (writer), WeightView (reader), PreparedFootprint (BOM/
pricing), cache keys + persistent corpora, wire/slot/movement identity
(ExpertTierWeightStream, MoEOverlayPreparedWeightSource, CpuExpertSlotPool,
ExpertPreparedMemoryGeometry), evidence seals. Same-version movement
unchanged; mixed-version meshes fail fast at rank admission with
re-prepare-from-GGUF; x86 keeps comp + versions independently; identity
rides the EXISTING publication/admission surfaces (no new side-channel);
fail-fast point = the rank-admission check that today validates source
identity.

## Diagnostic compact-decode tournament (non-installed, rule-166-conformant)
- Compact IQ3_S primary: raw payload 832 B + 128 B raw-coefficient plane +
  128 B scale/base = 1088 B/unit live floor = 4.25 bpw (2.24× below 9.5);
  tighter interleaving ≈3.9-4.1 bpw; vqtbl in-register decode; byte-oracle
  per unit.
- CompactMultiScale padding-aware variant (secondary; IQ2_S first).
- Matrix: M=1 + M=15 (+ M=4 ALU check), IQ3_S/IQ2_S/IQ4_XS, all three ARM
  families, mixed@20T + fast@10T.
- Report: decode µs vs current prepared decode at identical M, achieved
  bpw, projected engine win through the Phase-A attribution (weight-
  traffic-bound stages only). Diagnostic-only: non-installed namespace, no
  register gate programs, no production paths; Phase B gated on the plan
  thresholds (≥+5% engine (A) / ≥+15% projected 35B (B) + M=4 ALU check +
  idea-16 attribution).

## GO/NO-GO gates (unchanged; Phase A feeds them)
Phase A adds no gates. Phase B conditional GO requires: idea-16
attribution still weight-bandwidth-bound (FFN 26.7% + GDN projection 23.2%
+ LM head 10.2%; GDN ≈30-34 GB/s = lowest big stage; LM head 45; routed
experts ~36) AND the tournament/engine thresholds AND the M=4 ALU check.
NO-GO triggers unchanged.

## Window request (when the current queue clears)
One CPU window on .80: (a) resident per-tensor BOM census on the loaded
35B (Release build + BOM tooling, no model-campaign runs); (b) the
compact-decode tournament (diagnostic bench build at the then-current
tip). Both measurement-only; no ABI/layout change/production paths.
