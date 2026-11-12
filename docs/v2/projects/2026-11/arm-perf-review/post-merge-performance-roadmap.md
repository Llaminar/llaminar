# Post-merge performance roadmap

Created: 2026-11-12 (coordinator synthesis, operator-commissioned full review).
Context: PR #43 at tip `d5e04fdce` (B1 pooling landed per operator override)
awaiting the base owner's develop merge. This document is the durable form of
the full review delivered to the operator at ~13:50Z; it supersedes nothing —
it ORDERS the open ideas by expected impact and defines the launch sequence.
Acceptance bar for every item (operator, 2026-11-12): **same-or-better
performance, anything worse rejected; x86 performance must stay the same.**

## Where the measured gaps are (evidence-bound)

| Gap | Measured | Proven mechanism |
|---|---|---|
| GPU prefill vs llama.cpp (GB10) | 319 vs 1063 tok/s (3.3×) | ~80% = captured bucket-pipeline overhead (replay/launch/sync per boundary); kernels alone support ~1660 tok/s (0.60 vs 3.009 ms/token) + ~279 ms fixed per-request intercept (ideas 24/25) |
| CPU decode vs llama.cpp (GB10) | 22.8 vs 28.8 tok/s (~26%) | Bandwidth-bound; IQ3_S prepared weights expanded ~9.5 bpw vs 3.44 bpw source (~2.7× bytes); dead compensation bytes (NibbleLUT 10%, ExpandedInt8 5.6%) (idea 8) |
| MTP on the 35B MoE | MTP-dyn 18.5 vs no-MTP 25.2 tok/s (−27%) | LM head = full-vocab M=1 GEMV per verifier row (20.1% of MTP-dyn decode, zero amortization; a grouped path exists but isn't taken); grouped GDN ~1.0×; MoE transaction ~1.8×; mode-artifact (38.9% stochastic acceptance vs 99% greedy @ depth 7) unresolved (ideas 14/16/18) |
| M=1 decode | 1T +6.9% vs mixed-20T amortization | Activation-reload headroom (idea 4 residual; M=1 window = the decision input) |

## Dead ends (do not revisit)

SVE2 2× width (X925 VL capped 128-bit, probe-sealed); cache-topology tiling
(measured regression, retired); I8MM on Q6_K (~0); overlay microbench wins
without in-model validation (methodology rule, idea 20).

## Tier 1 — structural prizes (large refactorings, operator-authorized)

1. **Idea 25 — prefill bucket-pipeline overhaul.** Up to ~3× GPU prefill.
   Two structural candidates: single-bucket long-prompt capture + async
   multi-bucket submission (overlap replay with prior-bucket work). Design
   lane SPAWNED (arm-p25design, GLM-5.3 max, 2026-11-12) — deliverable =
   design + phased plan + non-attached PerfStats measurement design (NCU/NSYS
   on safety HOLD) + x86 A/B gate. Shared engine code → x86 same-or-better
   is a hard gate, measured not assumed.
2. **Idea 8 — compact ARM prepared-weight layout.** The 35B decode reads
   ~2.7× more weight bytes than its quantization needs. Phase A plan approved
   + GO. Inputs: (a) census = estimator-based SIZING input in flight on .79
   (per-tensor predicted/estimator columns + observed aggregate BOM, strictly
   labeled — authorized because estimator byte-equality is a tested production
   invariant; the OBSERVED per-tensor enumeration needs a debug-namespace
   seam (StageDumper-pattern reader over the prepared store's per-tensor
   publication) = post-merge, normal review path, added as an idea-08
   dependency); (b) tournament bench on SOURCE HOLD — repair lane in flight
   (arm-i08repair) for the four reviewer findings, then reviewer ACK, then
   sparks re-run. ABI-bearing (packer → view → footprint → cache keys →
   wire/movement); per-ISA layout identity keeps x86 untouched by
   construction.

## Tier 2 — MTP economics (cheap-to-medium, high certainty)

3. **Greedy MTP-dyn rider — SETTLED (2026-11-12)**: the −27% is REAL, not a
   sampling artifact (greedy rider seal: acceptance 38.89% fully greedy,
   18.39 vs 25.2 tok/s). **Decision it feeds: a depth-policy/model-class-
   aware default (MTPDepthDefaults machinery) recovers most of 27% with
   zero kernel work — now the top quick-win candidate.** NEW follow-up:
   identify why draft acceptance collapsed (38.89% @ depth 10 now vs 99%
   @ depth 7 recorded in an earlier protocol epoch). T2 control (same-day
   no-MTP arm) authorized to complete the pair.
4. **Route verifier rows through the existing grouped LM-head path**
   (byte-neutral, ceiling ~6.5 ms/token ≈ +16% MTP-dyn decode; LMHeadStage
   has the path, the verifier route doesn't take it).
5. **B2 — shared-expert one-team transaction** (~2× headroom, two regions
   remain). Stacks with B1's landed pooling mechanism.
6. **Idea 14 B-wave depth tuning** (booked; 27B MTP wins on the new base —
   depth tuning is the remaining margin).

## Tier 3 — mechanism extension + regression fixes (cheap)

7. **B1 family extension** to remaining per-member fork sites + 35B layout
   mitigation (hot-function alignment/reorder; the merged base re-links
   anyway — re-measure before investing).
8. **Idea 28 — IQ4_XS M=1 multi-thread regression** (standing regression,
   pure fix; source-level investigation startable device-free).
9. **Idea 22 — FA2 tournament policy tail** (p95 regret 8.9%, tq4_tq4 17.6%;
   GPU decode attention policy).

## Tier 4 — medium projects, evidence-gated

10. **Idea 4 residual — M=1 activation reuse** (input = the running M=1
    window on .80). 11. **Idea 5 — typed X925/A725 scheduling**. 12. **Idea 2
    — live page-cache admission** (memory → context/KV headroom). 13. **MXFP4
    lane** (post-merge, operator-kept open): W4A4 halves weight traffic for
    that model class; the DeepSeek prerequisite.

## x86 protection classes (applies to every item)

- **ISA-fenced** (NEON kernels, ARM layout identity, panel widths): x86
  untouched structurally; WSL AVX2/AVX512 gates + budget/exactness authorities
  prove it.
- **Shared-engine byte-neutral** (pooling, verifier routing, B2): the bar
  applies symmetrically — x86 A/B before landing.
- **Shared pipeline architecture** (idea 25): x86 GPU A/B is part of the gate.

## Launch sequence (post-merge)

1. Immediately: **MTP policy decision** (mode-artifact settled — real
   regression; the policy fix is the top quick win) + acceptance-collapse
   investigation; idea-28 investigation; B1 layout mitigation attempt
   (cheap experiments first).
2. Parallel structural lanes: idea-08 (census → repaired tournament → reviewer
   ACK → implement) and idea-25 (design → counters → phased implementation).
   Station = fast implementer, Spark-first verification (operator directive).
3. Then: verifier LM-head routing, B2, activation reuse (on the M=1 result),
   idea-22 tail, heterogeneous scheduling.

## Lane assignments (2026-11-12)

- **arm-i08repair** (spawned, GLM-5.3 max): tournament bench repair on
  codex/arm-idea16-tournament-repair; reviewer ACK gate before any re-run.
- **arm-p25design** (spawned, GLM-5.3 max): idea-25 design doc on
  codex/arm-p25-design; arm-perf integrates into the tracker.
- **arm-sparks**: hosts (.79 census + tournament re-run post-ACK; .80 M=1
  window now, then the MTP rider).
- **Main**: on-call for merge-attributed fix-forwards only until the merge.
