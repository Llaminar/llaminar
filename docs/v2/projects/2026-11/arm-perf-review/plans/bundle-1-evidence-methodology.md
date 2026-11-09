# Bundle 1 — Evidence & methodology (measurement-only, no production change)

Status: PLAN-FIRST (2026-11-09) · Lane: measurement/qualification capacity (Mac
qual session); owner: arm-perf coordinate; reviewer checkpoint at end; no
merge (evidence + method changes only where pull-requested separately).

## Scope
- 09 Rounding-barrier evidence: produce the non-volatile rounding-barrier
  proof for the ARM CPU kernels (documented procedure + measurement; evidence
  file as the artifact; no code change required unless the evidence shows a
  defect — then a focused fix + regression via the normal lane).
- 20 Overlay collection methodology: codify the in-model collection rule /
  per-format E2E validation (method doc + example walkthrough on the 35B
  overlay plan; pull-requestable as docs-only if wanted).
- 01 Harness gaps: close the listed Perf__ gaps (list from 01; candidates:
  grouped verifier M sweep, lonely-row, steady-state decode with MTP).
- 15 NUMA-less / cache-unpublished host attribution: consolidate the WSL +
  DGX env sets (64K pages, memory-only NUMA nodes, single-socket) into the
  known-failure classification doc; add the Grace/Neoverse-V2 record; check
  whether any failure class is a real product gap on such hosts (report only).
- 19 Short-prompt prefill-vs-decode sweep on GB10 (idea 25-adjacent
  attribution; window via arm-sparks when the GPU is free).

## Verification bundle (measurement + decision)
- 14 Dense MTP verifier economics: re-verify on the new base (5.45-5.55 vs
  5.28 — MTP wins; confirm at the current tip + record the on-demand
  decision rule).
- 18 GDN grouped verifier economics: grouped-vs-serial <1.0 for all M
  (M=4 0.447 @20T) → decision record: narrow/disable grouped verifier for
  GDN or accept; if a change is warranted it goes to an implementation lane.

## Gates / method
- Measurement-only: no production src changes; all runs on existing hosts
  (.80 CPU, WSL x86, GB10 CPU side; GPU windows only for 19/18/14 where
  needed); every claim = raw log + per-file evidence + SHA; no budget or
  gate changes; final = reviewer checkpoint + consolidation into the
  tracker.

## Sequencing note
- Runs after R2 gates finish (hosts quiet) OR in parallel where CPU-only
  and no overlap; 19/18/14 need GPU windows (book via arm-perf after the
  GB10 CUDA gates).

## Executing (2026-11-09, operator GO)

- Lane: qual/measurement session 01a1bd53 (reopened after ideas-7+25 close;
  it offered). Start: 15 (doc) + Mac-fit (01/09) now; 19/14/18 deferred to
  GPU windows (book after GB10 CUDA gates at 7d1a94479; no self-booking).
  Rules: measurement-only; evidence = log + SHA + per-file; reviewer
  checkpoint. arm-sparks heads-up (future slots; no collision with R2).
