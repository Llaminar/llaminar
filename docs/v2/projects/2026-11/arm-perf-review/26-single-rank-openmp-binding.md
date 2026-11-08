# Idea 26 — Single-rank OpenMP binding on GB10 is A725-first (capacity-ordered binding needed)

Status: **Open**. Found 2026-11-08 by the arm-i8mm lane while designing idea 5's
FastestClass placement; code-reading finding, not yet measured.

## Finding

- `MPIBootstrapPhase::configureClusterRankThreads` binds workers
  fastest-first (`capacityOrderedCpus` + `sched_setaffinity` by omp thread id)
  — but only on **cluster paths**: hostfile / rank-selection / automatic
  multi-rank discovery.
- An **ordinary single-rank GB10 serve** with `OMP_PLACES=cores`,
  `OMP_PROC_BIND=close` gets **libgomp's CPU-id place order**: threads 0–4
  land on CPU 0–4, which are the **A725** cores (`core_id` ordering from the
  handoff: A725 = CPU 0-4, 10-14). So "thread ids 0..w-1 = X925" is **NOT
  guaranteed** in the common single-rank path.
- Consequence: worker 0 and any static partition can land on slow cores in
  single-rank serving; a FastestClass placement must not assume a fast prefix.

## Options

- **(a)** Apply the same capacity-ordered binding for **every rank shape**
  (single-rank included). This is a **base-behaviour change** — needs the ARM
  implementer's decision. It also fixes worker 0 / static partitions landing
  on A725 in single-rank serving (independent of the typed placement).
- **(b)** Per-thread class probe published at bootstrap (sched_getcpu →
  inventory class, cached once per team member); `FastestClass` **refuses to
  engage** (hard error or explicitly unadmitted placement) unless the fast
  workers form a prefix.

The lane prefers (a). Verify on `.80` (sched_getcpu per team member) before
deciding — the lane will run that check.

## Validation

- Recorded GB10 CPU topology provides the expected mapping; a runtime check on
  .80 for OMP_PLACES=cores / bind=close single-rank serve; then A/B of
  binding change: single-rank 35B/27B decode+prefill, worker classes measured
  (X925-only / A725-only / mixed).

## Implemented (2026-11-08) — binding-only SHA a7f719dfb (parent afe62c5ef)

- One helper bindOpenMPWorkersFastestFirst + capacityOrderedCorePlaces(mask)
  (one place per physical core, allowed siblings only); used by the cluster
  path and every self-launched OMP_PLACES=cores/close rank.
- Proof: V2_Integration_OpenMPCapacityOrderedCorePlaces (GB10 layout/SMT/
  partial mask) registered in preflight; Test__MPIBootstrap
  CorePlacesFollowCapacityWithinTheRankMask; Test__CPUStartupAffinity now
  asserts every full/half/single close team on capacity-ordered places
  (failed before: worker 0 on CPU 0; passes after; localhost:-1:cpu:0 /
  cpu:0 / localhost:0:cpu:0 on .80).
- Engine effect (C→B, 35B): no-MTP 23.42→23.55, MTP 33.94→34.01, prefill
  noisy — no regression, ≤ +0.5% (dynamic tiles already balance load; gain
  is in serial/static work, as Main predicted).
- Status: pending Main's review/cherry-pick onto PR #43. Rebase conflict with
  the upstream fused recorder was resolved inside the lane's tip
  (recordNativeNEONFusedLaunches, 21-tag schema).

## Taken by Main (2026-11-08) — binding is in PR #43

Main reviewed and cherry-picked a7f719dfb; PR #43 is now a7f719dfb (same
commit object, fast-forward). Local pass: OpenMPCapacityOrderedCorePlaces,
MPIBootstrap*, OpenMPIdleSpinPolicy. The i8mm lane rebases onto a7f719dfb and
drops its binding commit; remaining tip goes to the gate.

## Defect in the taken binding + fix (2026-11-08)

a7f719dfb reads LLAMINAR_SELF_BOOTSTRAPPED/OMP_PLACES/OMP_PROC_BIND via
std::getenv in MPIBootstrapPhase.cpp → fails the getenv-hygiene sanitizer
(V2_Unit_Static_NoDirectGetenvOutsideDebugEnv, V2_Integration_
HostServiceDebugEnvAuthority). Fix offered: ef19d19f1 (DebugEnv::envValue,
3 lines, behaviour unchanged; clean cherry-pick onto 0868a5573). Expected
gate set includes these two as base-class until Main takes the fix.
