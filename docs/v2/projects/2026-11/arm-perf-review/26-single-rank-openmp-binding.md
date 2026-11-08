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

## Decision (Main, 2026-11-08): option (a), two conditions

1. **One implementation.** Factor the binding in `configureClusterRankThreads`
   into one helper used by EVERY bootstrap path that sets OMP_PLACES=cores /
   OMP_PROC_BIND=close (single-rank CPU, explicit `cpu:N`, GPU-NUMA cpu-set,
   shorthand). Keep the existing OpenMP-partition authority: bind within the
   rank's admitted mask (`sched_getaffinity` /
   `ScopedOpenMPPartitionAffinity`), never widen it. On hosts without
   `cpu_capacity`/`cpufreq` the order stays plain CPU-id order — x86 unchanged.
2. **Proof.** Device-free ordering test with a fake sysfs root
   (`capacityOrderedCpus` already takes one) + the .80 sched_getcpu-per-member
   check in single-rank serve and with explicit `cpu:0`; registered in
   `ProductionTestPreflight`. Rerun `V2_Integration_CPUStartupAffinity` and
   `OpenMPIdleSpinPolicy` on .80; measure 35B MTP-off decode before/after (must
   not regress — dynamic tiles already balance load; the gain is in serial and
   static work).

Implemented by arm-i8mm in commit 3 on top of `b2cf793bf`; Main reviews the
diff by SHA.
