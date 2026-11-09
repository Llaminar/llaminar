# Idea 20: overlay admission rule and collection procedure

Status: methodology record. It is docs-only and makes no production change. Source anchors refer to
the ARM tip `a27c7ed4f` unless stated otherwise.

This record turns idea 20's lesson into an admission rule, a procedure and an evidence schema. The
lesson is that microbenchmark-selected exact overlays won in isolation but lost in the model. The
record applies the rule retrospectively to the retired GB10 table on the 35B plan.

## 1. Admission rule

An exact-overlay table (or any measured schedule-selection table) may be installed only when one of
these holds:

- **A. In-model collection.** Every installed key was observed and timed inside the production model
  run that will consume it. Collection keeps warm caches, the concurrent stages, and the actual
  `(codebook, superblock, N, K, M, workers)` keys the graph resolves.
- **B. Per-format end-to-end validation.** The complete candidate table passes an alternating
  present/absent A/B on the target host. The A/B uses a Release build of each affected source format
  and the production plan, and must meet all of these:
  - Output text is byte-identical. This is the correctness gate.
  - End-to-end decode and prefill tok/s improve or stay unchanged with mean±sd, n≥2 per arm,
    alternating order. This is the economy gate.
  - PerfStats proves that the table was actually consulted (see §4, gap R2).

A kernel-level win alone is not admission evidence (AGENTS.md: measure production paths; correctness
and economy are simultaneous invariants). A table that only some formats validate is installed for
those formats only. Unvalidated formats keep the total policy.

## 2. Why: the 2,519-key GB10 table

| Run | Commit | 35B MTP-dyn decode | 35B no-MTP decode | Prefill |
|---|---|---|---|---|
| A, baseline | `878a33fea` | 18.18±0.40 | 14.14 | 24.0–24.7 |
| B, +2,519-key overlay table | `144ece037` (on #43: `658277659`) | 17.45±0.21 (−4%) | 13.27 (−6%) | 22.6–22.8 (−7%) |
| X, head with the table | `15f87f755` | 27.35±0.06 | 20.54 | — |
| Y, head with the table emptied | `d397154da` | 28.51±0.29 (+4.2%) | 20.95 (+2.0%) | +9–11% |

Source: `12-cache-topology-tiling.md` and `00-measured-baseline.md` in this tracker, evidence
K14-OVERLAY-AB-HEAD. Text was byte-identical in all arms. The table was retired in `264970e14` (on
#43: `4f87ea04c`; the generated include is now an empty array).

The keys were selected by `Perf__CPUNativeNEONExactOverlay`. The collector times one prepared
projection at a time, at the invocation worker count, round-robin across candidates. It
authenticates bytes against serial M=1 rows. Selected keys showed ~20–25% kernel wins. The collector
is correct as a byte-authenticated candidate timer. What it cannot reproduce is how the projection
runs inside the model:

- other stages evict its weights and activations;
- concurrent stages and barrier counts change which tile and chunk count is cheapest; more chunks
  per task means more claimed tasks per region (ideas 5 and 13);
- the in-model `workers` and `M` for a projection come from the actual team and graph geometry
  (`automatic()` in `CPUNativeNEONGemv.h` takes `nativeQuantisedInvocationWorkerCount()`), not from
  the collector's process-wide setting.

## 3. What the tooling already guarantees (keep these)

Generator: `tests/v2/performance/kernels/native_quantised_dispatch/neon_exact_overlay.py`, schema
`neon-exact-overlay-v2`.

- **Byte exactness.** Every candidate output is compared with serial production rows before its
  samples count. Non-exact bytes fail closed.
- **Replication.** Each (runtime key, source alias) needs at least `--min-replicas` independent
  collector processes (default 2), so one run's winner's curse cannot install a key.
- **No fitting or interpolation.** An unmeasured key resolves to the total `Four`/`nbc=1` policy.
  The lookup is exact `lower_bound` equality (`CPUNativeNEONExactOverlay.h`).
- **Margins.** An entry needs at least `--min-improvement` (default 3%) geometric-mean gain and must
  not be slower than the total policy for any alias in any replica.
- **Build identity in the evidence.** The collector stamps `compiledNativeQuantisedBuildISAName()`
  and the resolved instruction family (fixed by `89adcc6c7`, closing the "NEON+DotProd" mislabel
  recorded in idea 20). Mixed build/instruction identities fail closed in `collect()`.
- **Receipt.** The generator records every input digest and every decision.

## 4. Residual gaps (ticket candidates; each needs `src/` work outside this bundle)

- **R1. Installed table is not bound to its build envelope.**
  - `render()` writes the ISA only into a comment.
  - The runtime key is `(codebook, superblock, n, k, m, workers)` with no envelope dimension, and
    there is no `static_assert` against the compiled envelope.
  - Consequence: a table collected on a NEON_I8MM binary and compiled into a NEON_DOTPROD build
    would be consulted silently. The arithmetic stays the same, since overlays are numerically
    neutral, but the economics are those of another instruction family.
  - Proposal: the generator emits the envelope/instruction as a `constexpr` identity, and the header
    `static_assert`s it against the compiled envelope (or the runtime key gains the dimension).
- **R2. No PerfStats proof of consultation or key census.**
  - Today nothing in `src/` counts exact-overlay hits versus total-policy resolutions. `NativeNEONSchedule::exact` is
    read by no `src/` code; its only reader is `Test__CPUNativeNEONProjectionBatch.cpp:510`.
  - This blocks rule A, because the in-model key census cannot be observed. It also blocks the
    PerfStats leg of rule B.
  - Proposal: a cold, aggregated counter per (stage role, codebook, M bucket, workers) with
    `exact_hits`/`total_policy`. Increments must be hot-path-safe: no per-call formatting or
    allocation, aggregated at request end (maxims 26 and 33).
- **R3. No in-model collection mode.**
  - Rule A needs candidates timed inside a running model.
  - The neutral way is to time whole-request decode with one key's candidate forced, the rest on the
    total policy, alternating, using the same plan and prompt.
  - This means a diagnostic-only per-key override in `DebugEnv.h`. It must never be a production
    knob, and the receipt must record it.
  - Until it exists, rule B (per-format end-to-end validation) is the only admissible path.

## 5. Procedure

**Collection (rule B path, available today)**

1. Pin one Release build. Record:
   - SHA;
   - `LLAMINAR_CPU_ISA` envelope;
   - compiler;
   - host record (CPU model, per-class cores, page size, NUMA layout);
   - the saved plan (`llaminar2 plan … --output plan.json`).
2. Microbenchmark collection with the existing collector:
   - at least 2 independent replicas per source format, in separate processes;
   - same worker count as the plan's CPU team;
   - core-pinned;
   - unprofiled.
3. Generate a candidate table with the generator. Keep its receipt.
4. Build two binaries that differ only in the generated include: candidate table present, and empty
   array. Verify by diffing the two include files plus the remaining-tree hash.
5. End-to-end A/B on the target host:
   - `llaminar2 benchmark --config plan.json`, same prompt bytes, `-n 256 --temperature 0 --seed 42`;
   - alternate A/B/A/B and at least 2 per arm;
   - include MTP-dynamic and no-MTP lanes;
   - include prefill.
6. Per format: install only formats whose arm passes both gates. Otherwise keep the total policy.

**Collection (rule A path, after R2 and R3)**

1. Run the production model once with the R2 counter, to obtain the in-model key census: keys,
   invocation counts and time share.
2. Rank keys by time share. For each top key, force each candidate in turn via R3 and time
   whole-request decode, alternating with the total policy.
3. Admit a key only on the same margin and not-slower rules as §3. Then run one rule-B confirmation
   for the combined table.

## 6. Evidence schema (one record per admitted table)

```text
table_id, generator_receipt_sha256, collector_schema (neon-exact-overlay-v2),
build: {sha, envelope, instruction, compiler, build_type=Release},
host: {cpu, core_classes, page_size, numa_nodes, cache_observation},
plan: {plan_sha256, model, context, mtp, workers_per_team},
collection: {mode: microbench|in_model, replicas, rounds, min_improvement},
e2e_ab: [{format, arm: present|absent, order_index, decode_tok_s, prefill_tok_s, text_sha256}],
perfstats: {exact_hits, total_policy, by_key},          # R2
verdict: {admitted_formats, rejected_formats, reason}
```

## 7. Worked example: the 35B GB10 plan replayed against the rule

| Step | What happened | What the rule requires |
|---|---|---|
| Key source | 20-worker microbenchmark keys, 2 replicas, byte-exact (`144ece037`) | Allowed as candidate generation only |
| Shapes | The manifest's exact-overlay set (`native_quantised_decode_shapes_v5.json`, 84 `exact_overlay` shapes). Only `35BMoE_Expert_GateUp` (512×2048) and `35BMoE_Expert_Down` (2048×512) are 35B-specific. | Rule A would replace this list with the in-model key census (R2) |
| Install gate | Kernel wins ~20–25% on selected keys | Rule B: per-format end-to-end A/B before install. This was **not** run before `144ece037`. |
| Outcome | End-to-end −4% MTP, −6% no-MTP, −7% prefill. Retired after the head A/B (+4.2% / +2.0% / +9–11% without it). | Rule B would have rejected the table before install |
| Envelope | Collector then labelled I8MM builds "NEON+DotProd" | Fixed in collection (`89adcc6c7`). Install-side binding still open (R1) |

Conclusion for the 35B plan: with today's tooling only rule B is executable. The retired table would
have failed it in every lane. No overlay should be re-collected for GB10 until R1 and R2 are
installed, so that a candidate's consultation and envelope are provable rather than assumed.
