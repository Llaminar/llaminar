# QFN hybrid attribution after OpenMP placement corrections

The current unprofiled Release checkpoint measures **314.868 prefill /40.035
after-prefill decode tok/s** on two MI50s and two CPU socket ranks. The target
remains 500/40. The [placement note](QFN_CPU_OPENMP_TEAM_PLACEMENT_20261110.md)
owns that comparison and its limits. This follow-up measures existing host
scopes to choose the next optimization; it changes no execution policy.

## Workload and collection

Original Release owner, Qwen3.8-Flash-Next UD-Q4_K_XL plus Q8_0 MTP sidecar;
context262144, 512 input/256 output tokens, seed42, one warmup and three measured
requests. Both MPI socket ranks retain 28 physical workers; the NUMA1 rank owns
the two local MI50s. Dynamic MTP and expert movement, RAM ngrams with local GPU
caches, and 16/32GiB RAM/disk prefix tiers stay enabled. Initial expert capacity
and placement match the unprofiled checkpoint.

The existing `cpu_moe_grouped_rows`, `retained_parent_transaction`,
`moe_overlay_endpoint` and `moe_overlay_participant_graph` filters enable their
own host clocks. General CPU/GPU stage timing and TP timing stay disabled.
Existing narrow residency counters accompany those clocks. No new timer,
execution owner, synchronization, profiler event or observation thread was
added to production. Rank-qualified final PerfStats files are read only after
original native, driver and process-tree retirement.

Fresh admission and the native lifetime pass. All 768 output IDs, complete MTP
records and resolved initial expert plans match the unprofiled checkpoint.
The two output paths are the only runtime-configuration differences. Diagnostic
rates are 311.024/38.150; added clocks/counters and dynamic movement make these
attribution results, not a new canonical throughput comparison.

## Current root prefill scopes

Totals below are divided by three measured requests. Actual root gather calls
are288: 48 main layers × two disjoint448/64 chunks × three requests. Component
call counters, rather than retained-transaction timer counts, own normalization.

| Existing scope | ms per512-token request | Meaning |
|---|---:|---|
| Captured dispatch input readiness |711.626|Wait for the GPU-produced input ticket; overlaps producer work.|
| CPU endpoint packet service |564.647|Encloses validation, setup, synchronous compute and return handling.|
| CPU compute submission |551.711|Nested complete CPU compute-stage execution.|
| Native complete expert FFN |467.892|Nested gate/up, SwiGLU/Q8 and down with the existing single OpenMP team.|
| Transported router-Q8 publication |33.802|Unique input quantization and required duplicate Q8 row copies.|
| Canonical route-slot publication |29.149|Reorders raw down results into their declared route destinations.|
| Route scheduling and validation |8.610|Serial metadata preparation, excluding publication preparation.|
| Root shard gather |106.468|Includes peer readiness as well as transfer.|
| Root fused combine/preweight/publication |85.675|Actual fixed-order combination and result publication.|

Do not add nested endpoint/FFN scopes or overlapping waits into a proposed
speedup. Follower records include a different lifetime cohort and remain
unnormalized; they are not another three-request total to add to the root.
The generic retained-parent envelope currently labels all roles `decode`;
only the explicitly typed root children above supply semantic phase identity.

The diagnostic prefill is approximately1646ms/request. Reaching500tok/s needs
about622ms less elapsed time. Even deleting the entire measured468ms root FFN
would be insufficient, before accounting for overlap. The largest unresolved
interval is input readiness. A current continuation/GPU timeline is therefore
the next priority; small CPU publication/fork-join changes alone cannot close
the gap. Existing source inspection confirms that unique input rows are not
requantized per expert, and gate/up, activation and down already share one team.

## Movement interpretation

The final root PerfStats contains five proposals and five commits through
epoch8, with20promotions. Measured request brackets contain only four waves
and16promotions through epoch7. The additional commit belongs to the tail;
do not count it as measured-request movement. There are177 payoff rejections
and one residency rejection in the final counter cohort. These counts and
projected cost fields neither measure planner latency nor prove a profitable
backlog beyond the four tier exchanges per wave.

Increasing physical transfer slots from5to9 would add192MiB of staging per
GPU, plus host staging and shared-shadow capacity, under the existing typed
admission plan. It could reduce resident expert capacity. No larger fabric or
controller policy is installed; current evidence does not yet justify that
experiment ahead of producer attribution.

## Local evidence and cleanup

- `/tmp/qfn-current-root-timing-next/observed`: original admission, model,
  native/resource/driver and retirement records.
- `analysis.json`, `INTERPRETATION.md`, `model-comparison.json` in the same
  experiment root: closed phase totals, cohort limits and output comparison.
- `/tmp/qfn-cpu-ffn-region-audit-next/NOTE.md`: source ownership/region audit.
- `/tmp/qfn-moe-wide-wave-next/PROPOSAL.md`: existing staging/shadow coupling.

After retirement,727,912,507bytes of owned prefix files were reclaimed by
reading28,683metadata bytes and zero payload bytes. No cache hashing or model
payload scans were performed. Full preflight and the OpenCode application gate
remain deferred; this diagnostic supplies neither certificate.

## Fresh ROCm continuation trace

The selected SDK10.1 profiler prefixes the original application, retaining its
automatic MPI bootstrap and canonical session owner. The accepted resolved
placement remains fixed. A separate512/16 diagnostic uses one warmup and one
measured request; no profiler rate replaces the unprofiled baseline. Warm
attachment was not used because the pinned SDK excludes descriptor queues
from its supported late-attachment path.

Both actual MPI processes, their discovery/execution mapping and mapped
GNU/HIP/HSA/profiler providers are observed. GPU execution rank0 is discovery
rank1, PID1267620 in this lifetime. Its PID-qualified ROCPD3.0.4 database is
distinct from the CPU rank's startup trace. Native, driver, observer and
process-tree retirement pass. All16 generated IDs match the control prefix;
MTP performs12accepted drafts and4verifier runs over16rows. Initial placement
is identical.

The existing correlation reader associates96,492dispatches with320graph
launches, with zero count or agent mismatches. Actual ready/replay probes and
the preparation/warmup/measurement sequence identify measured prefill's
7,995root and6,361peer dispatches; other graph families remain unclassified.

| Measured GPU interval | Root MI50, ms | Peer MI50, ms |
|---|---:|---:|
| All-kernel union |1611.728|1611.274|
| Source-proven pure wait |698.991|0classified|
| Other-kernel union |912.737|1611.274|
| NativeQuantised GEMM union |202.741|179.992|
| Q8 residual correction |75.822|75.963|
| Canonical CPU-result materialization |130.839|absent|
| RCCL compute/communication/wait union |154.427|1036.984|
| Pooled immutable lookup union |47.687|47.290|

The peer's long collective interval overlaps683.465ms of the root's pure
wait. It therefore cannot be called an independent1.04second communication
cost. Kernel sums, interval unions and host dependency waits remain distinct;
these rows are not an additive wall-time breakdown.

The result confirms substantial CPU-result readiness alongside multiple GPU
costs. Ngram `coordinatorPlanFree` takes13.137ms in two launches with four
threads; `coordinatorPhase<9>` takes32.058ms in two single-thread launches.
The latter is `CompleteVictims`, not a wait loop. Its queue update walks every
original observation and only collapses adjacent duplicates. The former
updates four independent frequency lanes. These are live-policy operations,
not cache-capacity scans:448/64 token chunks carry7168/1024 original ngram
observations even when their unique rows hit. Exact batching of already
protected reads is being investigated; pressure, promotions, counters and
timestamps must remain equivalent to the original ordered policy.

Canonical CPU-result materialization reads exactly the published live route
count times the hidden width. The measured168,257 entries at width2560 carry
1,722,951,680bytes, giving13.17GB/s. Retained ready-payload measurements of the
same original transport are13.24–13.27GB/s for large cases. The materializer
has no polling loop and already issues two independent HIP scalar loads.
There is little evidence for a useful additional vectorization campaign;
reducing CPU-served routes is the more promising way to reduce this transfer
cost. This comparison does not establish a universal PCIe bandwidth ceiling.

No new kernel or controller policy is installed from these timings alone.

Closed source/SDK/identity and trace records are under
`/tmp/qfn-current-rocm-timeline-next/`; `analysis/primary` and `analysis/prefill`
retain graph correlations, phase proofs, actual launch resources and interval
intersections. Original trace generation and retirement take284.210seconds,
including model loading and profiler finalization. After retirement another
727,912,506bytes of prefix blobs are reclaimed with28,682metadata bytes read
and zero payload bytes. The trace database is retained for focused analysis.

## Observed tier-selection cut

The optional `moe_overlay_residency.precut_tier_candidate_cut` leaf reads the
existing scores and selection order before the tier cut. It does not repeat
scoring, serialize plans, change the controller or enlarge transfer storage.
Disabled filters avoid construction of its metadata. It reports isolated
eligibility separately from the final admitted transaction's coherent score.

The original-authority regression fails at the absent witness on the original
body, then passes with the leaf installed. It exercises both a capped positive
tail and complete candidate exhaustion. The full authority unit target and
four focused integration registrations pass in3.038seconds, including the
explicit `V2_Integration_MoEPrecutTierCandidateWitness` preflight entry.

The original512/256 model owner then passes at313.964P/39.584D, with all768
output IDs, full MTP records and initial expert placement matching the accepted
314.868/40.035 checkpoint. Only output paths differ in runtime configuration.
This counter-enabled run is diagnostic, not a newly promoted timing result.

| Observed generation / source epoch | Full tier candidates | Selected | Unvisited isolated-positive candidates |
|---|---:|---:|---:|
|3/3|4257|4|3877|
|4/4|4266|4|3191|
|5/5|4263|4|3829|
|6/6|4299|4|3935|
|7/7|4284|4|3264|

All five observations use the existing five-slot physical fabric and four-tier
cycle budget, with no preceding shadow rejection. These are individually
positive scores against each source epoch, **not a jointly feasible backlog**;
their forecasts must not be summed. Final records include the tail proposal,
whereas the three measured requests contain four waves and16promotions.
Authored proposals during measurement are separated by roughly4.7–6.7seconds
and already use new observation generations. Reusing an old window only when
new evidence is absent may therefore be insufficient: planner cost and bounded
wave scheduling need investigation before selecting a behavioral change.

Artifacts are under `/tmp/qfn-moe-precut-tier-witness-next/`: original model
and admission owners, `witness-analysis.json`, `model-comparison.json`, and the
source-only continuation analysis. Focused RED/GREEN receipts are under
`/tmp/qfn-cpu-eight-row-rocm101-root/precut-witness-*`. The native lifetime
retires cleanly in251.716seconds;727,912,503bytes of prefix payload are reclaimed
using28,679metadata bytes and zero payload reads. Kernel, controller and fabric
policies remain unchanged by this diagnostic slice.
