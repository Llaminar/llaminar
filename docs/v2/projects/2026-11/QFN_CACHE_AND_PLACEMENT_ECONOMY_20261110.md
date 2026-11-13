# QFN protected-hit cache and exact placement economy

This slice reduces existing work without changing cache policy, placement
objectives, migration capacity or numerical precision. The full-model result
is **314.818 prefill /39.988 after-prefill decode tok/s**: essentially unchanged
from the accepted **314.868/40.035** control. The 500/40 hybrid goal remains open.
The component wins below are real, but are not whole-model speedup factors.

## Fixed workload and ownership

Release, selected ROCm 10.1 SDK, two MI50s on their owning NUMA socket and two
physical CPU socket MPI ranks, 28 physical workers/socket. The original
application performs automatic MPI bootstrap. Both compute processes load
GNU `/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0`.

The model is the original four-part Qwen3.8-Flash-Next UD-Q4_K_XL GGUF with
the shared Q8_0 MTP sidecar. Context is262144; prompt/output512/256, seed42,
one warmup and three measured requests. Dynamic MTP and expert balancing stay
enabled. RAM ngrams, ConsumerScheduled local GPU caches and16/32GiB RAM/disk
prefix budgets remain unchanged. Fresh automatic admission and actual initial
placement agree on8066 GPU experts,4033/card, with the remaining17022 logical
experts tensor-sharded across the CPU cohort. No manual rank launcher or
memory-budget override is used.

## Protected-hit queue batching

The original CompleteVictims operation performs recency updates in original
observation order. Already-protected hits can instead publish each distinct
entry once in final-occurrence order, provided the complete real frontier is
successful, pinned, all-read, within the protected limit and free of pending
promotion/demotion. The existing original timestamps serve as the reverse
visit marker. No buffer, host readback, launch, allocation or separate cache
authority is introduced. Other legitimate policy states retain their original
ordered implementation.

The batched join reuses the scalar append-boundary validator. Review caught
that the initial candidate omitted checks on an untouched residual tail,
including out-of-range links and future timestamps. The focused regression
now requires sticky failure and rejection of subsequent reuse. Adjacent
repeated IDs are coalesced in the reverse scan. Full state comparisons cover
unequal positive owner capacities plus empty owners, interleaved and contiguous
7168-row histories, partial/empty requests, reset, pressure and clock errors.
Native reply extents also use the declared16-row reply capacity independently
of the fixture's32 physical cache slots; production validation is unchanged.

The matched original-bootstrap ROCm trace uses512/16, one warmup/measurement,
the same448/64 prefill chunks, output IDs, MTP work and initial plan. The
actual GPU process is authenticated separately from the CPU rank/bootstrap.
All96,491 declared dispatches associate with320 actual graph launches, with
zero count/agent mismatches. Measured prefill contains7995 root and6361 peer
dispatches, identical counts to the earlier trace.

| Measured prefill component | Original | Final | Comparison |
|---|---:|---:|---|
| Root CompleteVictims, two launches |32.058ms|4.857ms|6.60×;27.202ms removed|
| Peer CompleteVictims, two launches |31.374ms|4.788ms|6.55×;26.586ms removed|
| Root complete pooled-lookup union |47.687ms|20.677ms|Same actual lookup workload|
| Root pure-wait union |698.991ms|665.235ms|Includes changed background timing|

Action9 retains grid1/block1, with112 SGPRs,16 VGPRs and zero private bytes in
the final gfx906 trace. The compiled spill guards pass. These ordered metadata
operations have one live policy writer; the launch is not a capacity-sized grid.
Kernel unions, host waits and independent GPU intervals must not be added into
a projected wall-clock saving. Instrumented rates do not replace the unprofiled
model comparison.

The final optional native timing owner records31 fully checked warm transactions
after two conditioning frames and five warmups:17.320ms median on gfx906 and
7.048ms on gfx1100. These are entire transactions, not Action9 alone. The initial
unhardened candidate measured16.173/7.094ms; the final MI50 component cost is
higher, while still showing the above large improvement against the original
production queue. No CUDA memory-health claim comes from the private AMD-DRM
observer: its earlier CUDA timing lifetime retired successfully but that
auxiliary observer cannot certify CUDA VRAM visibility. Final CUDA correctness
is established by the actual native replay registrations below.

## Exact placement and owner ranges

For two tiers and fixed quota q, selecting q experts for tier0 minimizes

    constant + sum_selected(cost(expert,0) - cost(expert,1)).

Sorting the existing checked lexicographic differences therefore preserves the
exact phase-weighted integer service objective, incumbent-movement tie break
and deterministic expert ordering. Both tier costs are validated even for
zero/full quotas. More than two tiers retain the original general min-cost
flow implementation. This specialization is about the objective structure,
not a two-device or two-rank assumption.

`MoEExpertOwnerMap::placementsForLayer` borrows the already canonical immutable
owner vector using two binary searches. Migration scoring reads this range
instead of scanning every model layer for each candidate. Whole ownership and
CPU shard cohorts remain represented by their original rows; there is no new
index, placement cache or accounting ledger.

Same public planner, pinned physical CPU28, authored512-expert/quota168 cases,
reversed comparison orders and six measured samples per geometry:

| Component | Original median | Final median |
|---|---:|---:|
| One layer, exact two-tier assignment |29.264ms|0.0868ms|
|49 layers, exact two-tier assignment |1433.326ms|3.122ms|
|4257 owner-layer visits,49×512 relation |316.754ms|6.952ms|

The first two rows compare the same two-tier public owner before/after. The
zero-quota third-tier flow oracle is used for correctness, not as a faster or
slower timing substitute. The last row is an isolated traversal diagnostic,
not a replay of a complete live migration proposal. Full assignments match
the retained general solver and independent exhaustive feasible assignments.

## Model result and next decision

| Unprofiled build | Prefill tok/s | After-prefill decode tok/s | Measured promotions / waves |
|---|---:|---:|---:|
| Accepted maintenance-affinity control |314.868|40.035|16 /4|
| Initial queue candidate only |321.402|39.884|16 /4|
| Final queue plus exact planner/ranges |314.818|39.988|24 /6|

All768 generated IDs, complete MTP records and initial expert plans match in
both comparisons. Only the benchmark-output and prefix-directory paths differ
in resolved configuration. These are single matched runs, not a statistical
claim that the final build gains or loses throughput. Extra movement is actual
completed work, not a sum of isolated projected candidate benefits. Final peak
per-rank RSS is36.008/77.066GiB, comparable to36.012/77.108GiB in the control;
native/driver/process retirement passes.

Cheaper planning completes more fresh waves, but each fresh256-logical-row
observation still admits only one bounded four-promotion wave. That threshold
counts completed routed rows at the last main layer, not expert activations;
an actual448-row transaction is one complete observation. The next candidate
is reuse of immutable observations for further profitable waves after original
publication/reclamation, bounded by new inference progress and with fresh READY
evidence taking priority. It is **not implemented or qualified by this slice**.
MPI, transfer slots, source arithmetic and policy thresholds remain unchanged.

## Focused qualification and evidence

Canonical Release rebuild:288.466s, unrestricted parallelism, configured Ninja,
CUDA80/86/89/90 and ROCm gfx906/gfx1100. All configured GPU spill checks pass.
Focused registrations:10/10 host in11.692s;5/5 CUDA in14.638s;5/5 MI50 in6.559s;
5/5 gfx1100 in5.275s. This is not a complete Unit/preflight or OpenCode certificate;
the user deferred those broad gates for this iteration.

New explicit preflight registrations are
`PooledImmutableProtectedHitQueueBatchReplay` (host/CUDA/ROCm),
`MoEExactTwoTierAssignmentReplay`, `MoECanonicalOwnerLayerRange` and
`MoECanonicalCohortLayerRange`, under their `V2_Integration_` names. The malformed
tail cases belong to the existing protected-queue wildcard registration.
The original owner suites and prior source/frequency/cancellation/failed-epoch
registrations run alongside them. Range checks are named as range checks,
not controller replay.

Local records, excluded from source control:

- `/tmp/qfn-ngram-protected-queue-batch-next/`: first model comparison, final
  profile and PID-bound graph/phase analysis, component timings and review fix.
- `/tmp/qfn-moe-exact-planner-next/`: original/final public-owner timings,
  component comparison, final model/config/output comparison and next proposal.
- `/tmp/qfn-cpu-eight-row-rocm101-root/`: `queue-final-*` build/CTest records.

Each completed model/profile lifetime reclaimed its own approximately728MB of
prefix blobs using about29KB of metadata and zero payload reads. No model or
cache payload hashing was performed; trace databases remain available for
focused attribution.
