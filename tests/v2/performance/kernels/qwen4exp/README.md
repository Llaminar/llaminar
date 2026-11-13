# Pooled ngram replay inputs

`Measure__PooledNgramLookup.cpp` times the original native fixture using the
explicit disjoint-bank trajectories in `PooledNgramReplayTrace.h`. Those inputs
are diagnostic controls. They are not real QFN conversation traces.

`qfn_ngram_trace.py export` imports existing full-stage reference
`PLE_HASH_INDICES.npy` snapshots with their exact prompt/teacher history and
source tensor descriptor. Supply the existing frozen panel manifest, original
reference directory, source checkout, PLE layer and the authored full-query
bound. The canonical `GGUFParser` reads split headers only. No weights or
reference results are regenerated; output must be fresh and remain outside the
source tree. `validate TRACE.json` rechecks hashes and every original integer
row against those saved snapshots. NumPy is required for these explicit
external-input operations; the fast Unit contract uses only Python's standard
library and tiny authored metadata.

Exported reference frames describe current, consumed **reference inputs**.
They do not prove native execution, future read-ahead, cancelled MTP work,
cache transitions or timing. Never relabel teacher tokens as rejected drafts.
Changed-conversation and scan-pollution replay require corresponding original
hash captures. Explicit adversarial metadata controls are separately labeled;
they validate the acceptance contract without inventing observed MTP events.
Trace payloads belong in ignored local artifacts or the corpus repository, not
this directory.

The same module checks terminal consistency using the existing LocalHot
`Counters` fields, actual frame promotions/evictions and native live payload
extents. It rejects speculative consumption/promotion credit, capacity-sized
traffic, duplicate-row inflation and response storage beyond the single
admitted frame. These fields must come from the real completed native path;
passing arithmetic cannot authenticate a source, event or execution. Shared-only
and ConsumerLocal counters are not renamed as L1 counters. The trace comparison
configuration validator checks identical authored PMA
ceilings and complete declared BOMs as a fixture input constraint. That check
does not establish equal admitted cache costs or qualify native performance.

The current timing actor has no admitted real-trace combined L1 path. Actual
production-Operations replay and native performance remain pending integration;
this tooling emits no measured timings or performance qualification.

## Genuine model-operation diagnostic route

Use the existing `llaminar_model_kld_canary --workload WORKLOAD --output
OUTPUT -- ENGINE_ARGS` route with the existing
`llaminar.model-kld-workload.v2` descriptor, with the same canonical corpus, topology/config document and
retirement observers used by the numerical panel. `ModelKLDCanary` owns the
original RuntimeInit model, PMA, workers and retained storage participants. It
executes the original prompt and Pending teachers; it does not inject lookup
responses or replace the real GGUF table with the patterned component file.
The numerical and reset gates remain mandatory.

Natural-history campaign projection requires a positive 32-bit sampling seed.
If the selected source leaves its seed unspecified (`-1`), supply `campaign
--seed N` and pass the same `--seed N` to the public benchmark invocation. The
projection preserves source metadata and the sampling law; it never substitutes
an implicit seed or changes the numerical panel. Use one explicitly fixed seed
for every policy in a matched comparison.

Each completed history row additionally reports `diagnostic_phase_timings` in
its existing observation object. These are steady-clock nanoseconds from the
original prefill/teacher-decode call through its existing joined host-logit
publication. They include the diagnostic full-vocabulary D2H, and exclude
reset, logit validation and output-file I/O. The first Pending teacher merely
stages a token and receives no completed-decode timing. Row IDs and input-token
denominators are the existing scored-row identities. No additional D2H, event,
host controller or per-token native probe is introduced.

This is an executable full-model **diagnostic wall interval**, not a lookup
stall measurement or Release throughput qualification. The report's existing
`actual_owner`, native/config identity, production path records and strict final
retirement proof must accompany it. ConsumerLocal, shared-only and combined
comparisons preserve the original
model/expert/KV obligations, source backing and admission limits. Match the
aggregate admitted physical **cache hierarchy** cost, including its mandatory
frames, metadata, staging, transport and native ordering resources as well as
optional cache roots. Report authoritative source backing separately as a
common cost. Attribute shared owner categories through their original typed
ngram BOMs; do not count whole model activation or transport categories as
cache storage. Report per-allocator used bytes and alignment residuals even
when aggregate cache costs match; identical whole-model layouts or per-device
usage are not required. Every complete plan must still pass ordinary PMA
admission. C0 and positive slots remain distinct authored policies. Equal
optional ceilings, slot counts or GPU-budget sums do not prove matched costs.

Use source-admission `admitted-resources.csv` and `admitted-owners.csv` from
the original final PMA plan and retained attestation to check actual costs.
Resident and new bytes partition the admitted total; preparation and borrowed
subsets must not be added again. The benchmark-only
`PooledNgramReplayBudget.h` resolver is fixture-qualified until supplied the
original model planning inputs. Its example caps are not live model budget
qualifications. For live comparisons, author explicit candidate ceilings from
actual source observations, then rerun canonical admission and report the
remaining cache-cost residual before timing.

Use the saved trace provenance only to identify which exact histories/layers
have independently retained input-row observations. The available two short
histories provide fourteen frames, not long-scan, speculative-MTP or broad
warm-conversation coverage. Terminal local/shared/RAM/SSD counter and byte
deltas must come from the actual completed production producer; those lookup
metrics remain separate from these host phase intervals.
