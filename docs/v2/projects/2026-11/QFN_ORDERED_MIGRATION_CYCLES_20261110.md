# Ordered adjacency for expert migration cycles

This slice preserves migration decisions while reducing repeated traversal of
large candidate populations. It changes no routing precision, placement
objective, controller cadence, physical transfer capacity or capture policy.

## Implementation and equivalence

The existing decomposition function builds invocation-local vertices from the
layer and complete typed placement target. Ordered scalar buckets narrow
comparisons; full target equality remains authoritative, including a cohort's
source, members and cuts. Outgoing edges retain original input-index order.
DFS therefore sees the same eligible edges, seed order and backtracking as the
former global scan. Active-path marks expire at each seed; no persistent cache
or ownership ledger is added. Recursion and worst-case branching are unchanged.

The cycle validator no longer clears a full-migration bitmap for every cycle.
Its existing logical `(layer, expert)` set already rejects every repeated edge
index, as well as duplicate expert identities. Bounds and diagnostics remain.

The new authored differential suite retains the original resolver, global-scan
DFS and bitmap validator as test oracles. It compares exact cycles, original
input fields and rejection messages across shuffled/unsorted layers, whole
target counts 1–8/17/65, cohort widths 2–8 with colliding bucket keys but different
cuts, self-loops, skewed rings, backtracking and large/empty/small invocations.
These are metadata equivalence tests, not physical 65-device qualification or
recorded inference traces. Their largest generated population has 447 edges.

The existing target and its full owner remain canonical. The new explicit
production-preflight entry is
`V2_Integration_MoEOrderedCycleAdjacencyEquivalence`.

## Focused qualification and economics

The three placement registrations first pass against unchanged production in
1.588 s. The optimized Release rebuild takes 29.032 s. Eleven host registrations
pass in 6.618 s, covering full authority, cost, maintenance and placement owners
plus continuation, cadence and retirement. Three original MPI publication and
retirement registrations pass in 5.298 s. Independent source review finds no
correctness blocker. Full production preflight remains deferred for this slice.

The existing opt-in public-proposal component runs on physical CPU 28, with
identical environment, observations, policy and seven post-warmup samples.
Complete proposal construction is timed; fixture setup is outside the clock.
Actual admitted edge order and coherent objective/epoch scores match exactly.

| Authored layers | Global-scan median | Adjacency median | Speedup |
|---|---:|---:|---:|
| 1 | 30.904 ms | 29.996 ms | 1.030× |
| 49 | 328.153 ms | 266.856 ms | 1.230× |

The candidate CV is 3.07% and 1.04%, versus 1.05% and 1.04% for the control.
This is one control/candidate pair, not an ABBA or statistical significance
claim. The source profile motivating the change included test setup; neither
its sample share nor these component speedups are inference speedup estimates.

## Model result

The unchanged original 512/256-token owner runs QFN UD-Q4_K_XL with its
Q8_0 MTP sidecar, two local MI50s and two physical CPU socket ranks with
28 workers each. Context remains 262144, dynamic MTP and balancing remain on,
RAM ngrams/local GPU caches and 16/32 GiB prefix budgets remain unchanged.
One warmup precedes three measured requests in each fresh lifetime.

| Implementation / lifetime | Prefill tok/s | After-prefill decode tok/s |
|---|---:|---:|
| Previous checkpoint db80eb235 | 332.183 | 40.372 |
| Ordered adjacency, first | 328.794 | 39.609 |
| Ordered adjacency, repeat | 331.278 | 39.794 |

Both candidates match all 768 generated IDs, complete MTP records and initial
expert placement; only output paths differ in resolved runtime configuration.
These observations do **not** establish an inference speedup. Relative to the
immediate control, prefill changes by -1.02%/-0.27% and decode by -1.89%/-1.43%.
The repeat is close to the earlier continuation control's 327.546/39.893,
but that is not a reversed control proving a noise band. This checkpoint
records a setup-inclusive planner improvement, not attainment of the 500/40
model target or a new throughput default.

The first/repeat measured cohorts complete 12/14 migration transactions,
48/56 promotions and 545,177,600/564,326,400 movement bytes. Their completion
timing differs from the control's 15 transactions/60 promotions; higher
movement counts are not assumed to be faster. Peak individual process RSS
is 36.013/77.165 GiB across the two runs; selected per-client VRAM stays at
or below 30.309/30.077 GiB. Both observers report zero errors. These short
lifetimes do not certify absence of long-duration leaks.

Original native, driver and process retirement pass in 252.723/251.537 s.
After retirement, 1,455,825,001 bytes of owned prefix files are reclaimed with
57,353 metadata bytes read and zero payload reads. No cache blobs are hashed.
The next experiment is the existing wider migration-wave policy with genuine
new capacity admission, followed by a separately qualified CPU integer-ILP
candidate. Neither is included in these measurements.

Local evidence: `/tmp/qfn-moe-ordered-cycle-adjacency-next/`, including patches,
independent review, comparison and model records. Original component logs remain
under `/tmp/qfn-moe-isolated-tier-swap-next/ordered-adjacency-{control,candidate}`;
build and CTest receipts are under
`/tmp/qfn-cpu-eight-row-rocm101-root/ordered-adjacency-*`. Result payloads are
not committed. No extra full-model accuracy or OpenCode certification is claimed.
