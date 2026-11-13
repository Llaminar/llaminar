# Exact scoring of isolated tier exchanges

The matched model passes at **332.183 prefill / 40.372 after-prefill decode
tok/s**, versus **332.087 / 39.851** for checkpoint606fd1e26. Prefill is flat;
decode is1.31% higher in this single comparison. All768 output IDs, complete
MTP records, initial placement and configuration apart from output paths match.
The500/40 hybrid goal remains open; this result reaches only its decode target.

## Scoped implementation

The pre-cut tier inventory previously called the complete-map service scorer
for every candidate cycle. An isolated two-edge exchange can use the existing
exact `ObservedTransactionServiceObjective::SwapSearch` instead.

The optimized branch requires two distinct experts in the same layer with
exactly reciprocal canonical execution targets. A proposal-local source map
and prepared search are reused within the current layer. Complete-map source
validation still runs before preparation, including unpriced phases, and
each edge checks its actual published source. CPU TP cohorts normalize to one
certified logical target through the existing target index; an absent scalar
owner is never treated as an execution target.

The existing integer arithmetic, phase multiplicity, overflow checks, sorting
and objective remain unchanged. General/longer cycles and all final composed
admission still use the complete scorer. There is no persistent cache, new
controller, transport, ledger, threshold or numerical-kernel change. Production
changes are confined to the existing authority implementation.

## Setup-inclusive public-operation timing

The existing `Perf__PreparedExpertSwapSearch` target gains an opt-in authored
public-proposal workload. It has512 experts per layer, three logical whole
targets, top10 routes, phase batches1/448/64/3, five cycle slots and four shadow
slots. It exercises84 tier exchanges per layer. This is a host-only controller
workload, not captured model routing or a CPU-cohort hardware timing claim.

The test owner was built first against the unchanged production body, then
against the optimized body. Both use the same CPU28 affinity and original
SDK/CPU-only environment. PMA/fixture setup and assertions are outside the
clock; the complete public proposal, including validation and preparation,
is inside it. Seven chronological post-warmup samples are retained per case.

| Authored layers | Original median | Optimized median | Speedup |
|---|---:|---:|---:|
|1|31.815ms|30.259ms|1.051×|
|49|357.240ms|319.345ms|1.119×|

Every actual admitted layer/expert/source/destination edge matches in order,
as do source generation, epoch and coherent service/net scores. Sample CV is
1.08%/1.08% for the two original cases and2.99%/0.83% for the candidates.
These are setup-inclusive component gains, not model speedup factors. The
comparison is one original/candidate pair; no reversed-order or statistical
significance claim is made.

## Focused regressions

The complete transaction-cost and authority units and both new aliases pass
against the unchanged scorer before applying the production optimization.
With the optimized scorer, eleven focused registrations pass: those full
owners, the full maintenance owner, controller/cadence/local retirement,
original MPI publication and continuation/retirement, plus the new explicit
production-preflight entries:

- `V2_Integration_MoEIsolatedTierSwapCostEquality`
- `V2_Integration_MoEIsolatedTierSwapJointAuthorityReplay`

Numeric coverage includes512 experts, three targets, M1/3/64/448, zero-demand
experts, poisoned padding, transaction multiplicity and existing invalid,
unpriced and overflow checks. The original source-bound joint authority
fixture covers real prepared2/4/8-member CPU cohorts with CUDA/ROCm GPU
targets. It uses the original planner and existing transaction test access
to assert the pre-cut reciprocal two-edge case, then executes normal public
proposal, final three-edge coherent admission, peer adoption, commit and
reset. It does not add a production test API or infer branch coverage solely
from the final transaction.

The original-owner benchmark needed two fixture corrections before timing:
use the existing `recordObservedExpertBatch` helper, and clear the explicit
rank list when declaring its GPU domain rank-local with owner0. Production
validation was not weakened. Final production rebuild takes28.493s; the
eight host registrations take5.286s and three MPI registrations4.194s.

## Real-model economy and lifecycle

The [M448 checkpoint's workload](QFN_ROCM_M448_NATIVE_Q8_20261110.md) remains
fixed: two local MI50s, two actual CPU socket MPI ranks,28 physical workers
per socket, Qwen3.8-Flash-Next UD-Q4_K_XL with Q8_0 MTP sidecar,262144 context,
512/256 tokens, seed42, one warmup and three measurements. Dynamic MTP and
expert balancing, RAM ngrams, GPU row caches and16/32GiB RAM/disk prefix
budgets remain enabled. The original application owns automatic MPI bootstrap.

The run reports60 measured promotions in15 completed transactions and
683,724,800 physical movement bytes, versus56/14 and471,552,000 bytes in
the immediate control. Epoch advancement can include publication still
retiring at a request boundary, so completed transactions—not epoch deltas—own
the movement total. Extra movement is not itself a performance benefit.

Peak per-process RSS is36.011/77.162GiB; selected per-client peak VRAM is
30.240/30.076GiB. The observer reports no errors; these are bounded short-run
observations, not a long-duration leak certificate. Native, driver and process
retirement pass in249.869s. The owned prefix files reclaim727,912,511 bytes
using28,687 metadata bytes and zero payload reads after retirement.

## Separate CPU profile and next target

After the model and unprofiled timings retired, `perf` sampled the current
public-proposal component on CPU28 at an8-million user-cycle period with
32KiB DWARF stacks. It captured1,856 samples, zero lost samples, and58.627MB.
This profile includes fixture/test setup as well as the selected proposal
workload; its percentages are not pure timed-region shares or inference
wall-clock savings.

Leading self-cycle shares are migration-cycle DFS16.38%, preferred owner-map
construction9.75%, owner sorting4.20% plus3.77%, and owner lookup3.88%.
The complete-map scoring functions account for1.83% and1.89% exclusive shares;
those are not inclusive call-tree totals. A source review confirms a later
inventory still uses the full scorer for some exchanges, but cycle traversal
is the stronger next measured target. Investigate preserving original edge
order while avoiding global repeated scans; do not broaden this into an
owner-map or graph-builder rewrite without new evidence.

Local evidence is under `/tmp/qfn-moe-isolated-tier-swap-next/`:
`component-comparison.json`, `model-comparison.json`, original/candidate logs,
`profile/`, model resource/retirement records, independent review and
`remaining-work/NOTE.md`. Build/CTest receipts are under
`/tmp/qfn-cpu-eight-row-rocm101-root/isolated-tier-*`. Result payloads remain
outside source control. Full current preflight, broader topology model panels
and OpenCode remain deferred; this slice does not claim those certificates.
