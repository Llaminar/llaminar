# QFN MTP on ROCm TP4

Updated 2026-10-28. Limit: 15,000 UTF-8 bytes. Checkpoint: fresh finite-trajectory
QFN MTP comparison and full OpenCode session complete on the same four MI50s
with the accepted configuration; EOS and scope limits remain explicit below. This plan and status table remain below 15 KB;
raw runs, profiles and failed experiments stay in ignored result directories.

## Baseline contract

- Llaminar qualification is checkpointed at `3d46cf4a9` (base `e2c971301`);
  Castagna is pinned to
  `3483d462715d61f21ed5877836be35a724446359`, matching its checked remote main.
- Both engines use the original four-shard Qwen3.8-Flash-Next UD-Q4_K_XL and
  `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, with metadata-only source identity.
- Four MI50s, homogeneous TP4, all routed experts on GPU, RAM ngrams, FP16 KV,
  28 physical NUMA1 workers, context 4096 and no prefix reuse for timing.
- Freeze the existing raw 512-token prompt, 256 emitted tokens, greedy seed 42,
  one warmup and three measured requests. Both timing denominators must count
  255 after-prefill tokens. Keep all outputs, sample ranges and failed runs.
- Llaminar A1 uses TensorParallel, GateUpOwnedDownColumns, sharded vocabulary
  and the current production arithmetic. Castagna's within-expert split differs;
  this is an engine comparison, not a claim of identical physical kernels.
- Measure Off and fixed depths 1/2/3 before adaptive policies. Record draft,
  accepted and verifier work. The two adaptive controllers are different;
  compare them as explicit engine policies with a common maximum, not as the
  same algorithm. Draft vocabulary is initially unrestricted on both sides.

Qualification and acceptance of fixed/dynamic MTP, stochastic sampling and
prefix request reuse precede further performance tuning. The table records
diagnostic timings; it does not certify the remaining production cases.

## Current status

**Accepted Dynamic: 101.367 decode tok/s; fresh Castagna fixed6: 100.098.**
Fresh Off is 43.512. On the exact 512/256 repetition trajectory, Dynamic leads
the fresh fork median by **1.267%** and its own Off by **2.330×**. Decode ranges
are 101.272–101.398, 98.183–100.989 and 43.471–43.572 respectively. Prefill
medians are 788.986, 435.514 and 788.898 tok/s. All nine measured token streams
match; native warmup count is retained, but its output IDs are not.

**EOS caveat:** native benchmarking ignores stop-token termination; fork raw
responses permit EOS and end at the token limit. This is a matched observed
finite trajectory/budget with a common 255-token decode denominator, not
identical EOS semantics or natural-EOS parity. Frozen packet labels overclaimed
that contract. No general-policy/noise-superiority claim follows; the dated
98.164 Castagna series stays dated. Fresh evidence:
`/tmp/qfn-fresh-head-to-head-fullspan-r1-private/report.json`.

**Chosen default (user decision 2026-10-28):** retain the controller for this
measured QFN ROCm TP4 scope. The 95%-of-best-fixed comparison remains 6/10 and
diagnostic; every miss is retained. Correctness, HTTP and full-app gates pass.

Current full-span source: complete gates **2,981 pass / 29 hardware skips / zero
failures**, including 786 Unit tests; coherent Release green. Canonical reuse
remains rejected after one unrelated exporter link. Reviewed local-only delta
admission preserves the original 3,010-test receipt/time and authenticates its
unchanged tested dependency/runtime/source/inventory closure. No CI/release
certificate or refreshed gate is claimed. Fresh Off and Serving-Dynamic HTTP
384-token fresh/full/partial/full lifetimes pass, including exact serial IDs,
retained ticket/branch evidence, prefix/resources/strict driver and retirement.

Fresh ten-phase OpenCode **passes**: 110 tool attempts, zero errors, 85 requests,
one client context compaction, 1,950.81 seconds; independent webapp acceptance
and 45 application tests pass. Native audit joins all 85 responses/110 calls
unchanged. Runtime/resources/archive/strict driver and normal unforced retirement
pass; owner cleanup reads zero payload bytes. Local after-admission passes.
Normal serving EOS semantics apply. The host fix is qualified for this workload,
not universal page immovability or general performance superiority. Earlier R14
aggregate red remains preserved; its old app counts are not current metrics.
[Host fix and evidence](QFN_HOST_REGISTRATION_INVALIDATION_20261028.md).

Historical HTTP and rejected controller economics remain in the
[retained history](QFN_MTP_DIRECT_OUTPUT_HISTORY_20261028.md),
[utility investigation](QFN_MTP_UTILITY_PROBE_PLAN_20261027.md) and
[confirmation study](QFN_MTP_TARGET_CONFIRMATION_20261028.md).
The preserved 6/10 speed result is unchanged; no prior focused/replay pass
substitutes for current full-app evidence.

| TRAIN prompt | Greedy dynamic/fixed tok/s | Difference | Stochastic dynamic/fixed tok/s | Difference |
|---|---:|---:|---:|---:|
| Python | 79.78 / 79.74 | +0.04% | 81.15 / 80.80 | +0.43% |
| Repetition | 101.19 / 108.13 | −6.41% | 100.48 / 107.79 | −6.78% |
| Builtin | 76.80 / 89.26 | −13.96% | 79.23 / 79.28 | −0.07% |
| Long Python | 66.28 / 70.27 | −5.68% | 60.38 / 62.49 | −3.38% |
| Prose | 62.28 / 64.76 | −3.83% | 75.78 / 74.77 | +1.35% |

Medians of three: refreshed greedy, original stochastic. Report the95% comparison
against each fresh frozen-best-fixed control as a diagnostic.
No normalization or dropped rows. Evidence:
`/tmp/qfn-mtp-confirmation-comparison-r1/comparison.json` and
`/tmp/qfn-mtp-confirmation-focused-r3/result.json`.
Fewer probes (72→15) did not establish a general speed win. Builtin spends32
transactions at depth2 versus8 previously. Repetition's locally supported depth6
changes future transaction boundaries and finishes slower than fixed5; its cost
profile agrees with measurement. Long-context repricing preserves all ten fixed
winners. Neither richer histograms nor a cost-table correction fixes these choices.
Controller checkpoint: `078f14c88`; the [host-page fix](QFN_HOST_REGISTRATION_INVALIDATION_20261028.md) is now qualified locally.
Diagnosis: `/tmp/qfn-mtp-confirmation-repetition-diagnosis-r1-private/NOTE.md`;
baseline plan: `/tmp/qfn-mtp-castagna-train-coverage-review-r1-private/NOTE.md`.

Scoped control preserves original counter ordinals. Guarded CUDA80/86/89/90 and
gfx906 builds pass after eliminating an outlined-helper callee-save spill;
metadata serialization and startup authority have regressions.

## Current performance

Rates are prefill/decode **tok/s**, medians of three requests after one warmup.
The table below retains historical R9/R11 rates; the fresh natural-EOS control
is above. Llaminar retains capacity15. Castagna's native baseline uses capacity
equal to depth; capacity15 controls are also shown because its larger reservation
measurably costs throughput. The newest exact-row results are R9/R11.

| Policy | Castagna native capacity | Castagna capacity15 | Llaminar capacity15 | Scope |
|---|---:|---:|---:|---|
| Off |466.22 /42.67|—|819.75 /43.43|Serial controls; L R6 |
| Fixed1 |437.88 /64.55|—|789.27 /59.11|Earlier lease-fix control |
| Fixed2 |435.81 /78.45|—|789.10 /73.59|Qualified checkpoint control |
| Fixed3 |434.57 /91.04|—|789.22 /90.11|L R9 exact rows4 |
| Fixed4 |432.68 /92.22|409.21 /85.73|789.05 /96.82|R10/R11; L exact rows5 |
| Fixed5 |431.60 /93.76|414.94 /88.08|788.99 /**108.43**|R10/R11; L exact rows6 |
| Fixed6 |424.11 /**100.21**|408.86 /92.82|788.54 /86.01|R10/R11; L exact rows7 |
| Default dynamic, cap15 |—|410.76 /65.49|789.16 /89.16|Different controllers; L R9 holds3 |

L fixed4/5/6 request work respectively:208/215/300 drafts,
204/213/206 accepted,0/1/42 rejections,52/43/50 verifier calls and260/258/350
logical rows. Post-rejection and terminal-budget unused drafts are not accepted;
training uses accepted/drafted, matching controller economics. Fixed5 samples
108.19–108.65; fixed6 samples85.92–86.20. Each stream equals own-Off IDs.
Historical rate differences are15.64% at depth5 and8.20% versus fork-best6;
those historical fork requests suppressed EOS; use the fresh matched control above.

Completed arms retain native, quiet-window, driver and normal-retirement passes.
L uses `OMP_PROC_BIND=close,OMP_PLACES=cores`; Castagna uses explicit `false`.
Results under `parity-results/qwen-flash-next/`:
`mtp-tp4-llaminar-{default-dynamic,fixed3}-exact-r9-r1`,
`mtp-tp4-llaminar-fixed{4,5,6}-exact-r11-r1`,
`mtp-tp4-castagna-fixed{4,5,6}-cap15-r10-*`, and
`mtp-tp4-castagna-fixed{4,5,6}-cap{4,5,6}-r11-r1`, with quiet siblings.
Fork token proof: `/tmp/qfn-fixed456-castagna-baselines-r1.json`.

Hosted execution retains exact rows2–16, ordinary M1 separately, with one
admission inventory for prep/outcome/publication and pipeline followers. Native
CUDA conditional parents retain their envelope. Continuation uses the genuine
device-authored admission ticket and preserves learned depth, without host state
mirrors or serial row replay. Exact-row tags are absent from timing's disabled/
reset PerfStats; HTTP must prove actual retained replays separately.

R9 versus R6 Dynamic: four-GPU sampled VRAM rises0.372GiB; summed process RSS
rises4.04GiB, not deduplicated physical RAM. Maximum overlapping worker family
setup rises44.683→105.945s, not total launch readiness. Later idle VRAM returns
to the approximately40MiB baseline. The ordered-view candidate was removed:
90.271 D overlaps90.164/90.369 control/repeat. The fixed3 CPU profile attributes
94.64% sampled cycles to ticket waits and0.97% to identity checks; these are not
wall-time fractions. [Details](QFN_MTP_TP4_HOST_PROFILE_20261027.md).

## Dynamic-depth retraining

Use the existing generated-table trainer and production BenchmarkRunner. Reuse
one admitted runtime across validated, reset requests through the typed
`configureMTPRequestPolicy` API. Keep canonical timing and output evidence.

1. Freeze short/repetitive, Python generation, prose and long-context prompts
   before collection; separate train and holdout prompts. Measure fixed1–15,
   Off and old dynamic with greedy and seeded stochastic sampling, common
   retained capacity and request budgets.
2. Authenticate prompt IDs, real model class, topology, execution family,
   sampling, source/runtime and work counts. Scope policy updates to measured
   domains; preserve unmeasured models/backends/topologies.
3. Derive labels from useful after-prefill throughput and accepted/attempted
   drafts. Select startup from equal-weight training requests only. Holdouts
   cannot select startup, thresholds or the near-best band.
4. Test emitted first-match behavior and unchanged domains. Measure held-out
   dynamic against old dynamic and fixed controls, including rejection, low
   acceptance, longer context and retained graph reuse. Report throughput regret
   as well as rule classification; aim within5% of best fixed without baseline
   regressions. The repetitive512-token control alone cannot qualify defaults.
5. Promote the measured winner, run fresh complete prerequisites, then refresh
   stochastic/prefix HTTP and OpenCode. Fixed5/6 exports preserve existing
   six-mode qualification; the automatic serving startup is a separate check.

The refreshed main-model panel is
`parity-results/qwen-flash-next/mtp-tp4-baseline-accuracy-r1/report.json`.
Required short and held-out groups pass with reset byte equality and clean
retirement. Mean KL / top-1 / top-5 overlap are respectively:

| Group | Prefill | Decode |
|---|---|---|
| Short | 0.02561 / 93.75% / 92.50% | 0.02828 / 95.83% / 92.29% |
| Held-out short | 0.01491 / 100% / 95.00% | 0.00455 / 100% / 91.67% |

These are the canonical within-group prompt means, not pooled phases. This
local diagnostic does not grant fresh full-preflight/image certification;
the QSA-threshold groups remain diagnostic rather than accepted. Active MTP
continuation and actual runtime work require their separate checks below.

## Qualification and tuning

The GDN checkpoint passed complete native prerequisites (2841 pass, 29 hardware
skips), then 782 Unit and five packaging checks after the packaging-only update.
Its four GPU generation topologies used Qwen3.6, not QFN. QFN's new 24-request TP4 qualification is recorded above; the earlier four-token
lifecycles and Qwen3.6 timing are separate evidence. Do not transfer either
model's speed claims or infer a completed coding session.

The existing A1 mathematical declaration is Off-only and its reference has no
sidecar gold. The typed TP4 execution/admission declarations do not invent reference
capacity or grant certification by registration. Keep main-model
frozen-history KL/top-1/top-5 panels, serial/grouped byte equivalence, actual
dynamic work, accepted-state continuation and full/partial prefix tests as
separate necessary evidence. Retain layer/stage CSV diagnosis on a failure.

Use supported local diagnostic entrypoints during tuning. Preserve failed and
superseded receipts; do not alter timestamps or present scoped checks as a fresh
full receipt. Refresh the complete gate for the accepted implementation, not
for every unchanged timing experiment.

First establish unprofiled throughput. Then attribute the measured deficit with
actual-depth HIP traces and isolated captured kernel timings. Distinguish
prepared graphs from executed work; never sum overlapping spans as latency.
Optimize the largest exposed deficit, preserve exact grouped arithmetic and
fixed live communication extents, add focused regressions, and profile each new
kernel. Recheck the full model after a useful component change. Require a repeat
or interleaved comparison whose gain exceeds observed run-to-run drift.

## Full coding-session gate

Run the existing `opencode_tool_stress.py` ten-phase `webapp` session, one
conversation, after smoke, accuracy and reasonable-speed checks. Use default
dynamic MTP, PMA-admitted maximum model context, 32768 output tokens, 16 GiB RAM
prefix cache per participant and 32 GiB shared disk cache. Keep whole-turn and
whole-session generation deadlines absent; monitor progress and token output.

Require every app phase to finish, native tool/protocol/runtime evidence,
strictly less than 5% tool errors, bounded RAM/VRAM/disk growth against actual
admission/cache allowances, fresh stats, clean driver observations and normal
retirement. Report model-authored app quality separately. Remove only owned
scratch/cache files after retirement and retain compact evidence. A local full
session does not claim immutable-image certification.

Green measured slices are checkpointed on the feature branch with their exact
settings. Goal completion requires both the matched TP4 MTP win and the complete
coding session; admission or a short smoke alone is insufficient.
