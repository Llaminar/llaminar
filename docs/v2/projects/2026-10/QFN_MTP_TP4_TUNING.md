# QFN MTP on ROCm TP4

Updated 2026-10-28. Limit: 15,000 UTF-8 bytes. Active objective: exceed Castagna's measured QFN MTP decode
throughput on the same four MI50s, then complete a full OpenCode coding session
with the accepted configuration. This plan and status table remain below 15 KB;
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

**Accepted dynamic default: 101.05 decode tok/s versus fresh Off 43.40.**
On the original 512/256 repetition workload, all token IDs match across one
warmup and three measured requests per arm. Dynamic is 2.94% above the dated
Castagna 98.16 control; no new Castagna run or general-policy win is claimed.
[Current evidence](QFN_MTP_DIRECT_OUTPUT_HISTORY_20261028.md).

**Chosen default (user decision 2026-10-28):** retain the new controller for the
measured QFN ROCm TP4 scope despite imperfect economics. The 95%-of-best-fixed
comparison passes only6/10 and is now diagnostic, not a promotion blocker.
Every measured miss remains visible. Correctness, HTTP and full-session gates
still apply; no unmeasured cost domain is added.

R13 Unit/preflight: 2,978 pass, zero fail, 29 skips. Post-audit Unit: 786/786.
Release replay/resources pass. Native-event medians CUDA 1.136µs and ROCm
1.9475µs are amortized operations, not speedup claims. All nine HTTP arms pass:
Off is the serial control; eight arms prove widths/comparison. Serving starts
at depth 5 and uses depths 1/2/3/5. Ten app phases pass: 128 tool calls, two
errors, 65 tests and independent acceptance. Archive/native-token audit repairs
pass. The strict driver gate remains red; complete certification is open.
[Diagnosis and receipts](QFN_MTP_DIRECT_OUTPUT_HISTORY_20261028.md).

Historical HTTP:24 requests/9,216 tokens, exact serial equality at
T0.7/k40/p0.9/seed4242, full/partial prefix restoration:
`/tmp/qfn-a1-http-qualification-setup-r5/acceptance.json`.
This does not certify current source.

Scoped training, exact transaction capture and canonical controller replay are
implemented. Global fitting, longer sample cadence and capped upward steps have
not passed the complete economic gate. See the
[training investigation](QFN_MTP_DEPTH_RETRAINING_20261027.md) for retained failures.

Earlier reversible-utility and provenance trials failed economics. Their full
evidence remains in the [utility investigation](QFN_MTP_UTILITY_PROBE_PLAN_20261027.md).

The previous downward-only candidate passed TRAIN10/10 but failed frozen HOLDOUT3/4;
it remains rejected. Its exact results are retained in
`/tmp/qfn-mtp-directional-downward-final-economic-report-r1/final-evaluation.json`.

The previous verified-progress candidate passed correctness but only6/10 speed
rows; its evidence remains in the [investigation](QFN_MTP_VERIFIED_PROGRESS_20261028.md).
The [confirmation successor](QFN_MTP_TARGET_CONFIRMATION_20261028.md) requires the
same target across three disjoint windows using admitted hysteresis. Complete786
Unit and75 focused registrations pass without skips, including native CUDA/ROCm,
terminal validation and genuine recorded preflight replay. Release,10 captures
and40 strict replays pass. All90 timed outputs match serial controls. However,
speed reaches95% in only6/10 rows. The user accepted this trade-off for the
default. Current-source correctness preflight and HTTP/app qualification remain
required; the failed economic results are retained as diagnostics.

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
Next: investigate the strict driver failure; checkpoint the scoped green work.
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
