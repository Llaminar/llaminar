# QFN MTP on ROCm TP4

Updated 2026-10-27. Active objective: exceed Castagna's measured QFN MTP decode
throughput on the same four MI50s, then complete a full OpenCode coding session
with the accepted configuration. This plan and status table remain below 15 KB;
raw runs, profiles and failed experiments stay in ignored result directories.

## Baseline contract

- Llaminar starts at `e2c971301`; Castagna is pinned to
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

## Current qualification

**Accepted for checkpointing:** fresh complete prerequisites and all six real
QFN TP4 HTTP modes pass. Performance tuning against Castagna is next; no MTP
throughput win or full OpenCode completion is claimed yet.

- Complete prerequisite R3: **783 Unit + 2,065 integration tests pass**, with
  29 unavailable-hardware skips and no failures. Every terminal XML and the
  unchanged build/inventory identity were verified. Receipt:
  `/tmp/qfn-mtp-qualified-prerequisites-r3/prerequisites.json`.
- Release/Integration rebuilds, ten focused registrations and six actual source
  runtime exports pass. The accepted implementation retains native allocation
  borrows across replay, declares independent FA/QSA participant obligations,
  publishes completed native capture evidence and fixes main/predictor bounds.
  Each defect has an explicit production-preflight regression.
- Runtime exports separate admission diagnostics, HTTP qualification and
  production serving. Runtime uses production prefix/maintenance defaults;
  diagnostic short-prompt block geometry cannot leak into serving. Qualification
  retains capacity 15 and starts dynamic depth at 15; serving delegates initial
  depth to the native policy. Cold startup remains within its source-owned 300s
  readiness allowance.

| HTTP policy | Committed tokens | Recorded rejections | Result |
|---|---:|---:|---|
| Off | 1,536 | 0 | Green control and exact repeats |
| Fixed1 | 1,536 | 306 | Green; exact serial agreement |
| Fixed2 | 1,536 | 472 | Green; exact serial agreement |
| Fixed3 | 1,536 | 568 | Green; exact serial agreement |
| Fixed15 | 1,536 | 664 | Green; exact serial agreement |
| Dynamic, initial15 | 1,536 | 664 | Green; 38 updates, terminal depth 5/6 |

Each mode runs the same four 384-token stochastic requests, T0.7/k40/p0.9/seed4242:
fresh harbor, full repeat, partial mountain continuation, full repeat. All active
responses match the fresh Off control exactly. Full hits restore 214/214 or 289/289
prompt tokens; the partial hit restores 214/289. Active cases restore both MTP
and hybrid recurrent state. All four GPUs publish prefill/decode QSA capture
proof, with separate FA predictor obligations when enabled. Dynamic also passes
the native stochastic-controller/retained-transaction ticket validator.

Total: **24 requests / 9,216 tokens**. PMA, captured replay, transfer/movement
policy, driver observations and normal retirement pass. VRAM returns to the 40 MiB
host baseline after each run. Passive peak process-tree RSS sums are 39.66 GB Off
and 40.91–41.07 GB active; these sums are not deduplicated physical memory.
All observed owners retire. Exact owned cache cleanup removes 14.49 GB, leaving
about 506 GB free. No model/cache payloads are read or hashed for this accounting.

Evidence: `/tmp/qfn-a1-http-qualification-setup-r5/acceptance.json`, its invocation,
execution and resource receipts, and `qualification-progress.json`. Canonical
logs, observations, PerfStats, driver reports and cleanup receipts remain in
`parity-results/qwen-flash-next/a1-cap15-http-{off,fixed1,fixed2,fixed3,fixed15,dynamic}-r2/`.
Actual exports are in `/tmp/qfn-a1-runtime-exports-r5/`.

The earlier Off r1 remains red: generation passed but its FA-only observer
rejected QSA. Fresh r2 proves the fix; failed receipts were not restamped.
This is local runtime qualification, not corpus/image/release certification.
**Full OpenCode has not run.** The prepared client 1.18.34 adapter has 32 passing
pure tests and uses the separate production-serving export; refresh its plan
if source/binary identity changes before execution.

The historical 42.959 tok/s Castagna and 43.453 tok/s Llaminar results are
**MTP-off**. Castagna's published 119 tok/s profile uses TP2×PP2 and restricted
draft vocabulary. Neither number establishes this goal's TP4 MTP target.

| TP4 policy | Castagna prefill/decode | Llaminar prefill/decode | Accuracy / state |
|---|---|---|---|
| Off control | 466.22 / 42.67 tok/s | 819.46 / 43.36 tok/s | Fresh lease-fix control; unchanged IDs and speed; prior main-model panel green |
| Fixed 1 | 437.88 / 64.55 tok/s | 789.27 / 59.11 tok/s | Lease fix: 4.12× faster; exact original/own-Off IDs, 100% acceptance; still below fork |
| Fixed 2 | 435.81 / 78.45 tok/s | Pending | Fork: 100% draft acceptance; exact own-Off IDs |
| Fixed 3 | 433.91 / 91.38 tok/s | 788.79 / 89.91 tok/s | Lease fix: 2.85× faster; exact original/own-Off IDs, 100% acceptance; 1.61% below fork |
| Native adaptive, cap 3 | 435.80 / 90.86 tok/s | Pending | Fork: depth-3 work; exact own-Off IDs |

All completed timing arms retain native/quiet/driver/retirement passes and exact
own-Off output IDs. These are the earlier frozen lease-fix cohort; prepare a
fresh packet after checkpointing. The synthetic repeated-token prompt has 100%
draft acceptance and isolates execution cost. Use the prepared natural Python/
prose pair as a separate newly authenticated comparison, not a mixed series.

Raw timings live under `parity-results/qwen-flash-next/mtp-tp4-castagna-*` and
`mtp-tp4-llaminar-{off,fixed1,fixed3}-lease-r1`. Fixed3 samples span 89.8980–90.1235
for Llaminar and 90.9197–91.4698 for Castagna. The 1.61% median gap needs repeat or
interleaved observations before statistical claims. Native MPI uses
`OMP_PROC_BIND=close,OMP_PLACES=cores`; Castagna retains its own `false` policy.

The original fixed1 CPU profile found 41.14% sampled cycles in repeated
`ROCmBackend::retainNativeAllocation` scans during graph identity validation.
The accepted fix borrows once and checks frozen owner/backend/span identity on
replay. Existing borrower guards still forbid physical free/reuse; there is no
new allocation owner or PMA ledger. The fixed1 diagnostic improved 4.12× and
fixed3 improved 2.85×, with unchanged output IDs and verifier work. These rates
are not a new post-qualification benchmark.

Original captured traces showed 39.26 ms median spread between participant main
launches, while the final-arrival AllGather tail was about 0.081 ms. This exposed
host submission as the original deficit; it did not establish poor collective
bandwidth. Sampled-cycle percentages and profiled spans are not wall fractions.
Evidence: `mtp-tp4-fixed1-host-cycles-r1`, `/tmp/qfn-a1-fixed1-host-cycles-r1`,
`mtp-tp4-fixed1-captured-trace-r3`, and the clean Off diagnostic
`mtp-tp4-off-captured-trace-r5`. Earlier observer/profiler failures remain failed.
Profile the remaining post-fix deficit before choosing another optimization.

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
