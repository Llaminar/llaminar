# vLLM-Style MTP Tuning Dashboard

Current qualification frontier: 2026-10-09, hybrid PP(TP(2 ROCm), TP(2 CUDA)).
This dashboard stays below 200 lines. Older measurements and investigations are
preserved in [the October 9 archive](../2026-10/MTP_TUNING_DASHBOARD_HISTORY_20261009.md)
and [the project plan](MTP_VLLM_STYLE_PROJECT_PLAN.md).

## Completed evidence

- The four homogeneous dense/MoE CUDA2/ROCm2 OpenCode app cells completed:
  609 requests, 702 tools, seven stochastic tool errors; engine/protocol/progress
  checks passed. Model-authored application acceptance failures remain separate.
- r36 complete native prerequisites: 1,691 outcomes, 1,671 passed, 20 optional
  hardware skips, zero failures. Driver diagnostics were clean.
- r36 repaired RCCL protocol-default initialization passed native source-bound
  positive/negative and concurrent checks plus 20 fresh four-GPU repetitions.
- r36 dense Qwen3.8-27B hybrid Release smoke passed fresh, exact-prefix and
  extended-prefix requests with default dynamic MTP and context 262,144.
- The dense untraced 512-prefill/256-decode benchmark measured median
  693.17 prefill tokens/s and 24.80 decode tokens/s, essentially unchanged from
  the matched r28 baseline of 689.86 and 24.72.

## Active r37 corrections

MoE Qwen3.6-35B-A3B admitted maximum context but its first request failed:
root request-identity publication inspected only a flat expert placement and
missed independently admitted PP stage plans. The focused four-GPU regression
proved zero publications instead of one. Publication now authenticates the
common request authority across all stages before delegating to the children.

The deeper regression then exposed two further ownership defects:

1. Retained upstream MTP transactions omitted expert-placement acquire/release
   graphs. Those captured boundaries now surround verification and accepted-state
   publication and participate in graph reuse identity.
2. A restored prefix with a one-token suffix required upstream verifier-only
   participants to own shifted MTP KV. The scalar completion check now uses the
   same typed runtime ownership as chunked prefill. Completion is independent
   of whether a participant publishes logits, and the bridge submits all
   native pipeline domains through their concurrent captured transport owner.

Fixed, dynamic, and disabled MTP each pass twice in both four-GPU vendor orders,
including the complete three-request lifecycle. Disabled MTP exposed an ordinary
composition that incorrectly treated pipeline placement and sparse authority as
mutually exclusive. An explicit combined terminal composition preserves sparse
semantic replay and maintenance. Its restored-prefix scalar input now traverses
the captured pipeline instead of the sequential activation-transfer loop.
The combined matrix is green: twelve runnable MoE/dense four-GPU cases passed
twice (24 executions), six eight-GPU cells skipped, driver diagnostics were clean,
and every native owner retired. The sequence covers fresh requests,
exact prefix reuse, a one-token suffix, and 1/17/1 decode budgets. It calls the
production maintenance acknowledgement after each completed step and asserts
retained follower graph reuse. Both vendor orders, disabled/fixed/dynamic MTP,
and optional eight-GPU registrations belong explicitly to production preflight.
Ten related device-free CTest registrations passed. Matching Integration and
Release builds are green. The complete r37 prerequisite gate passed 1,703
outcomes: 1,677 passed, 26 hardware skips, zero failures; driver diagnostics
were clean and every native owner retired. The Release overlay is built and
r37 dense smoke and its benchmark passed (691.97 prefill, 24.59 decode tokens/s).
All three MoE answers passed, including emoji and exact repeated token IDs. The
post-run artifact collector rejected the stage-scoped movement schema. The
server retired normally with clean driver diagnostics; the preserved native
evidence passes after correcting the collector. MoE timing was not started.

## Current r39 qualification

The movement-artifact reader and prefix-epoch observer now preserve every PP
stage. Their focused cases, old-code negative controls and matching builds pass.
r39 completed 1,682 passed tests and 26 hardware skips, with no test failures;
its driver audit failed on one archived MI50 `SetWorkloadMask` firmware message.
All 100 targeted follow-up executions passed with clean driver windows and
complete retirement. The user waived only that archived firmware finding for
local progress; later driver checks remain mandatory. Fresh dense/MoE smoke
passed at context 262,144 with dynamic MTP, emoji and full/partial prefix reuse.
r39 benchmark medians were dense 692.26/33.25 and MoE 1,661.09/182.49 prefill/decode
tokens/s. The native processes retired normally with clean driver diagnostics.
The corrected benchmark reader admits unused prepared buckets while proving all
consumed rows; its old-reader negative control and twelve corruption variants
pass. No inference source changed or benchmark was rerun for that reader fix.
The resource observer originally rejected a valid MiB DRM mapping. r90 integrates
explicit binary-unit parsing, 53 passing scope/resource tests and its focused
preflight entry without repeating the app sessions.

## Remaining gates

1. Retain the focused and full native prerequisite evidence; eight-GPU execution
   remains unavailable on this host, which has only two CUDA devices.
2. Finish the matched whole-model context benchmark and refresh complete
   prerequisites for the integrated attention, ordering and metadata fixes.
   The user accepted partial dense/MoE app runs; further app stress is paused.
3. Complete the required commit gates, commit and push the feature branch.

The app contract remains maximum admitted context, 32 GiB shared disk cache,
16 GiB RAM cache per participant, default dynamic MTP, and no whole-turn
wall-clock deadline. Cache validation inspects metadata; it never hashes or
checksums model/cache payloads. The user paused r41 during dense's final review
after nine completed phases, then explicitly accepted dense and requested MoE
only. The original interrupted result remains unchanged: normal shutdown exceeded
the controller's wait, exceptional cleanup retired all owners, and the complete
driver window was clean. User acceptance is a local decision, not a completed
ten-phase or published-image certificate. The r43 MoE run completed five phases
and 37 app tests before the user's pause during input validation. Its server
shut down normally with exit code zero, all native owners retired, and the full
driver window passed without new findings. The harness preserved the app under
`moe/app/session-0000-webapp/workspace`. Original interrupted controller and
incomplete app-audit results remain unchanged; local user acceptance is recorded
in `hybrid-app-sessions-20261009-r43/user-pause-and-acceptance.json`.

The current priority is long-context decode. Passive r43 observations for
responses of at least 256 tokens fell from about 77 tokens/s below 16K prompt
tokens to 29 tokens/s at 64K–96K. Over 99.9% of the latter decode time was inside
`decodeStep`; native kernel attribution is still pending. These are changing
app requests, not a controlled scaling benchmark. The completed r44 Release
diagnostic holds topology, maximum allocation and dynamic MTP fixed: median
decode is 182.60/112.35/50.50/36.96 tokens/s at 512/16K/64K/96K prompt tokens,
with 99.3–100% MTP acceptance. All four cells and the driver window passed.
The r45 isolated retained-attention sweep passed three repetitions per vendor.
At 64K, physical M16/active M4 native-ring attention takes 9.10 ms on MI50 and
2.94 ms on RTX3090; these are isolated operation times, not model attribution.
ROCm tracing confirms the attention phase dominates that isolated operation.
Nsight's isolated CUDA probe produced four driver assertions; further CUPTI
attachment is stopped. Its receipt remains failed despite native test success.

The uninstrumented stochastic coding replay completed its first 60,184-token
request at 32.30 decode tokens/s (512 generated, 56.5% MTP acceptance), then
the repeated request failed in ROCm compact intermediate consumption during
its first decode step. The downstream CUDA transfer timed out and reported
Xid 43. This new correctness issue blocks repeated-request performance evidence;
the request, logs and failed driver receipt are preserved. A diagnostic-only
probe localized the mismatch before compact transfer. r92 then found zero
GDN/conv state in the cached verifier, despite correct restore samples and
identical projection input. Pipeline admission joined restoration on its
publication stream, but borrowed readiness before selecting the actual verifier
stream. The borrow now belongs to the forward prelude. r94 held-producer tests
fail all 20 requests/backend on the old core and pass on repaired ROCm/CUDA.
r96 and r100 registered native regressions pass. The r100 current Release replay
passes all three 60,184-token requests with 512 identical generated token IDs
against the saved fresh baseline, default stochastic dynamic MTP, clean driver
diagnostics and complete retirement. Fresh decode is 48.61 tokens/s versus 32.30
before; cached repeats are 45.78 and 45.65 tokens/s with 100% prompt reuse.
The first fixed run exposed a separate summary defect: a representative PP shard
omitted the CUDA tail's MTP state, and pointer presence misclassified metadata-only
lookups. Restore reporting now merges typed layout/tier metadata from all admitted
children and ranks, exchanging one protocol word. Focused metadata-only, four/eight
participant and native MPI regressions are explicitly registered in preflight.
The old summary negative control fails; all five r100 registered checks pass.

The r55 attention ABBA sweep matched all output bytes across 45 geometry/context
samples through 262K, with clean driver windows. At 64K, MI50 native M1 improved
from 2.854 to 1.098 ms and physical M16/active M4 from 9.136 to 4.143 ms. These are
1.45x and 1.42x the matched r53 RTX3090 timings. Static 256-wide specialization,
register-held FP16 queries and normalized ring cursors preserve reduction order.
r56 native profiling confirms three captured kernels, no graph copies and zero
spills; its instrumented timing is not benchmark evidence. r59 passed 38 captured
attention regressions. r69 additionally preserved FP32/FP16/BF16/Q8_1 bytes and
improved all four native formats. The focused ring regression passes and rejects
the old missing specialization. Its scalar oracle retains equal cache capacity.
The source is integrated and Release/Integration builds pass for shipping gfx906.
Expanded coverage found an existing narrow-head defect: scalar FP16 decode
converted to FP32 below width 64, unlike grouped native verification. Scalar
decode now retains the native format. Its old-code negative control fails;
the expanded test passes 195 geometries and six histories each, both directly
in Release and through its explicit Integration/preflight registration.
Installed r79 ABBA timing is 2.847 -> 1.077 ms (native M1) and 9.140 -> 4.073 ms
(physical M16/active M4), with identical bytes. r81's 48 isolated profiler runs
cover all eight native format/width bodies: three kernels per retained graph,
no graph copies, zero scratch spills and clean driver windows. r87's native
occupancy query admits 6–8 blocks per CU, with zero per-thread local memory;
this is maximum residency, not achieved occupancy. The r101 matched unprofiled
Release benchmark passed all four contexts with clean driver windows and retirement.
512/16K/64K/96K decode is 184.69/142.26/80.22/61.54 tokens/s, gains of 1.1%/26.6%/58.8%/66.5%.
Prefill is unchanged. Complete Unit/preflight evidence for the final source is
recorded separately in `long-context-final-prerequisites-20261009-r102/`.
Live-KV reuse remains deferred until the agreed feature checkpoint.

## Local receipts

All paths below are relative to the ignored
`parity-results/opencode-tool-calling/resource-growth-work/` directory.

| Evidence | Receipt |
| --- | --- |
| Four homogeneous app cells | `four-cell-completion-audit-20261008-r3/result.json` |
| r36 complete prerequisites | `hybrid-candidate-prerequisites-20261009-r36/controller.json` |
| r36 real smoke/benchmark | `hybrid-smoke-benchmark-20261009-r36/` |
| r37 real-model outcomes | `hybrid-smoke-benchmark-20261009-r37/` |
| r39 artifact collector proof | `hybrid-terminal-artifact-20261009-r39/result.json` |
| r39 active continuation | `hybrid-candidate-continuation-20261009-r39/controller.json` |
| Current machine-readable progress | `hybrid-progress-20261008.json` |
| Long-context candidate ABBA | `long-context-attention-ab-20261009-r55/` |
| Native-format timing and bytes | `long-context-attention-formats-20261009-r69/` |
| Installed Release regressions | `long-context-attention-installed-regressions-20261009-r73/` |
| Cached-prefix ordering, both vendors | `pipeline-verifier-admission-registered-20261009-r96/validation.json` |
| Prefix metadata composition/MPI | `prefix-restore-metadata-focused-20261009-r100/validation.json` |
| Current three-request 60K replay | `long-context-http-admission-metadata-fixed-20261009-r100/controller.json` |
| Installed context benchmark | `long-context-decode-installed-20261009-r101/controller.json` |
