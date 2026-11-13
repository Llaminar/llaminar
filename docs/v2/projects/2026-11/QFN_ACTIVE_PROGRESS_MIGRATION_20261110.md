# QFN migration continuation bounded by active inference progress

Two matched Release lifetimes measure **331.486/39.679** and
**327.546/39.893 prefill/after-prefill decode tok/s**, versus the preceding
checkpoint's **314.818/39.988**. Prefill improves4.04–5.29%; decode is0.24–0.77%
lower. This is a measured prefill improvement with a small observed decode
cost, not completion of the500/40 goal or a statistical significance claim.
Both candidates match all768 output IDs, complete MTP records and resolved
configuration except output paths.

## Why and what changed

The [preceding cache/planner slice](QFN_CACHE_AND_PLACEMENT_ECONOMY_20261110.md)
made exact two-tier placement cheap but still allowed only one bounded wave
per fresh histogram. An immutable observation can justify further beneficial
placement after its original wave has fully retired.

`MoEOverlayResidencyAuthority::CommittedObservation` retains the original
PMA-owned immutable demand and a weak exact committed-snapshot identity. It
does not retain old expert banks, reader leases, transactions or transport
storage. Only a successful certified placement publication issues this handle.
Reproposal uses the original current-owner solver and coherent objective;
it neither re-ingests the histogram nor advances or resets its cadence.

The existing maintenance service attempts continuation only after publication,
acknowledgement and old-reader reclamation, and only when completed inference
progress has advanced. Each attempt consumes all currently visible progress;
idle polling cannot accumulate a queue of credits. Fresh Ready evidence wins
at every new selection boundary; pending drains retain precedence. Deferred
attempts retain their original admitted transaction. Every continuation must
still have strictly positive coherent service gain and net benefit.

The first fresh publication establishes its progress watermark at Published,
conservatively discarding progress during that first wave. Later continuations
preserve their admission watermark, allowing progress during their active wave
to fund one later attempt after reclamation. Empty/no-benefit, stale, failure
and shutdown paths release the handle. Existing MPI codecs, acknowledgement
rules, five physical transfer slots, costs, thresholds and numerical arithmetic
are unchanged. The implementation remains in the original authority/service;
there is no second controller or accounting ledger.

## Fixed real-model comparison

The original application owner performs automatic MPI bootstrap on two actual
CPU sockets, with28 physical workers/socket. The NUMA1 rank owns both local
MI50s. Model: four-part Qwen3.8-Flash-Next UD-Q4_K_XL GGUF and shared Q8_0 MTP
sidecar. Context262144, actual512-token prefill in448/64 chunks,256 output
tokens, seed42, one warmup and three measured requests per lifetime. Decode
rates use255 after-prefill tokens/request. Dynamic MTP and expert balancing
remain enabled, with RAM ngrams, ConsumerScheduled GPU caches and16/32GiB
RAM/disk prefix budgets.

Fresh automatic admission remains identical:8066 GPU experts total,
4033/card, and17022 logical experts tensor-sharded across the CPU cohort.
Neither capacity nor initial expert placement changes. Both processes retain
the qualified GNU OpenMP runtime and the selected ROCm10.1 SDK environment.

| Lifetime | Prefill tok/s | Decode tok/s | Measured promotions / waves | Actual movement bytes |
|---|---:|---:|---:|---:|
| Checkpoint0a2876b3e |314.818|39.988|24 /6|219,852,800|
| Continuation first |331.486|39.679|52 /13|515,072,000|
| Continuation repeat |327.546|39.893|48 /12|434,790,400|

Root-participant completed CPU prefill routes decrease from488,500 to455,700
and458,676 (6.71% and6.11%). Completed grouped-verifier routes decrease from
246,675 to227,514 and231,095. These counters describe actual work on one
participant, not socket sums or estimated savings. Empty local work bypasses
the nonempty completion counter, so packet counts can differ. Generated
histories and complete MTP work still match exactly.

More waves also mean more scoring, repacking and transfer work. The existing
projected interference costs do not measure additional planner CPU contention,
and their65536-token payoff horizon is not the remaining benchmark duration.
An already admitted wave must finish before fresh evidence can be selected.
Do not convert forecast benefits, overlapping waits or extra movement counts
into a claimed wall-time gain; the measured phase rates above own that result.

## Memory and lifecycle

Peak per-process RSS is36.015/77.069GiB and36.008/77.079GiB, versus
36.008/77.066GiB in the control. Selected per-client peak VRAM stays about
30.308/30.076GiB, also matching the control. Original resource observers report
no errors. These short lifetimes show comparable bounds, not a long-running
leak certificate.

Original native, driver and process-tree retirement pass in248.755s and
252.270s. After retirement each run reclaims about728MB of owned prefix files,
reading28,677/28,682 metadata bytes and zero payload bytes. No model/cache
payload scans or hashes are used.

## Focused regression qualification

Canonical Release rebuild passes in167.118s. The controller change is host
code; the configured CUDA80/86/89/90 and HIPgfx906/gfx1100 policy remains
unchanged. A test-only correction rebuilt in17.680s: the histogram starts at
generation0, and three new assertions incorrectly assumed1. Tests now capture
the original generation and assert exact same-generation reuse or advancement
by one. Placement epochs and objective checks remain unchanged.

Eleven focused registrations pass: original maintenance, authority, exact
planner, tier rebalancer and owner-map units; original full MPI publication;
cadence and local/MPI retirement; and the two new explicit preflight aliases:

- `V2_Integration_MoEActiveProgressContinuationReplay`
- `V2_Integration_MoEContinuationOriginalPublisherReplay_MPI`

Each new replay registration also passes20 consecutive runs (40 total,33.083s).
The controller cases cover four same-observation waves, residency0/2,
independent coherent-objective decrease, fresh priority, progress exhaustion,
old readers, deferred retries, identity rejection, shutdown and failure.
The actual MPI publisher runs three sequential same-generation/later-epoch
publications and rejects rearming before its original storage retires. Existing
delayed-ack tests supply complementary distributed lifecycle coverage.

The authored controller fixture proves state/economy semantics; it is not a
model-route or hardware performance replay. The actual two-GPU/two-socket
model above supplies the physical cohort comparison. Complete current
preflight, other-topology numerical panels and OpenCode remain deferred;
these focused results do not claim those certificates.

## Evidence and next work

Local records are under `/tmp/qfn-moe-active-progress-continuation-next/`:
`model-comparison.json`, `repeat-comparison.json`, both original model owners,
`repeat-focused/`, and `review-followup/NOTE.md`. Build and focused CTest
receipts are under `/tmp/qfn-cpu-eight-row-rocm101-root/active-progress-*`.
Generated result directories are intentionally excluded from source control.

Next isolate the measured NativeQuantised M448 prefill dispatch changes and
investigate redundant exact scoring of two-edge tier exchanges. Keep smaller
rows, decode/MTP dispatch and the general coherent objective unchanged while
measuring each candidate independently.
