# CPU GDN fixed-epoch production experiment

Follow-up: the [CPU epoch tuning results](gated-deltanet-epoch-tuning-results.md)
record the subsequent fixes, green selected qualification and exact-develop comparison.
The measurements and failures below remain the original experiment's evidence.

Date: 2026-10-25. Branch: `feat/gdn-speculative-leaf-harness`, based on
`3dd32e833287f027924b595895f176af16fd29a5`. CPU only. The user authorized changing
arithmetic while retaining one byte-identical serial/grouped contract and asked
for production implementation and whole-model measurements.

This follows the [three-candidate research comparison](gated-deltanet-shared-readout-and-backward-query-results.md).
The selected candidate combines backward query contraction with shared rounded
prefix advancement, using a fixed eight-position checkpoint period. The earlier
1.26x result applied to branching component workloads. The production verifier
currently supplies linear candidates, so that result is not a model forecast.

## Implementation and arithmetic

`CPUGDNStateLayout` owns the CPU physical representation and arithmetic identity
`fixed-epoch8-backward-fma-v1`. Each head contains a phase, a dense checkpoint,
and eight prepared key/value/decay/beta records. Inactive records are zero.
For 128×128 heads this is 73,796 bytes, versus 65,536 bytes previously: +12.6%.
CPU allocation, memory admission, request banks, verifier slots and durable
prefix snapshots all use the complete extent. GPU dense state is unchanged.
Old CPU prefix entries have a different fingerprint and cannot be reused.

Each query traverses the live records newest to oldest with increasing-key
FP32 FMA reductions. It projects the checkpoint once and applies value FMAs
oldest to newest. Up to four grouped queries share checkpoint loads without
changing any query's reduction order. Exactly every eighth position advances
the checkpoint through the ordered rounded updates; intermediate prefixes are
shared for that advancement, never substituted into a readout. The period is
part of the contract, independent of verifier depth, grouping and thread count.

Ordinary decode, prefill, merged QKV and ragged request batches call the same
program. Verifier execution owns its working state in the final snapshot slot,
preserves the initial live bank and publishes a complete state for every real
row. Rejection and acceptance select those states without a partial flush.
Padding neither advances phase nor contributes a raw update. There is no
production loop calling the M=1 kernel for each grouped row.

The CPU code records readout rows, projection groups, rows using grouped
projections and canonical checkpoint flushes in PerfStats. The independent
oracle includes a cancellation witness: retaining history gives a continuation
of `2^-25`, while prematurely materializing a dense state gives zero.

## Qualification

The candidate is **not qualified for promotion**. Complete CPU prerequisites
pass: 737 Unit and 613 ProductionTestPreflight entries, 1,350 total. Seven dense
Qwen3.5 CPU cells and five selected Qwen3.6 IQ3S cells pass unchanged HF reference
and serial/grouped gates. The latter cover MTP off and fixed depths 1, 2, 3 and
15. The dynamic-depth cell fails on three consecutive runs; these are twelve
passing model cells and one failed cell, not a complete model certificate.

The dynamic failure is the existing requirement that a designated full-width
transaction accept at least one draft. It accepts zero of fifteen proposed
drafts. Its numeric comparisons and serial token oracle pass; both emit
`271;71093`. The old one-hop implementation emitted `271;760;…` and accepted nine
drafts in its witness. At that second position the new output agrees with the HF
reference, but changes the path on which the acceptance witness depends. Neither
the test nor its reference was weakened. This evidence does not establish an
arithmetic defect, but it leaves the required dynamic acceptance gate red.

The independent scalar oracle checks 55 key/value geometries through width 512,
17-step continuations, all eight incoming phases and grouped lengths through
63, every accepted endpoint, complete rejection and subsequent M=1 continuation.
It also rejects malformed phases and aliased/undersized snapshot banks before
publication. Focused suites pass in both AVX2 and AVX-512 builds. A discovered
zero-layer memory-accounting defect was fixed and registered separately as
`V2_Integration_CPUFixedEpochGDNZeroLayerAccounting` in production preflight.

A separate instrumented Release MoE run completes 384 stochastic tokens with
dynamic depths 1–15: 153 verifier transactions/commits, 231 accepted drafts and
zero transaction validation failures. PerfStats proves 1,164,480 epoch readout
rows, 352,320 projection groups, 1,124,160 rows using grouped projections and
144,960 checkpoint flushes. This confirms execution of the intended production
path; it does not replace the failed canonical witness.

All receipts remain under the ignored `parity-results/gdn-epoch-production/`
directory. The tested scope is CPU only and selected model cells; it is not a
full MoE format matrix, GPU qualification or release certificate.

## Measurement protocol

The frozen baseline is the previously qualified one-hop CPU implementation,
`adjusted-parent-fma-v1`, including its executable and core DSO. Its executable
SHA-256 is `e9f2bd86d7a1be01f47e90efb28366282a1c22e2a1d38ca8b5cebc3e45483095`;
its core SHA-256 is
`3c65f3f1b6007da7ab1fb94b267a42734ece7fee368c48a327dc3fc722a43c89`.
The snapshot includes the source diff and untracked source files. The baseline
process explicitly loads that frozen DSO, rather than the rebuilt worktree DSO.
The candidate uses the same executable hash and core SHA-256
`cd6081edccad9fa7b11aaeba87d19accfd8357677308640db60fd9b636caf16c`.
Both loader closures select the configured GNU OpenMP runtime.

Canonical timing uses Release AVX512, normal single-rank MPI startup, one
physical socket of the Xeon Gold 6238R (CPUs 0–27), 28 workers, a 4,096-token
context and 384 generated tokens. Each fresh process uses one warmup and one
measurement, seed 42, the same built-in 595-token prompt and prefix caching.
The alternating order is baseline/candidate/candidate/baseline/baseline/candidate.
Dense workloads use greedy sampling; the MoE workload retains stochastic
dynamic-depth MTP. Instrumented path proofs are separate from timing.
Every measured request processes all 595 prompt tokens: prefix caching is enabled
but the recorded hit count is zero. Combined throughput below covers prefill
plus generation, excluding model loading, startup and warmup.

An initial small-model series overlapped another benchmark and is excluded.
The final eighteen-run series started after ten seconds without another model
benchmark; every completion probe also found none. This experiment launched no
builds, tests or profiler runs during the series. Raw excluded runs are retained
separately.

Hardware counters are denied by the host (`perf_event_paranoid=4`); the failed
`perf stat` probe is retained. Assembly inspection covers fourteen compiled
AVX2/AVX-512 projection/flush routines and finds no packed-vector stack references.
The four-row AVX-512 projection reuses a checkpoint load across four FMAs.
This is assembly evidence, not measured IPC or cache-miss evidence.

## Grouped kernel economy

The production verifier-row diagnostic uses M=8, sixteen key heads, sixty-four
value heads, dk=dv=128, complete post-row snapshots, ten warmups and forty
alternating trials. Times below are best-of diagnostics within the new common
contract, not comparisons to the old baseline or averages over checkpoint phases.

| Physical workers | Grouped ms | Serial ms | Serial / grouped | Economy gate |
|---:|---:|---:|---:|---|
| 1 | 6.920 | 3.093 | 0.447x | Fail |
| 7 | 1.170 | 0.706 | 0.603x | Fail |
| 14 | 0.660 | 0.561 | 0.850x | Fail |
| 28 | 0.501 | 0.575 | 1.148x | Pass |

Every output/state byte comparison passes. Grouped latency scales 13.8x from one
to twenty-eight workers, but that does not satisfy the economy gate at the lower
worker counts. Every candidate still publishes a complete checkpoint/history
snapshot; reduced projection arithmetic does not remove that traffic. Without
hardware counters, the measurements do not establish which traffic dominates.

## Full-model results

All eighteen fresh-process runs completed successfully. Each implementation's
384 generated token IDs repeat exactly across its three runs for every model.
Throughputs are medians within each implementation. The combined speedup is the
median of three adjacent baseline/candidate ratios, not the quotient of the two
independently computed median throughputs. Ranges show all three paired results;
they are not confidence intervals.

| Model | Prefill tokens/s, old → new | Decode tokens/s, old → new | Paired prefill + decode speedup | Pair range |
|---|---:|---:|---:|---:|
| Qwen3.5 0.8B Q4_0 | 1,669.03 → 1,488.75 | 37.77 → 41.16 | **1.083x** | 1.026–1.128x |
| Qwen3.5 4B Q8_0 | 316.64 → 295.36 | 15.47 → 15.57 | **0.981x** | 0.931–1.008x |
| Qwen3.6 35B-A3B IQ3S, dynamic MTP | 189.60 → 181.44 | 14.46 → 20.46 | **1.328x** | 1.282–1.352x |

Thus the observed combined throughput changes are +8.3%, −1.9% and +32.8%.
Paired decode-only median changes are +9.0%, −1.3% and +39.2%, respectively.
There is no demonstrated 4B win, and median prefill throughput decreases on all
three models. These are three repeats of one prompt on one CPU socket, not a
cross-workload performance claim.

The arithmetic change alters generated tokens: the first differences from the
old contract occur at zero-based positions 83, 51 and 54, respectively. This does
not violate the new contract's serial/grouped byte equivalence, but it means the
comparison is not a fixed-token replay. In particular, MoE changes routing and
dynamic verifier work. Its counts repeat within each implementation:

| MoE measured work | Old one-hop | New fixed epoch |
|---|---:|---:|
| Generated tokens | 384 | 384 |
| Accepted drafts | 243 | 231 |
| Draft steps | 817 | 465 |
| Verifier transactions | 141 | 153 |
| Verifier token count | 958 | 618 |

The MoE speedup includes this reduction in draft/verifier token work. These
measurements cannot attribute the full gain to faster GDN arithmetic. All runs
use the same prompt hash, model files, sampling settings, physical cores and
worker count; executable/core identities and exact commands are in the manifest.

## Verdict and evidence

The combined P=8 candidate is implemented in the CPU production path and retains
the required common arithmetic contract. It shows a useful full-model gain on
the measured 0.8B and MoE workloads, but is **not a general replacement ready for
promotion**: the dynamic acceptance qualification remains red, grouped economy
fails below twenty-eight workers in the measured case, state storage grows 12.6%
at 128×128, and prefill regresses. Keep it experimental. Acceptance coverage and
economical grouped execution must both pass before promotion; a favorable model
timing does not waive either gate.

Local evidence root: `parity-results/gdn-epoch-production/` (ignored, not a source
or corpus payload). Review entrypoints:

- `prerequisites-r2/prerequisites.json`, `avx2-final.log`: complete prerequisite
  receipt and separate AVX2 focused checks.
- `model-dense-r1.json`, `model-moe-{off,depth1,depth2,depth3,depth15}-r1.json`,
  `model-moe-dynamicdepth-r{1,2,3}.json`, `qualification-status.json`: unchanged
  model gates, including all three failed dynamic witnesses.
- `baseline-onehop/identity.json`, `candidate-release-identity.json`,
  `candidate-source-identity.json`, `release-runs.json`,
  `measurement-model-identities.json`: source snapshots, binary identities,
  exact commands, environments and model paths.
- `release-{08b,4b,moe}/`, `performance-summary.json`, `measurement-status.json`:
  all eighteen timing/token artifacts, paired calculations and completed status.
- `run_gdn_epoch_benchmarks.py`, `summarize_gdn_epoch_measurements.py`: preserved
  local drivers; the benchmark driver refuses to overwrite existing runs.
- `release-path-proof-summary.json`, `release-moe-perfstats.json`: separate
  instrumented production-path proof.
- `cpu-scale-quiet-{1,7,14,28}.csv`, `assembly/`, `perf-access.log`: grouped
  economy, assembly inspection and unavailable hardware-counter evidence.
- `concurrent-timing-exclusion.json`, `release-08b-concurrent/`: excluded initial
  timing series; these values do not enter the reported results.
