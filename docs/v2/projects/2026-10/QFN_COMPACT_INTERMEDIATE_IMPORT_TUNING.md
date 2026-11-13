# QFN compact intermediate import tuning

Date: 2026-10-21. Candidate67 follows the worker-affinity checkpoint
`545192c8bc99f29f5d68d68ba63ebe43b8da74c4` on `feat/qwen-flash-next`.

**Outcome: parked.** The candidate passed accuracy and reduced isolated import
time, but its measured model throughput stayed within the control's variation.
Its source/test changes were restored to checkpoint69. This note records the
experiment, including the failures, rather than a new production implementation.

## Change and ownership

The counted MoE projection pipeline imports every participant's live compact
packet in one CUDA/HIP kernel. Previously it launched one import per participant.
The two-dimensional grid names a source participant and a source-local route.
The expected four-participant reduction is four imports to one per routed layer.

The original counted binding issues a closed, owner-retaining view of its device
extent bank. The public import authenticates the original device, participant,
packet owners, addresses, stride and physical capacities before capture. It adds
no allocation, pointer table, host count readback or second accounting authority.
Only acquired live records are consumed; communication extents and transport
scheduling are unchanged. The payload is copied as original 32-bit words, leaving
Q8 values, FP32 scales, floating intermediates, down projections and reduction
order intact. The old scalar public import and production loop are removed;
the low-level scalar kernel remains an explicit component control.

## Defects found while validating the candidate

The first CUDA version failed all five affected focused paths. Memory checking
reported an invalid read at address `0xa`; emitted PTX used scalar-launch field
offsets against the outer cohort argument when inlining a temporary aggregate.
Passing explicit scalar and pointer operands to the shared copy helper restores
the selected source addresses. The corrected PTX resolves the actual count bank
and source offset on SM80/86/89/90. HIP uses the same helper and arithmetic.
Original failures and corrected evidence remain separate.

The initial Release component invocation discovered no tests because the new
performance registrations were enclosed by an Integration-only block. Moving
them beside their executable target exposes all four exact selectors in Release.
The configure/build pass took 4.06 seconds with no compilation work; the three
focused correctness registrations retain identical commands and properties.

## Completed correctness and build evidence

- The complete owning Integration rebuild passed before running consumers of
  the changed virtual interface. The corrected [focused suite](../../../../parity-results/qwen-flash-next/compact67-after69-fixed-focused/result.json)
  passed **21/21** registrations, including CUDA2, ROCm2/4, public counted
  projections, all-format checks and adversarial replay.
- The complete [Unit suite](../../../../parity-results/qwen-flash-next/compact67-after69-unit/result.json)
  passed **754/754**. The experimental patch includes explicit
  `ProductionTestPreflight` registrations for its contract and native cohort
  regressions; those registrations remain with the parked patch.
- The corrected [CUDA memory check](../../../../parity-results/qwen-flash-next/compact67-after69-cuda-memcheck-fixed/result.json)
  reported **zero errors**, with clean driver observations and native retirement.
- The [Release build](../../../../parity-results/qwen-flash-next/compact67-cuda-fixed-after69-release-build/result.json)
  passed for CUDA SM80/86/89/90 and HIP gfx906 with spill guards enabled.
- The frozen [A1 Release accuracy diagnostic](../../../../parity-results/qwen-flash-next/compact67-cuda-fixed-after69-release-accuracy-diagnostic/report.json)
  passed its required short and heldout groups, reset, driver and retirement
  checks. All **26 final-logit files / 185 full-vocabulary rows** are
  [byte-identical to checkpoint69](../../../../parity-results/qwen-flash-next/compact67-cuda-fixed-after69-release-accuracy-diagnostic/ngram69-logit-byte-comparison.json),
  including reset and diagnostic boundary rows. Thresholds and histories are
  unchanged; the N2047 diagnostic group remains outside numerical acceptance.

Short-panel prefill/decode mean KL is **0.033535869 / 0.032070057**, top-1
agreement **93.75% / 97.92%**, and top-5 overlap **87.50% / 92.29%**. Heldout
mean KL is **0.009691368 / 0.003837619**, with **100%** top-1 agreement in both
phases. These focused observations do not replace complete production preflight;
G60 remains the latest complete prerequisite qualification.

## Component timings and compiler resources

Each Release selector runs one backend and one import mode in its own process.
The synthetic A1 geometry is F640, ten routes, four participants, local owner
zero. One retained graph contains four packet packs plus four scalar imports or
one cohort import, without transport, down projection or folding. Five warm
batches precede 31 measured batches of 64 replays. Native events supply timings;
byte, count and poisoned-tail checks remain active.

| Backend | Scalar median | Cohort median | Time reduction |
|---|---:|---:|---:|
| MI50 / ROCm | 23.3125 µs | 16.7250 µs | 28.26% |
| RTX 3090 / CUDA | 16.4000 µs | 10.3200 µs | 37.07% |

The [component comparison](../../../../parity-results/qwen-flash-next/compact67-cuda-fixed-after69-component-comparison.json)
retains all samples and separate driver/retirement receipts. These are component
costs, not full-model throughput results.

Final Release object inspection gives cohort/scalar CUDA register counts of
**32/24** on SM80, **36/24** on SM86 and SM89, and **38/26** on SM90, with zero
shared/local/stack storage. HIP gfx906 uses **29/21 VGPRs** and **62/44 SGPRs**,
with zero vector/scalar spills, private storage and LDS. Cohort native attributes
report at most 12 blocks/SM on CUDA and 16 on HIP at 128 threads/block. These are
resource and occupancy ceilings, not measured occupancy. Raw isolated object/ISA
evidence is retained under `/tmp/qfn-compact67-release-static-results-r1/`.

## Model measurement and decision

The matched A/B/B/A comparison uses checkpoint69's archived Release runtime and
the corrected candidate. Both use four MI50s, `TensorParallel` dense computation,
`GateUpOwnedDownColumns` routed experts, RAM ngrams, 28 physical NUMA1 workers,
512 input tokens, 256 output tokens, M512, C4096, FP16 KV, greedy seed42, MTP off,
prefix caching off and PerfStats off. Each arm runs one warmup and three measured
requests. Model shards, exact prompt IDs, native precision policies and loaded
runtime dependencies are authenticated independently of timing.

The production policies remain `single_q8` FA/GDN output and GDN QKV operands,
`fp16_moe_alpha_beta`, `two_term_q8` HC up and `fp16` routing. Original quantized
weights remain unchanged. The candidate receipt is
`/tmp/qfn-a1-candidate67-cuda-fixed-after69-registration-fixed-runtime-receipt.json`
(SHA256 `d28aa3bd46efbda0a77b17b84ea78cffb665fca1bca574b8967fda263348b8e0`).

The [complete A/B/B/A comparison](../../../../parity-results/qwen-flash-next/compact67-cuda-fixed-after69-abba-comparison.json)
passed every runtime/source, driver, resource and retirement check. All twelve
measured 256-token output sequences are identical. No arm was excluded or replaced.

| Arm | Median prefill tok/s | Median decode tok/s |
|---|---:|---:|
| A1 control | 784.938 | 33.365 |
| B1 candidate | 784.453 | 33.378 |
| B2 candidate | 785.071 | 33.441 |
| A2 control | 785.184 | 33.252 |
| All six control samples | 785.062 | 33.325 |
| All six candidate samples | 784.594 | 33.389 |

The observed changes are **−0.060% prefill / +0.194% decode**. Ranges overlap:
control decode **33.224–33.408**, candidate **33.334–33.443**; control prefill
**784.451–786.254**, candidate **784.085–785.507**. This does not establish a
model-level benefit beyond measurement variation. **Do not promote this candidate
as a throughput improvement.** Preserve its complete source/test patch and results
under `parity-results/qwen-flash-next/compact67-parked-source/`; checkpoint69
remains the accepted implementation and dashboard throughput record.

The separate [retained-graph inventory diagnostic](../../../../parity-results/qwen-flash-next/compact67-cuda-fixed-after69-graph-inventory-diagnostic/analysis.json)
completed native execution, driver checks, retirement and all 185 byte comparisons.
Each device's selected complete-decode family contains 96 cohort nodes across two
aggregated captures, grid **10×4×1**, block **128×1×1**, and no scalar import.
Each device reports all **159** decode/reset replays. The existing inventory tags
aggregate identical captures, so the analyzer reports
`FAMILY_COUNTS_MATCH_INSTANCE_RESOLUTION_UNAVAILABLE` and does not assert individual
capture identity. This is positive family/submission evidence with an explicit
schema limit, not a complete per-instance certificate or a timing measurement.

The complete 19-file candidate patch has SHA256
`dedf298a5548d613cfa9c781a15bea4bac9b8714b9619a49898aa7f0b07a7d31`.
After native retirement, all candidate production/test changes were restored to
checkpoint69, preserving the independent corpus edit. The restored owning
[Integration targets](../../../../parity-results/qwen-flash-next/compact67-restored69-integration-build/result.json)
rebuilt in **191.24 seconds**, and the [Release server and canary](../../../../parity-results/qwen-flash-next/compact67-restored69-release-build/result.json)
rebuilt in **23.24 seconds**, restoring executable/core ABI consistency. The
restored [complete Unit suite](../../../../parity-results/qwen-flash-next/compact67-restored69-unit/result.json)
then passed **754/754** in **115.54 seconds**, with clean driver observations and
normal native retirement.

The next performance step is fresh timeline attribution on the accepted
implementation. Earlier L62
traces predate transfer scheduling and worker-placement fixes; their kernel sums
cannot rank the current exposed costs. In particular, the component import win
does not explain the roughly **6.8 ms/token** remaining gap to Castagna's 42.959
tok/s. Complete current-source production preflight remains pending.
