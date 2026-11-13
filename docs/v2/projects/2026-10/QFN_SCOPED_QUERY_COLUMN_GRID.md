# Scoped ROCm query projection: column-stride grid

Status, 2026-10-25: component, canonical model accuracy and matched throughput
are green. This change reduces work in an
existing launch. Complete omission for admitted short requests is a separate
[retained-graph change](QFN_SHORT_CONTEXT_QUERY_OMISSION.md).

## Implementation and scope

The FP32-activation, native FP16/BF16-weight projection used by scoped QSA query
work now launches `min(N,128)` column blocks and iterates independent output
columns within each block. The old program launched one block per output column.
Rows, request owners, projection-group z coordinates and the exact five native
arguments retain their existing meaning. Inactive rows return before reading
operands. The column-loop exit is checked before incrementing to avoid signed
overflow.

Each column retains the original 256-thread K streams, FP32 accumulation,
FP16/BF16 conversion and reduction tree. An explicit retirement barrier prevents
next-column LDS writes from racing outstanding readers of the preceding column.
This selects only the full `FloatingProjectionRows` specialization; the four
compact Prefix/Request FP16/BF16 kernel bodies and resource records are byte-
identical to the preceding build. There is no arithmetic mode, live host count,
new buffer or production environment selector.

## Component evidence

Both build trees select the complete configured architecture set: ROCm gfx906
and CUDA80/86/89/90. The changed native projection kernel is ROCm-only.

Same fresh Release fixture, separately loaded old and candidate production
cores, isolated A/B/A. Control is committed `8baf5c58` from
`/tmp/qfn-attention-online-rescale-qualified-runtime-r1`. N512/K2560; grouped
cases use three owners with sixteen rows each. Times are microseconds per
complete projection, using retained chains of 32 operations, five warmups and
twelve samples per arm. The control column is the mean of the two bracketing
arm medians. Every observed timed output, guard, input and publication is checked.

| Work | Control | Candidate | Change |
|---|---:|---:|---:|
| FP16, ordinary, no ranking rows | 7.6152 | 3.5775 | −53.0% |
| FP16, ordinary, all rows active | 46.9562 | 21.1700 | −54.9% |
| FP16, grouped, mixed request holes | 229.3723 | 171.3823 | −25.3% |
| BF16, ordinary, no ranking rows | 7.6163 | 3.5800 | −53.0% |
| BF16, ordinary, all rows active | 47.0012 | 19.9600 | −57.5% |
| BF16, grouped, mixed request holes | 213.5411 | 165.0426 | −22.7% |

One-operation graph replays also improve: FP16 empty 63.1997→14.6400 µs and
full 63.3600→36.7200 µs. Those short event intervals include graph submission
effects and are not interchangeable with the 32-operation measurements or
isolated kernel timings. None of these numbers is a model-speed claim.

Final linked gfx906 evidence: FP16 uses20 VGPR/42 SGPR and BF16 uses16/40,
each with1,024 bytes LDS, zero private bytes and zero spills. The old scoped
programs used14/32 and10/32 respectively. These are used-register counts;
rocprof reports allocated SGPRs separately. Disassembly retains the original
FMAC/reduction structure and the required continuing-column barrier.

Separate rocprof invocations isolate one actual production projection after
five dirty-state and five warm replays. At M1/N512 the grid changes512→128
blocks, or2,048→512 wavefronts. Empty `VALUInsts` is65 on both sides; full
`VALUInsts` changes231.75→712 per work item as each thread handles four columns.
Full `FetchSize` is unchanged at2,571.9375 KiB. `WriteSize` is0.09375/0.5 KiB;
these hardware counters include cache effects and are not semantic output
extents. Both programs still author the same2 KiB output. Profiler timings
never enter the speed score. The pinned HIP occupancy API, queried against the
actual extracted modules, reports the same8 active256-thread blocks/CU for all
four old/new FP16/BF16 specializations:32 wave64/CU, or80% of the device's
thread-limit maximum. This is resource-limit occupancy, not measured achieved
occupancy. The isolated query process launched no kernels and retired cleanly.

## Correctness and matched model gate

Six focused canonical Integration/preflight registrations pass without skips:
native16 ordinary/grouped column boundaries, existing ordinary/grouped
all-format query work, counted M1 and floating request-row projection. Dense
native16 cases cover N127/128/129/511/512/513, K2560, rows1–16, independent
request owners and dirty empty/partial/full replays. The Release fixture also
checks the original arithmetic oracle and the actual captured argument layout.

The canonical QFN TP4 model panel passes:26 files,185 final-logit rows, all
byte-identical to each of five retained references. Default policy and actual
four-rank execution are authenticated. This is a scoped numerical pass, not
complete production preflight or an image certificate.

Complete Unit plus the two build-selection preflight aliases passes780/780
CTest registrations (778 Unit), retaining exactly the same95 inner GTest
omissions as the earlier full gates. The C++-only target-selection fixture was
corrected to authenticate and neutralize the observation target's HIP source
inside its metadata mock; production HIP configuration is unchanged. The first
local result auditor borrowed an empty inventory from a narrower gate. Its
failed receipt is preserved; a separate passive reconciliation proves all95
identities/reasons and the unchanged historical skip guards. There are no new
omissions. This interpreted-test-only correction changes no measured runtime
binary and requires no repetition of the model benchmark.

Matched unprofiled model timing uses512 prompt tokens,256 emitted tokens,
255 after-prefill steps, context4096, M512, four MI50s, RAM ngrams, sharded
vocabulary, FP16 KV, MTP/prefix cache off,28 physical workers28–55/NUMA1,
greedy seed42 and one warmup plus three measured responses per arm. All three
arms completed normally in a clean507.6-second observed window; all nine
256-token trajectories match.

| Matched model rate (tok/s) | Control, six samples | Candidate, three samples | Change |
|---|---:|---:|---:|
| Prefill median | 820.877407 | 819.899716 | −0.119% |
| Decode median | 42.372337 | 43.387635 | +2.396% |
| Decode range | 42.329321–42.520648 | 43.257426–43.462534 | Disjoint observed ranges |

The candidate exceeds the recorded matched Castagna42.959 decode reference.
This meets the throughput target for the stated cell; it does not certify other
topologies or establish an additive prediction from the component measurements.

Local evidence (not committed payloads):

- `parity-results/qwen-flash-next/query-column-grid-component-summary-r1/summary.json`
- `parity-results/qwen-flash-next/query-column-grid-integration-focused-r1/`
- `parity-results/qwen-flash-next/query-column-grid-model-fp16-r1/`
- `parity-results/qwen-flash-next/query-column-grid-unit-r2/`
- `/tmp/qfn-query-column-grid-unit-skip-audit-private/reconciliation.json`
- `/tmp/qfn-query-column-grid-isa-private/result.json`
- `/tmp/qfn-query-column-grid-occupancy-private/run-r1/result.json`
- `/tmp/qfn-query-column-grid-model-benchmark-r1/`

The candidate helpers, exact commands, source bindings and runtime identities
are retained under `/tmp/qfn-query-column-grid-{collector,model,benchmark}-private/`.
