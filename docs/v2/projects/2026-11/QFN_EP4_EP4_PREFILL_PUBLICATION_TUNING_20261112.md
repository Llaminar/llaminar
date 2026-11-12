# QFN eight-MI50 prefill publication repair

## Result

Removing an addressable zero-vector temporary from shared-expert publication
improves matched full-model prefill from **734.870 to 789.433 tokens/s (+7.425%)**.
Median prefill time falls from **696.722 to 648.567 ms**. This is a focused
Release result on `PP(EP(4×MI50), EP(4×MI50))`, with ngrams sharded in VRAM across
all eight GPUs. It is not a complete preflight or release certificate.

The production change preserves every bank, copy, required zero, launch extent,
stream and collective. It changes no floating-point arithmetic, reduction,
quantization policy, expert placement or model input. CUDA receives the same
explicit copy/zero branches; its compiler already avoided the private temporary,
and its matched component timing is unchanged.

## Matched production workload

- Qwen3.8-Flash-Next-UD-Q4_K_XL, the original four GGUF shards. Model payloads
  are not copied, converted or hashed for this experiment.
- Discovery MPI rank 1 owns both pipeline stages on NUMA 1. Stage 0 uses
  ROCm 7/6/8/9, layers 0–23; the tail uses 1/3/4/2, layers 24–47. Physical
  UUID/PCI/NUMA/KFD inventory, original rankfile and configuration are unchanged.
- Fixed-head ngram VRAM ownership across all eight GPUs; Static/Ordinal experts.
- FP32 activations, FP16 KV, existing FP16 router operands/FP32 accumulation,
  MoE FP16 alpha/beta and HC two-term Q8 policy. MTP and prefix caching are off.
- Exact raw prompt `" test"` repeated 512 times, context capacity 4096, captured
  M512, greedy sampling, seed 42. One warmup, three measured requests.
- This prefill-focused comparison emits **one token**, with no after-prefill
  decode steps. It uses the public benchmark and original MPI/session lifecycle.
  `runPrefill` ends at durable inference completion, before token emission.
  Medians use each request's `512000 / prefill_ms`, not a mean-time aggregate.
- No profiler, GPU stage clocks, visibility override, driver change or competing
  compiler/native job is present during either timing run.

The earlier 512/256 baseline was 732.923 P / 37.260 D tokens/s. The fresh 512/1
control agrees closely, but the changed request length remains explicit. No
decode throughput or long-context scaling result is inferred from this pass.

| Measurement | Original, q153 | Repaired, q163 |
|---|---:|---:|
| Prefill request 1, ms |697.139366|649.368066|
| Prefill request 2, ms |696.721968|648.566754|
| Prefill request 3, ms |695.856174|647.960964|
| Median prefill, tokens/s |734.869896|789.433003|
| First generated token, all requests |1228|1228|

All eight owners' fixed-native ngram call, live-ID, payload-byte and retirement
tuples match exactly. Native processes and descriptors retire; all eight GPUs
return to **10,866,688 bytes** used VRAM. Driver windows are clean. RSS and VRAM
are sampled throughout loading/inference/retirement; these short runs do not
establish a long-duration leak bound or phase-specific memory peaks.
The two native processes' observed high-water RSS values are 750,653,440 and
21,941,452,800 bytes, reported separately; neither uses swap. Twelve samples
span 55.29 seconds. Peak sampled VRAM is 20.930–21.323 GB per GPU.

## Why this was selected

The retained full-model trace has three prefill launches per captured executable:
readiness, warmup and measurement. Selecting the third occurrence of every stable
node, authenticated against the benchmark's M512 replay counters, gives 13,340
measured dispatches and a 698.605 ms GPU envelope. CPU graph-launch intervals
are not used as GPU completion timestamps.

Shared-expert publication executes 24 times per GPU. Its measured totals are
23.754–25.673 ms/GPU, with **no overlap with any other selected kernel on that
GPU**. Each follows routed expert work and precedes the rooted collective.
Collective wait intervals and overlapping dense projections are not added to
invent a critical path or transport bill.

At M512, H2560, top10 and four participant banks, the publisher leaves the
50 MiB routed prefix untouched and writes a 20 MiB shared suffix: one 5 MiB
local copy and three 5 MiB zero banks. Those zeros preserve the canonical
publication and subsequent fixed-order reduction contract; this repair does
not remove them or change communication volume.

The old vector expression selected between two lvalues:

```cpp
const float4 zero = make_float4(0, 0, 0, 0);
destination4[index] = copy_local ? source4[index] : zero;
```

HIP materialized `zero` in private memory, selected a private/global address and
issued a flat vector load. The replacement uses a uniform `copy_local` branch
with either a direct vector copy or a direct register-zero store. The scalar
tail path, all offsets and launch geometry remain unchanged.

## Compiled and native evidence

| Target | Original → repaired compiler resource | Qualification |
|---|---|---|
| gfx906 | private32→0 B/thread; VGPR9→8; SGPR33→30 | Native bytes, paired events, isolated ROCprof |
| gfx1100 | private32→0 B/thread; VGPR8/SGPR27 unchanged | Native bytes on the 7900 XTX |
| sm80/86/89 | registers18→18; stack/local0 | All compiled; sm86 native bytes/events |
| sm90 | registers18→22; stack/local0 | Compiled only; hardware absent |

All shipped targets compile with the mandatory spill guard. AMD has zero
SGPR/VGPR spills and zero LDS; the removed private storage was an addressable
C++ temporary, **not a register spill**. AMD ISA has direct zero/copy vector
stores and no remaining private-base selection, scratch zero store or flat
load in this entry. Required global writes remain intact.

The final sm90 object retains a compiler-owned 1024-byte shared allocation,
present before and after the repair, without source shared storage or
shared-memory/barrier instructions. No sm90 runtime qualification is claimed.
CUDA profiler attachment remains paused for the previously recorded driver
assertion; native events and final linked ISA/resource inspection are available.

Release component timing uses M512/H2560/top10/four banks, 32 complete
publications per retained graph, eight warmups and 31 alternating ABBA/BAAB
rounds: 62 samples per variant. Buffers retain repeated addresses; a 20 MiB
output is not described as wholly resident in cache.

| Native component | Original median, µs | Repaired median, µs | Speedup |
|---|---:|---:|---:|
| MI50 |100.8455|32.9675|3.059×|
| RTX 3090 |31.2320|31.2320|1.000×|

These isolated clocks are not substituted for full-model publication costs.
The original full-model trace was approximately 1 ms/call; its broader working
set and runtime context are not reproduced by a repeated-buffer microbenchmark.
The separate full-model comparison supplies the user-visible gain.

The two isolated ROCprof runs each contain exactly 1,248 publisher dispatches
and one excluded setup copy. Their unchanged native grid is `(3,2048)`, block256;
runtime scratch falls32→0 bytes, rounded VGPR12→8 and SGPR48→32. The original
test helper matches all 176 retained old production ISA instruction sites.
The 20 MiB required-write/event-time ratio improves208→636 GB/s; this is a
logical useful-byte rate, not a measured HBM bandwidth counter.

## Regressions and diagnostics

`V2_Integration_{CUDA,ROCm}_MoESharedRankBankPrefillReplay` explicitly belongs to
ProductionTestPreflight. Each runs the new original-body captured comparison
and the existing full-facade canonical publication/finalization regression.
The new fixture covers:

- Every local bank at M512 and adjacent 511/513-row buckets; scalar/vector tails.
- Full, partial, empty, small, reset, negative and over-capacity live counts.
- Complete guarded outputs, untouched routed prefixes, unchanged source bytes,
  and bit-preserving ±0, subnormal, Inf, NaN and signaling-NaN copies.
- Retained real production launches, fixed pointers/geometry and no recapture.
- Original and repaired compiled resource observations, including zero repaired
  private storage. Frozen old bodies are test-only and never production choices.

Both aliases pass; the new fixture also passes on explicitly selected gfx1100
without hiding/reindexing devices. Release paired-event fixtures and isolated
ROCm original/repaired profiles pass with clean driver retirement. One initial
gfx1100 invocation rejected a missing session directory before native startup;
creating the required directory fixed that test invocation, not production code.

The same checkpoint retains diagnostic repairs found while preparing numerical
kernel investigation: HC write-projection logits no longer masquerade as the
completed residual, and 4H branch-normalization output has its own replicated
snapshot role. Exact diagnostic capture keys are frozen before parity setup
and do not add numerical-reference obligations. Full model-parity-definition
and snapshot Unit owners plus their three explicit diagnostic preflight aliases
pass. These changes were already in both prefill baseline and candidate binaries.
No KL limit, numerical policy or accepted canary threshold changes here.

Full production preflight was deliberately not repeated under the user's
focused-iteration direction. The model benchmark's diagnostic status remains
non-certifying; one matching output token is not a complete model parity suite.

## Deferred work and next experiment

The preceding HC residual-write/branch-norm fusion reached byte-exact native
component speedups of 9.36% on MI50 and 7.88% on RTX 3090. It was not integrated
or promoted: its source and evidence are archived, and production was restored
before this prefill baseline. Partial graph wiring is also archived and absent
from this checkpoint.

The next prefill target is HC-down M512/N320/K10240. All 388 corresponding
dispatches in the original measured trace have zero same-GPU overlap; they
consume about 20 ms/GPU. Existing NT64/MT16 and NT64/MT32 candidates can be
compared with the current NT128/MT32 launch without changing Q8 arithmetic.
Use the existing exact-overlay owner, all three native-Q8 source aliases and
one physical shape. An M448 winner does not qualify M512. No new tile policy
is installed by this publication repair.

## Local evidence

- `/tmp/qfn-ep4ep4-prefill-pass-20261112`: baseline invocation, source/binary
  identity, benchmark, resources and retirement receipts.
- `/tmp/qfn-ep4ep4-prefill-candidate-20261112`: matched candidate equivalents,
  exact phase/protocol comparisons and complete compiled source snapshot.
- `/tmp/qfn-ep4ep4-prefill-attribution-20261112`: selected original dispatches,
  occurrence proof, overlap analysis and timeline.
- `/tmp/qfn-shared-rank-bank-component-20261112`: paired event samples and
  separate original/repaired ROCprof outputs.
- `/tmp/qfn-shared-rank-bank-isa-audit-20261112.tar.gz`: original/new code
  objects, native ISA, resources and negative controls for all shipped targets.
- `/tmp/qfn-shared-rank-bank-profile-reduction-20261112.tar.gz`: exact isolated
  dispatches, resources and original-helper ISA equivalence.
- `/tmp/qfn-full-preflight-after-cpu-next/q150`–`q165`: build/test/run receipts;
  q152 covers diagnostic tests, q157 native aliases, q158 events, q160 gfx1100,
  q161/q162 separate profiles and q163 matched production measurement;
  q164/q165 refresh the final explicit test registration and diagnostic checks.
- `/tmp/qfn-ep4ep4-prefill-dense-q8-map-20261112`: next-shape source map,
  zero-overlap evidence and bounded existing-owner command plan.
- `/tmp/qfn-hc-write-norm-component-20261112` and
  `/tmp/qfn-hc-write-next-norm-partial-graph-20261112.patch`: deferred HC work.

Generated measurements and corpora are not source-repository payloads.
