# QFN unsigned CPU dots and independent ngram frequency lanes

This note separates component qualification from the matched Release model
comparison. The source baseline is `31a5cbb51`. The final scoped checkpoint
reaches 303.365 prefill / 39.901 benchmark decode tok/s (39.745 after-prefill),
versus its fresh original control at 296.123 / 40.189. All 768 generated IDs,
MTP work and initial placements match. This is a 2.45% prefill improvement and
0.72% decode reduction, not achievement of the active 500 / 40 target on two
MI50 continuation GPUs and two CPU socket ranks with tensor-sharded experts.
Full preflight and OpenCode qualification remain deferred.

## CPU identity-nibble dot

The ordered two-row AVX-512 implementation can feed native Q4_1/Q4_K decoded
weights, which are already unsigned integers in [0,15], directly to VNNI.
The other operand remains the original signed Q8 activation bytes. This removes
the activation sign-bias XOR and its integer compensation without changing
scales, minima, floating-point products, partition order or output layout.
Signed Q4_0 and nonlinear lookup-table formats retain their existing arithmetic.

Inspection of the compiled production perf owner finds 32 VNNI instructions
in the selected block before and after the change. It removes 16 XORs, two
compensation loads/extensions, two shifts and four integer subtractions. The
selected loop has no vector stack spills or calls. Floating-point operation
counts and reduction tree are unchanged; these facts alone do not prove
runtime performance or final model outputs.

Fourteen existing CPU integration registrations pass on both AVX2 and AVX-512
runtime routes in the selected Release build. Coverage includes
21 source formats, both numerical policies, owned/workspace banks, signed Q8
extremes, N tails and large/small/repeated row execution. Actual Auto dispatch
has a separate byte oracle; a forced pair policy does not qualify Auto.

Initial unprofiled measurements use the existing complete expert-FFN owner,
28 physical workers, an authored mixed E84/R3208 distribution, N320/K2560
gate/up, Q5_1 down and installed Auto scheduling. They are component evidence,
not recorded model routes. The first candidate reduces complete persistent
FFN time by 5.85% for Q4_K and 5.46% for Q4_1 versus its first original control.
A second original control and candidate repeat retain the gain after the
operand-order correction below: Q4_K complete FFN time falls 4.95%, and Q4_1
falls 5.42%. Their gate/up reductions are 5.02% and 4.03%. The unchanged-format
controls move by -1.05% (Q4_0) and -2.98% (Q5_1); retain that variation when
interpreting small changes. The matched model results below hold promotion
despite these component gains.

An added adversarial raw-scale check found a pre-existing NaN payload mismatch
in both original CPU ISA paths. For the first Q4_1 K512 failure, grouped output
is `0x7fc56000` (the last K-block payload), while serial output is `0x7fc02000`
(the first payload). Signed zero, infinity and finite cases pass. The FP16
decoder explicitly preserves payload bits, but that does not itself specify
arithmetic operand precedence. This is raw prequantized-interface coverage;
the canonical inference quantizer produces finite scales. The correction pins
the source order of the existing AVX2/AVX-512 additions with one ordinary
VADDPS instruction, including the existing ascending partition folds. It adds
no NaN checks, storage, mode or alternative execution path. BackendNative FMA
arithmetic remains unchanged. All 14 focused registrations, including the
original failures, pass without skips or relaxed byte comparisons.

The final compiled pair preserves the original floating-point operation counts
and has no hot vector spills or helper calls. Separate isolated gate/up profiles
show 22.08% fewer retired instructions for Q4_K and 21.02% fewer for Q4_1.
Those one-call counters explain the removed work; their clocks are not the
unprofiled economy result. Q4_1 IPC decreases, so this is not a universal IPC
or cache-efficiency claim.

## Ngram frequency-lane program: initial v5 evidence

PlanFree has four independent count-min frequency lanes. The candidate assigns
one logical thread to each lane in one CTA, preserving each lane's original
collision, aging, saturation and observation order. Validation completes before
mutation. Thread zero retains the original lease, clock and free-plan
publication authority. There is no extra allocation, transfer, count readback,
recapture or transport.

Native algorithm identity 5 retains the original scalar control program when
the immutable policy has zero entries: that configuration has no frequency
work. Positive-capacity transactions use four threads, with an empty-demand
control shortcut. This qualifies PlanFree's geometry; it does not certify that
other existing phases, such as the capacity-sized observation-run producer,
already satisfy the newer actual-live-grid development rule.

Twenty focused registrations pass: two host tests and six each on MI50,
7900 XTX and RTX 3090. They check complete policy state against the serial
oracle, real captured node geometry, cancellation, source success, UInt32
boundaries and every supported native row codec. The regressions are explicitly
registered in production preflight. The full suite is deferred by user request.
Compilation covers CUDA sm80/sm86/sm89/sm90 and HIP gfx906/gfx1100 with the
mandatory spill guards enabled.

The original native replay owner also measures 26 authored transaction cases:
N16/32/48/64/1024/7168, consecutive and interleaved warm hits, changing pressure,
zero capacity and empty requests. Each has two conditioning executions, five
warmups and 31 raw event samples. State and work match the original controls;
zero-capacity timing is the existing no-source cancellation trajectory.
These diagnostic sources do not measure RAM/SSD transport or model-token traces.

At N7168, paired complete-transaction changes are:

| Device | Warm interleaved | Changing pressure | Zero capacity |
|---|---:|---:|---:|
| MI50 | -0.47%, -1.28% | -12.75%, -12.90% | -0.16%, +0.03% |
| 7900 XTX | -2.42%, -3.97% | -20.63%, -23.05% | +1.99%, +1.59% |
| RTX 3090 | -24.41%, -24.38% | -62.21%, -62.18% | approximately flat |

MI50 warm consecutive requests are approximately flat; XTX warm consecutive
requests show +4.52% and +2.01% in these pairs. Keep those results visible.
They do not support a universal warm-hit speedup.

The isolated MI50 profile correlates exactly the eighth actual PlanFree
dispatch with the first measured replay after conditioning and warmup. It
reports one wave, 1,255,840 VALU instructions, 863,119 SALU instructions,
32,304 TCC hits and 2,100 misses for that authored case. XTX trace, state,
identity and launch geometry pass, but its ALU/cache counters return zeros;
those counters are explicitly unqualified and do not demonstrate efficiency.
Profiler intervals remain diagnostic, separate from the event timing above.

Final v5 ISA inspection reports 22 VGPR / 100 SGPR on gfx906 and 26 / 106 on
gfx1100, with 20 bytes LDS and no private memory, VGPR spills, scratch
instructions or hardware barriers. The original compiler's scalar-to-vector
register-lane saves remain distinct from memory spills. Native allocation
counts round up these register quantities; neither those counts nor the
reported active-block capacity is measured achieved occupancy.

### v6 correction and focused qualification

The v5 native failure control leaves a stale Success header with sequence 0
and 48 payload bytes after a current transaction fails. The original scalar
implementation passes the same captured regression. Version 6 restores fresh
Invariant headers with the current lease and zero payload bytes, including the
original preparation-failure publication boundary. Failed policy remains
poisoned through cancellation and attempted retained replay. It also removes
the redundant read-only frequency scan; each lane updates its own cells once,
and the sole control writer publishes only after the lane writers join.

All 23 focused registrations pass: two host tests and seven each on RTX 3090,
MI50 and 7900 XTX. The new fatal-epoch aliases are explicitly in production
preflight. CUDA sm80/sm86/sm89/sm90 and HIP gfx906/gfx1100 compile with the
mandatory spill guard. Full preflight remains deferred.

Both complete 26-case native timing repetitions preserve the original input,
state and work trajectories. At 7,168 live IDs, complete transaction latency
changes versus the retained scalar controls are:

| Device | Warm interleaved | Changing pressure / aging 1 | Zero capacity |
|---|---:|---:|---:|
| MI50 | -9.99%, -9.39% | -39.00%, -39.14% | -0.27%, -0.17% |
| 7900 XTX | -12.89%, -13.65% | -47.26%, -49.04% | +3.08%, +2.17% |
| RTX 3090 | -34.14%, -34.13% | -72.87%, -72.87% | -0.17%, 0.00% |

These remain authored component transactions, not RAM/SSD transfer or model
performance measurements. Warm consecutive XTX requests move +3.18% and
+0.74%; the unchanged scalar zero-capacity program also varies. All cells,
including these regressions, remain in `v6-economy-comparison.json`.

The isolated MI50 profile records 711,062 VALU and 489,735 SALU instructions,
down from 1,255,840 and 863,119 in v5 at the same actual dispatch. It retains one
wave, with 30,503 TCC hits and 2,082 misses. These are separate diagnostic
counters, not canonical timing labels. Final v6 code uses 22 VGPR / 100 SGPR on
gfx906 and 25 / 106 on gfx1100, 20 bytes LDS, and no private memory, VGPR spills,
scratch instructions or hardware barriers. Scalar-to-vector lane saves remain
34 and 20 respectively. Fresh XTX hardware counters were not collected; its
earlier inactive-counter limitation remains, alongside the CUDA profiler limit.

## Evidence and remaining decision

### Matched model progression and scoped checkpoint

Fresh passive Release runs preserve all 768 generated token IDs, MTP work,
runtime settings and initial expert placement on both ranks. The original CPU
and scalar ngram control retains the common coordinator refactor, so it is not
an untouched checkout of the baseline commit.

| Variant | Prefill tok/s | Benchmark decode tok/s | After-prefill decode tok/s |
|---|---:|---:|---:|
| Fresh original CPU / scalar ngram control | 296.123491 | 40.189430 | 40.032440 |
| CPU pair + ordered adds + ngram v5 | 295.853925 | 39.497211 | 39.342925 |
| Combined candidate, fresh repeat | 297.203365 | 39.013014 | 38.860620 |
| CPU candidate / original scalar ngram | 295.355601 | 39.242049 | 39.088759 |
| Add unsigned single-row CPU path / ngram v5 | 297.547819 | 39.979440 | 39.823270 |
| Plain single-row scales + EVEX constraint / ngram v6 | 303.364723 | 39.900628 | 39.744767 |

Prefill is approximately unchanged. Decode loses 1.72–2.93%; the CPU-only
isolation retains a 2.36% loss. The ngram program therefore does not explain
the main regression. The initial pair/v5 candidates are not model improvements; the final row
above supplies the scoped prefill gain, with its decode trade-off retained.
All native processes and observers retired, and the run-owned prefix archives
were audited through metadata and reclaimed without reading payloads.

Retained diagnostic dispatch records show that many MTP expert projections
execute a physical single-row body. The next CPU candidate extends the exact
unsigned-identity integer simplification to that body. An independent witness
reconstructs the original biased integer dot, avoiding a circular comparison
between two changed single-row implementations. It passes against the original
body on both runtime ISAs. The candidate then passes all 14 focused CPU
registrations, including both runtime ISAs and 21 source formats. Fresh model
output/MTP/placement still match; decode recovers to within 0.52% of the fresh
control. That is a recovery from the preceding candidate, not a demonstrated
improvement over the original model control.

Two authored component cohorts use the original complete FFN owner, 28 physical
workers, E24/R24 (all single-row decode) and E24/R30 (18 single-row and six paired
MTP experts), Q4 gate/up and Q5_1 down. Q4_K complete FFN latency falls 4.35–4.90%
in the decode cohort and 3.43–3.49% in MTP versus the matched older single-row
body. The unchanged-format controls also move materially, so these samples do
not independently establish causality or replace the full-model measurement.

Actual-object inspection confirms the same 32 VNNI dots, 16 FP multiplications
and eight FP additions in the selected fused single-row K loop. Compensation
work is removed, but the loop still has pre-existing helper calls and vector
stack traffic: static vector stack references decrease from 30 to 22. Four
calls per block load effective scales through a generic source-factor helper.
The ordinary identity dispatcher has no vector stack operands. A separate
rounding-boundary constraint also restricts intermediates to the lower 16
registers. These are specific next tuning targets; this revision does not
claim a spill-free fused single-row implementation.

The final CPU candidate uses the existing plain FP16 widening for decoder
kinds whose prepared layout proves ordinary scales. LUT/source-factor formats
keep their generic equation. It also allows all 32 EVEX registers at the
existing rounded-value boundary. All 14 focused CPU regressions pass again.
Actual compiled common M1 loops remove all four scale-loader calls, retain
32 dots / 16 FP multiplications / eight FP additions and four ascending
partition adds, and reduce static vector stack references from 22 to 16.
Remaining spills cross the activation-scale conversion branch; they are not
claimed eliminated. The ordinary identity body has no vector stack references.

The first bounded component comparison shows Q4_K gate/up reductions of
6–10%, but only about 1% for its complete FFN. Q5_1's complete FFN is about 9%
slower than the immediately preceding unpromoted M1 component control despite
roughly flat isolated projections. That variation is unresolved; it prevents
claiming a universal all-format performance improvement. The final full-model
row is the production workload evidence for this scoped checkpoint.

The original result-publication owner also passes its default control and a
separate sampled-IP diagnostic (47,882 samples, none lost). About 52.26% of
sampled cycles are in vec_add, 26.76% in the combine/weighting worker and about
17% in OpenMP wait work. Serial metadata validation is not the main sampled
cost. This authored cached-PMA fixture does not establish mapped-ticket store
cost, NUMA page placement or model-stage savings. Its next candidate fuses only
the final addition and weighting while retaining both required outputs; that
candidate is not part of this checkpoint.

Original artifacts are outside the repository:

- `/tmp/qfn-cpu-unsigned-nibble-pair-next/`: frozen CPU candidate, finite
  regressions, original/candidate ISA, economy samples and nonfinite diagnosis.
- `/tmp/qfn-ngram-four-lanes-r1150/`: all original/final native runs,
  `v5-focused/focused-results.json` and `v5-economy-comparison.json`.
- `/tmp/qfn-pair-ngram-model-r1151/`: coherent original-source control,
  isolated CPU profiles and fresh model admission/benchmark work.
- `/tmp/qfn-cpu-ordered-add-contract-next/`: original-failure correction,
  14 passing registrations and final actual-object inspection.
- `/tmp/qfn-ngram-rocprof-isolated-r1152/`: pinned-SDK single-dispatch
  collection, strict offline correlation and final v5 ISA extraction.
- `/tmp/qfn-ngram-v6-controls-r1153/`: failing v5 / passing scalar controls,
  corrected native gates and v6 profile/resource results. Original raw timing
  and profile records remain under `/tmp/qfn-ngram-four-lanes-r1150/`.
- `/tmp/qfn-cpu-linear-scales-evex-r1154/`: final CPU build, 14 passing
  registrations and first/steady-loop ISA. Its model run is
  `/tmp/qfn-pair-ngram-model-r1151/linear-scales-evex-v6-passive/`.
- `/tmp/qfn-cpu-rooted-publication-profile-next/`: original default-owner
  validation and separate sampled-IP evidence.
- `/tmp/qfn-cpu-unsigned-nibble-one-row-next/`: independent original M1 witness,
  14 passing candidate registrations, actual-object inspection and bounded
  decode/MTP component controls.

The first r1151 model control omitted r1141's passive PerfStats filter. It
matches all 768 output IDs and MTP work, but collects additional kernel
observations and is excluded from the throughput comparison. Its original
results remain intact. Subsequent timing uses the exact same passive filter
as r1141; the instrumented clocks are not relabeled as a slowdown or speedup.

The model comparison retains the public MPI bootstrap, 512 prompt / 256 decode
tokens, one warmup and three measurements, seed42, native sampling defaults,
dynamic MTP, dynamic expert balancing, context262144, RAM ngrams and the same
GPU cache and prefix-cache budgets. Every generated token, MTP workload and
placement must match. Profiling is separate from unprofiled timings. Native
processes, MPI sessions and resource observers must retire before another
build or timing run begins; run-owned prefix archives are reclaimed afterwards.

The prior full-model profile attributes about 25 ms of prefill per GPU to
PlanFree. Even removing it entirely would account for only about 1.3% of that
profiled prefill span. CPU expert-result readiness remains the larger target;
component improvements here must not be presented as closing the 500 tok/s gap.
