# QFN tuning dashboard

**Checkpoint190 installs the qualified ROCm ValueMajor default:** the final
override-free repeat measures **820.777 prefill / 41.562 decode tok/s**. Its
matched explicit-policy A/B/A measured **820.871 / 41.537**, versus
bracketed frozen 180 controls at **818.918 / 40.906**. Decode improves **1.542%**
in that A/B/A with non-overlapping measured ranges; prefill is preserved.
All nine A/B/A natural 256-token trajectories and all three default repeat
trajectories match. All 185 canonical final-logit rows remain byte-identical.
The fresh **873-entry Unit/affected gate** passes, including mixed CUDA/ROCm
RAM/disk prefix restart. Actual owner telemetry proves default selection on all
four MI50s with both layout overrides absent. **Castagna's 42.959 decode tok/s
target remains open.** This is focused
qualification, not complete preflight, whole-corpus or image certification.

The [GDN qualification record](QFN_GDN_RECURRENCE_TUNING.md) retains the exact
settings, per-arm scores, current source/build evidence and archived experiments.
The complete production convolution/recurrence boundary improves **15.3450 to
11.0900 µs (27.73% less time)**. The transposed persistent state supports prefill,
serial/grouped decode, snapshots, linked state and prefix archives without a live
transpose or host mirror. The native ROCm sweep covers all 22 new variants,
354 cases and 2,832 captured replays, with zero private storage/spills. CPU and
CUDA defaults remain key-major. Prepared Q/K publication is a separate later
candidate; the retired private kernels are removed from production source.

The [short-context attention comparison](QFN_SHORT_CONTEXT_ATTENTION_ATTRIBUTION.md)
finds that the large apparent per-head deficit was mainly local Q6 versus Q12
batching. With matched Q12 inputs and the new nested Fold12 merge, the residual
is **1.334% at 512 and 0.424% at 768**. Both use compact KV rows, shared KV panels and
two kernel launches. Extra empty-header/control work is present, but its latency
share is unproven; the 768 residual is comparable to measured run variation.
Fold12 reduces the matched complete boundary by 2.067% at 768 and preserves exact
bytes. Long-context checks retain the measured +0.411% pre-rotary 64K cost; no
whole-model gain is attributed to attention alone.

The prior [checkpoint180 qualification](QFN_SCALAR_COLUMN_PUBLICATION_TUNING.md)
remains retained at 819.594 / 41.030 tok/s, with its 823-entry Unit/affected gate.
Its [fresh stage attribution](QFN_CASTAGNA_STAGE_COMPARISON.md#fresh-checkpoint180-attribution-2026-10-24)
ranked GDN as the largest comparable deficit; it precedes 190 and must not be
presented as current stage timing. The [GDN structure map](QFN_GDN_STRUCTURE_COMPARISON.md)
explains which Castagna layout lessons preserve Llaminar's reduction order.
The routed SwiGLU181 and HC publication182 component screens remain archived
separately; neither changes the qualified model score. Complete preflight
remains G60 and the explicit FP16 attention policy is unchanged.

**Checkpoint179 passes scoped qualification:** **819.648 prefill / 40.656
decode tok/s**, versus bracketed frozen177 controls at 819.665 / 40.295.
Decode improves **0.897%** with non-overlapping sample ranges; prefill is flat.
All nine generated sequences and all 185 canonical final-logit rows match.
The complete **798-entry Unit/affected gate**, both Release counterparts,
5,354 ROCm / 2,354 CUDA captured cells, final resource checks and isolated
ROCm profiles pass. The short model trace proves 48 scan calls in every
model graph. See the [settings and qualification record](QFN_SMALL_GROUP_PREFIX_TUNING.md).
The **42.959 tok/s Castagna target remains open**. This uses explicit FP16
attention Prefill8/Decode33; default promotion and complete preflight remain
separate work. The [existing-quantizer screen](QFN_ACTIVATION_PREPARATION_TUNING.md)
finds a 1.72% complete HC-down projection gain for V2 with exact bytes and
18.9% fewer vector instruction issues. It is a synthetic one-shape result;
defaults and model scores remain unchanged. The same record now includes a
private GDN normalization/SingleQ8 fusion: **15.325→13.2925 µs (13.26%)** for
the complete synthetic native19/M1 projection boundary, exact bytes, no spills
and **38.8% fewer vector instruction issues** in preparation. Its expanded
122-case row/sums matrix and12 recorded-value cases pass; the prototype is
archived separately. Fixed-M1 MoE column publication is qualified by180 above;
the component results do not replace that measured model score.

**Previous checkpoint177 passes the focused production-integration slice:** **819.876
prefill / 40.328 decode tok/s**, versus frozen175 controls at 820.235 / 40.151.
The measured +0.439% decode gain is small and the ranges narrowly overlap;
the **42.959 tok/s Castagna target remains open**. All nine natural token
sequences and all 185 canonical final-logit rows match the accepted control.
The 804-entry gate passes the complete 778 Unit registrations and affected GDN
regressions. This is focused qualification, not complete ProductionTestPreflight.

Direct merged-input GDN decode now removes its rearrangement launch on CUDA
and ROCm while retaining exact recurrence arithmetic and state ownership.
The final complete-boundary gains are 16.45% on MI50 and 26.05–30.09% on RTX3090;
compiler/resource checks report zero spills. See the
[qualification and benchmark settings](QFN_GDN_RECURRENCE_TUNING.md#scope-and-current-disposition).
The [NativeQ8 fusion trial 178](QFN_NATIVE_Q8_FOLD_TUNING.md) is rejected:
all four exact-byte variants increase complete projection latency by 30–34%.
ROCm counters show about 52% more vector instructions despite essentially
unchanged fetched bytes. The private kernels are removed.
The [small-group prefix checkpoint179](QFN_SMALL_GROUP_PREFIX_TUNING.md) measures
**13.270→4.403 µs** for complete ten-route grouping on MI50, with exact bytes,
88.2% fewer vector instruction issues and no spills. Final production builds
and the 5,354-cell ROCm / 2,354-cell CUDA captured sweeps pass. Repairs for
fixture-contract failures and unnamed rebalance-command padding now pass the
complete **798-entry Unit/affected gate**, including both broad MoE suites and
paired Release checks. Fresh production events and eight isolated ROCm counter
runs also pass. Its qualified model accuracy and throughput are recorded above.

**Previous candidate175 passes scoped regressions and the selected Release accuracy
panel, but model decode throughput is unchanged.** The matched169/175 A/B/A
measures **820.229 prefill /40.170 decode tok/s** versus818.657 /40.221.
All nine natural256-token sequences match. All185 final-logit rows across
26 files remain byte-identical to151, including the ranking threshold and reset.
Explicit Decode now omits unused indexer query projection
and preparation below visible context2052 while retaining keys, pools and
history. The final repair run passes778 Unit registrations and six CUDA
registrations plus the model fixture;147 unchanged Integration results are
retained with source/binary/runtime checks. The resulting932-entry evidence is
a bounded two-source composition, not complete ProductionTestPreflight.
See [the qualification record](QFN_ATTENTION_STRUCTURE_COMPARISON.md#demand-limited-indexer-queries175).
Isolated query projection falls46.006→6.191µs at context512; rocprof confirms
2570.313→0.188KiB fetched with zero scratch/spills. Active projection remains
approximately46µs; active preparation adds approximately1.58µs. These synthetic
component results do not establish a model gain. The fresh complete-model trace
confirms the intended interval kernels execute: measured query projection
falls54.453→10.386µs/call versus165, with unchanged key projection. The model
result remains flat; GDN preprocessing/recurrence is the next bounded target.
A separate host uprobe observation measures only202–241µs median per model
`hipGraphLaunch`, compared with approximately25ms per unprofiled decode step.
All four participants and three measured token sequences validate. The long
host calls in rocprof are instrumentation effects, not ordinary submission cost.
The [GDN follow-up](QFN_GDN_RECURRENCE_TUNING.md#scope-and-current-disposition)
rejects the176 preparation variants for single-token decode. Direct merged-input
candidate177 removes deinterleave. Frozen175 A/B/A confirms a complete M1
component reduction from14.529075 to12.10125µs (16.71%). All261 extended
geometry cells and12 cells of repeated in-place state updates pass exact-byte
checks. Profiles show zero spills and LDS bank conflicts. The final production
integration and matched model results are recorded above and in the GDN report.
The approximately43tok/s decode target remains open. CUDA full-work interval
overhead has now been measured separately; NCU counter collection is rejected
after driver assertions, with further attachment stopped. See the linked record
for the accepted unprofiled measurements and diagnostic limits.

**Attention checkpoint169 is green:** **819.319 prefill /40.175 decode tok/s**,
versus frozen165 controls'818.872 /40.112. The +0.055% /+0.159% differences
have overlapping ranges: whole-model throughput is effectively unchanged.
All nine natural256-token sequences are identical.
The ordered per-output denominator removes HIP33 broadcast banks without changing
FP32 chains/division. All eight component cells beat165: **14.745µs at M1/512,
18.521µs at M1/64K and81.506µs at M16/64K**. All21 native tests, the retained
wide-counter regression and eight independent numerical exports pass. Canonical
model groups/phase counters pass;26 final-logit files/185 rows are byte-identical
to151. **All832 Unit/affected entries and matched165 A/B/A pass.**

The [169 qualification and benchmark settings](QFN_ATTENTION_STRUCTURE_COMPARISON.md#ordered-per-output-normalization169)
record124/124 VGPRs,708B LDS, zero scratch/spills and exact default8 bodies.
Lower-register168/170 and counter-tree171/172 lose economy and are rejected.
VALU utilization is active-lane utilization, not achieved occupancy. No169
whole-model throughput gain is claimed.165 remains the preceding qualified
checkpoint below; defaults/full preflight are unchanged and43tok/s remains open.

Follow-up probes173/174 are rejected: distributing merge coordinates across
more CTAs and shortening counter register lifetimes preserve bytes but do not
deliver consistent timing gains. The useful gate-coordinate regression remains;
the restored169 kernel passes all21 native tests and a fresh832-entry Unit/
affected gate. Its captured attention ELF matches the qualified bytes exactly.
See [the rejected probes and retained regression](QFN_ATTENTION_STRUCTURE_COMPARISON.md#rejected-merge-scheduling-probes-173174).

**Attention checkpoint165 is green:** **818.073 prefill /40.190 decode tok/s**,
versus frozen151 controls'818.437 /40.151. The −0.045% /+0.095% differences
have overlapping ranges: whole-model throughput is effectively unchanged.
All nine timed sequences,185 final-logit rows,20 native tests, eight independent
numerical exports and832 Unit/affected registrations pass.

Retaining original key-row addresses reduces the native pair from19.444 to
**19.127µs at64K** and84.354 to**82.285µs for M16/64K**. The512 pair is
15.054µs. Registers remain63 used/64 allocated with zero spills; the wider
QK probe166 is rejected for grouped slowdown. The
[qualification and fresh profiles](QFN_ATTENTION_STRUCTURE_COMPARISON.md#retained-key-rows165-and-refreshed-paired-profiles)
retain all settings, timings, numeric comparisons and source/ISA evidence.

Fresh full-model rocprof traces pass for both engines. Recorded attention-core
work is **0.386ms/GPU/decode step versus Castagna's0.163ms**, down from the
earlier0.582ms Llaminar trace. This is instrumented work, not request latency;
H6×12 versus H12×6 ownership remains explicit. Per-call producer means are
22.64 versus21.62µs, but merge means are **9.56 versus5.57µs**. The next tuning
target is the ordered merge. The fork's14.424µs short-context native pair and
approximately43tok/s whole-model target remain open. Defaults and complete
preflightG60 are unchanged; this remains the explicit FP16 phase8/33 experiment.

**Previous attention checkpoint164:** **819.682 prefill /40.265 decode tok/s**,
versus frozen151's819.102 /40.186. The +0.071% /+0.196% differences have
overlapping ranges: whole-model throughput is effectively unchanged.
All nine timed sequences and185 final-logit rows match. The expanded20 native
tests, eight independent numerical exports, and832 Unit/affected registrations
pass. The new analytic nonzero-K1 regression also passes through its explicitly
updated production-preflight registration after the complete suite.

K1/V0/V1 load overlap and separate64/32 QK scheduling windows improve the native
pair over checkpoint160 from15.446 to**15.086µs at512**,20.179 to**19.444µs at64K**,
and88.597 to**84.354µs for M16/64K**. Source-bound ISA proves the original128
ordered dot products and all three overlap windows, with63 used/64 allocated
VGPRs and zero spills. Two faster-M1 trials were rejected for grouped slowdown.
The [qualification record](QFN_ATTENTION_STRUCTURE_COMPARISON.md#k1-overlap-and-split-qk-scheduling164)
retains every probe, isolated rocprof counters, independent byte comparisons,
new oracle, model metrics and matched benchmark settings. The fork's14.424µs
short-context pair and42.959tok/s model target remain open. This is an explicit
FP16 phase8/33 experiment; defaults and complete preflightG60 are unchanged.
The subsequent165 qualification above retains key-row address work.

**Earlier phase-schedule checkpoint151 is qualified** with separate prefill/decode
programs: **818.579 prefill / 40.110 decode tok/s**, versus the matched uniform8
control's **818.788 / 39.884**. Decode improves **0.566%**; prefill is effectively
unchanged. All nine timed token sequences match, the required model accuracy
panels are unchanged, and the nine boundary-diagnostic decode rows improve mean
KL from0.000368 to0.000210. Complete Unit plus affected registrations pass
**832/832**; all19 Release native tests and eight numerical exports pass.
The [phase qualification record](QFN_ATTENTION_STRUCTURE_COMPARISON.md#phase-schedule151-focused-model-qualification)
records settings, top-1/top-5 and limitations. This remains an explicit FP16
experiment; production defaults and complete preflightG60 are unchanged.
The smaller-head-group candidate152 is byte-exact but rejected for speed:
20.672µs versus16.503µs at context512. Fresh isolated rocprof profiles of both
engines show extra producer instruction work, with no spills or measured LDS
bank conflicts. The [profile comparison](QFN_ATTENTION_STRUCTURE_COMPARISON.md#fresh-rocprof-comparison-and-rejected-two-head-schedule152)
records actual head geometries and resource limits. Castagna's42.959tok/s target
remains open.

The subsequent copy-address and V0-prefetch candidates155/157 preserve all185
qualified final-logit rows and pass all19 focused native tests. Candidate157
reduces the measured post-RoPE attention pair to **15.878µs at512** and
**21.481µs at64K**; the pinned fork's512 pair is **14.424µs**. Isolated rocprof
and exact ISA show zero spills and partly overlapped V loads; premature waits
remain a tuning target. These component results do not replace151's measured
whole-model throughput or its complete Unit/affected qualification. See the
[profile and candidate record](QFN_ATTENTION_STRUCTURE_COMPARISON.md#fresh-rocprof-comparison-and-rejected-two-head-schedule152).

**L106 split load scheduling is qualified** at **815.992 prefill / 39.274 decode
tok/s** on A1. Its matched immutable HC103 control measures **815.330 / 37.520**:
**+0.081% prefill / +4.676% decode**. Decode ranges do not overlap; prefill is
effectively unchanged. All **776 Unit tests**, **six focused registrations**,
**185 byte-identical final-logit rows** and **twelve identical generated
sequences** pass. The immutable CTA/K geometry selects scalar, four-load or
eight-load schedules while retaining the original FP32 dot/reduction order.
The decode HC component is **2.242× faster**, with M512 unchanged. Exact native
profiles and gfx906 ISA show zero spills/private storage for every schedule.

The [matched comparison](../../../../parity-results/qwen-flash-next/injection106-split-model-comparison.json)
and [arithmetic, dispatch and profile evidence](QFN_FLOATING_REDUCTION_TUNING.md#l106-overlap-loads-while-preserving-each-original-dot-product)
record the workload and every sample. The **42.959 tok/s** Castagna decode
target remains open. The [subsequent attention and selector experiments](QFN_PACKED_ATTENTION_TUNING.md)
rejected the packed FP16 implementation for speed. The retained bounded
selector improves the measured ROCm ranking work; CUDA keeps its prior default
because the bounded winner depends on score distribution. Cheaper observation
counting saves 1.52–3.97% in the measured MI50 attention component, while the
matched model run remains effectively unchanged at **814.979 / 39.208 tok/s**
versus **815.694 / 39.239** for its controls. All nine generated sequences and
185 final-logit rows remain exact; the cleaned tree passes **776 Unit plus45
focused Integration tests**, without skips. These experiments are not yet a
new checkpoint or a refreshed full preflight. Current work targets matched
attention-stage time below Castagna, with FP16 arithmetic admitted only after
measuring final-logit KL and prefill/decode top-1 and top-5 agreement. Unused
short-context indexer queries remain a separately owned implementation.
Complete preflight remains G60.

The new [shared-tile FP16 attention candidate123](QFN_PACKED_ATTENTION_TUNING.md#same-build-final-logit-precision-comparison)
passes the unchanged required QFN numerical groups against a fresh same-build
FP32 control. Main-group mean KL changes from **0.033536 to 0.025614 prefill**
and **0.032070 to 0.028276 decode**; held-out means change from **0.009691 to
0.014911** and **0.003838 to 0.004553**. Top-1/top-5 trade-offs are retained in
the comparison table. The user accepted this accuracy trade-off. Measured
selected-RoPE attention-pair component time falls from **71.173 to 23.868 µs**
at M1/context512 and **11730.900 to 3983.325 µs** at M512/context65536. The
M1/65536 sample is still 0.67% slower. These historical component cells use
pre-RoPE keys; the model's native FP16 cache uses post-RoPE keys. The corrected
fixture now requires/reports that distinction, authenticates the actual native
specialization and tests both through grouped/serial dirty replay. Model
accuracy used the real graph and is unaffected. Removing
redundant barriers and unowned-head arithmetic in124 retains all185 FP16 logit
rows exactly and passes all825 Unit/affected registrations. The matched
same-runtime A/B/A now measures **817.871 prefill /39.754 decode tok/s** for
FP16 versus **816.092 /39.390** for FP32: **+0.218% /+0.925%**. Decode ranges
do not overlap; prefill ranges overlap. All nine natural token sequences match.
The fresh pinned Castagna repeat is **462.166 prefill /42.657 decode tok/s**,
with the same generated trajectory and passing runtime/driver checks. The
candidate retains **1.770×** its prefill rate and is **6.804% behind** in decode.
The42.959 historical target remains open. Fresh production-specialization
profiling confirms post-RoPE execution and zero private scratch. The complete
core still takes **0.582ms/GPU/decode step versus0.163ms** for Castagna, and
**12.2255 versus3.9578µs/prompt row** in prefill. These instrumented core times
exclude external wrapper work; broader stage attribution remains open.
Corrected post-RoPE components expose an11.76% M1/64K regression versus FP32;
default promotion is held while this is tuned. The
[actual source comparison](QFN_ATTENTION_STRUCTURE_COMPARISON.md) identifies
wider prefill reuse and decode loop scheduling. The QK unroll16 winner128
improves seven component cells by4.4–8.5%, preserving every output/numerical-
partial byte and passing15 native tests; the long M1 regression shrinks to1.71%.
Its exact ISA/profiles show load-ahead scheduling, unchanged fetch volume,
zero spills and zero LDS conflicts. Unroll32 was slower and is not retained.
The retained128 model panel passes all required groups and keeps all185
final-logit rows byte-identical; the refreshed825-registration Unit/device gate
passes. Its source-frozen A/B/A measures **818.838 prefill /39.987 decode tok/s**
versus **816.165 /39.283** for the FP32 controls: **+0.328% /+1.792%**. Decode
ranges do not overlap and all nine generated256-token sequences are identical.
The same prompt512/decode256/context4096 workload uses fullTP4, sharded
vocabulary,28 physical NUMA-local workers, RAM ngrams, native post-RoPE FP16 KV,
and disabled MTP/prefix cache. This is a focused checkpoint, not a refreshed
full preflight; full preflight remains G60. A packed-merge trial was exact but
slower at decode and was removed. The next candidates target smaller decode
head groups. Per user direction, prefill tuning is parked: preserve its current
lead and qualify separate prefill/decode physical regimes, including byte-exact
serial/grouped decode. Castagna attention performance remains the target.

Direct decode profiling133 now joins20 isolated producer/merge counter runs
for both engines. Our pair rises20.157→32.481µs from512→640 visible tokens,
while the fork rises14.424→15.221µs at its different H12 shape: our eight
partitions start a second tile, while the fork increases its partition count.
The [source/profile comparison](QFN_ATTENTION_STRUCTURE_COMPARISON.md#direct-decode-profiles-and-partition-bottleneck)
records exact shapes, multiplicity normalization and scope. G1/G2 sharing were
byte-exact but slower and removed; G6 has only a small gain and remains archived.
The33-logical-tile decode prototype is undergoing focused native qualification;
production and prefill still use the retained eight-stream program.
Its first 18 native tests pass. At M1, the component pair falls from
59.975 to 29.575 µs at 64K, but rises from 20.3175 to 25.24625 µs at 512;
short-context merge overhead remains unresolved. The live-metadata reducer
trial136 is byte-exact but slower and has been removed. A single-pass
producer137 reduces used registers from 75 to 48 without a meaningful timing
gain. Both candidates retain all saved 33-part output/partial bytes. These
are component findings; the new reduction program has no model-KL or
production-default qualification yet. See the
[decode qualification record](QFN_ATTENTION_STRUCTURE_COMPARISON.md#fixed-logical-decode-tiles-component-qualification).
Subsequent load/address and fixed-fold cooperative-metadata trials retain all
checked bytes and bring the33-part M1 pair to **19.070 /28.822 µs at512 /65536**,
with M16/65536 at127.910 µs. The same-build old-eight controls are20.301 /
60.085 /138.251 µs respectively. All19 native tests pass, including an
independent native signed-zero underflow regression. The two-bound merge145
loses about12% at640/768 and is removed; full-panel copy144 is byte-exact and
effectively flat despite fewer profiled vector instructions. Six-head sharing146
and full-tile PV guards147 improve this to **17.237 /25.434 µs at512 /65536**,
with M16/65536 at101.905 µs. Their19 native tests and eight numerical exports
pass; exact profiles show zero scratch and unchanged52 used/allocated registers
between146 and147. Unroll64 in148 and parallel scale calculation in150 further
reduce the pair to **16.536 /22.046 µs at512 /65536**, with M16/65536 at97.603 µs.
All19 native tests and eight saved output/partial byte comparisons still pass.
The retained producer uses51/52 VGPRs;150's merge uses70/72, with zero scratch
and zero measured LDS bank conflicts. Castagna's pinned512 pair is14.424 µs,
so a short-context component win remains open. The explicit Prefill8/Decode33
schedule now passes the focused model qualification151 above; the component
measurements remain separate from whole-model accuracy and throughput. Source review
also confirms that the fork has two active whole-KV-head owners per attention
layer versus Llaminar's four query-head shards; per-GPU work ratios are not
request-latency ratios.

The shared attention fixture now proves old/current output and partial bytes
on both GPUs at M1/M16 and 512/65536 context. The initial CUDA M16/512
regression is fixed by retaining query values across selected tokens: the
case improves from its original63.792 µs to55.634 µs. All twelve additional
M512/1024/2048 comparisons also remain byte-exact, with14–18% lower CUDA
attention time and no ROCm regression. The [GDN recurrence candidate](QFN_GDN_RECURRENCE_TUNING.md)
passes 32 geometry checks and preserves all 185 model-logit rows exactly.
It reduces recurrence time by 3.88%/5.61% at the TP4 M1/M16 shapes, with a
3.83% regression at the larger DK128/DV128/H48/M16 shape. Its matched model
benchmark is neutral at **815.850 / 39.237 tok/s** against **815.592 / 39.255**;
all nine generated sequences match. Neither experiment is a new qualified
throughput win. The refreshed gate passes all 823 CTest registrations: 776 Unit,
46 affected Integration and one setup registration. The subsequent model check
preserves all 185 final-logit rows byte for byte. The independently expanded
GDN historical test also passes normalized and scale-only policies; further
GDN kernel tuning is held while attention takes priority. Full preflight
remains G60.

**HC103's focused qualification is green** at **815.461 prefill / 37.576 decode
tok/s** on A1. The matched L104 control measures **814.723 / 37.439**:
**+0.091% / +0.367%**, with overlapping ranges. The fusion preserves all 185
final-logit rows byte for byte and twelve generated sequences; all 776 Unit
tests and the final 23 focused registrations pass. These are small descriptive
gains. The mixed-build CPU AVX2 one-worker component regression remains an
explicit limitation; separately compiled AVX2 and default socket measurements
are recorded independently. Complete production preflight remains G60.

The [matched model evidence](../../../../parity-results/qwen-flash-next/hc103-model-comparison.json)
and [qualification details](QFN_TP_COMPUTE_SCALING.md#hc-fusion-control-calibration-and-counted-row-defect)
record the activation/publication fusion, native allocation capture leases,
counted CPU row fixes and retained failures. Fresh Castagna/Llaminar stage
attribution is recorded in the [paired stage comparison](QFN_CASTAGNA_STAGE_COMPARISON.md).
Castagna's second fresh repeat is **453.233 / 42.468
tok/s**, following an earlier **465.370 / 38.192** repeat. Both are retained;
the historical **42.959 decode tok/s** target remains open. The variation is
not resolved by selecting the lower reference.

**L104 counted-row and typed-bias correctness is qualified** at **815.121 prefill /
37.460 decode tok/s** on A1. Matched L102 measures **815.181 / 37.420**;
the changes are **−0.007% / +0.106%**, with overlapping ranges. Throughput is
effectively unchanged. All **776 Unit** tests, **27 focused registrations**,
**185 byte-identical logit rows** and **twelve identical generated sequences**
pass. The fixes protect inactive rows in quantizer/output publication, admit
device-counted M1 on both GPUs, forward typed bias, and declare CUDA fused-M1
workspace. No precision or ordinary decode policy changes.

The [matched model comparison](../../../../parity-results/qwen-flash-next/counted104-model-comparison.json),
[component ABBA](../../../../parity-results/qwen-flash-next/counted104-component-abba-r1/REVIEW.md)
and [40 isolated ROCm counter runs](../../../../parity-results/qwen-flash-next/counted104-profile-summary-r1/REVIEW.md)
record the evidence. Static ISA and native resources show zero spills/scratch
and unchanged allocated registers for the measured producers. See
[qualification and retained failures](QFN_TP_COMPUTE_SCALING.md#hc-fusion-control-calibration-and-counted-row-defect).
Complete production preflight remains G60. HC activation/publication fusion is
the next private candidate; the **42.959 tok/s** Castagna decode target remains open.

**L102 is the preceding floating projection checkpoint** at **815.053 prefill /
37.424 decode tok/s** on A1. The fresh control, carrying the same cold hipBLAS
workspace fix, measures **814.156 / 37.134**: **+0.110% / +0.781%**.
Decode ranges do not overlap (control 37.014–37.210; candidate 37.360–37.504);
prefill is effectively unchanged. All 776 Unit tests, the focused native gates,
185 byte-identical logit rows and twelve identical generated sequences pass.
Custom FP32/native16 kernels now receive immutable addresses directly in their
captured arguments; genuine BLAS/half table consumers retain their tables.
[Matched comparison](../../../../parity-results/qwen-flash-next/pointer102-model-comparison.json)
and [qualification details](QFN_TP_COMPUTE_SCALING.md#floating-projections-immutable-captured-bindings-l102)
record the settings, exact arithmetic, resources and regression coverage.
Complete production preflight remains G60. Next measure the HC activation and
two-term Q8 publication fusion; the 42.959 tok/s Castagna decode target remains open.

The HC fusion's independent control calibration exposed the existing counted-row
and typed-bias defects before fusion was installed. L104's repair passes
[27 focused registrations](../../../../parity-results/qwen-flash-next/counted104-final-focused-r2/result.json)
on CPU, CUDA and ROCm. This includes all 21 quantized source formats, mixed-format
fused M1, full/partial/empty retained replays, independent typed-bias witnesses,
all six ROCm quantizer variants and generated dispatch refresh. The original
failures and intermediate test corrections remain recorded. The rebuilt
[complete Unit suite](../../../../parity-results/qwen-flash-next/counted104-unit-r3/result.json)
passes **776/776**, and the adjacent GEMM checks pass. The
[Release model panel](../../../../parity-results/qwen-flash-next/counted104-release-accuracy-r1/report.json)
passes, with [all 185 logit rows byte-identical to L102](../../../../parity-results/qwen-flash-next/counted104-release-accuracy-r1/logit-byte-comparison.json).
Matched performance qualification is complete; HC fusion remains private and
has no measured gain.

**L101 is the preceding branch-normalization checkpoint** at **814.610 prefill / 37.088
decode tok/s** on A1. Matched L100 measures **814.645 / 37.103**: **−0.004% /
−0.041%**, with overlapping ranges. Model throughput is effectively unchanged;
the isolated decode-shape normalization kernel is 2.794% faster. All 776 Unit
tests, three focused native registrations, 185 exact logit rows and twelve
identical generated sequences pass. The reduction order, arithmetic precision
and runtime settings are unchanged; native profiling confirms zero scratch.
[Matched comparison](../../../../parity-results/qwen-flash-next/branch-norm101-model-comparison.json)
and [component/profile evidence](QFN_TP_COMPUTE_SCALING.md#branch-normalization-original-reduction-order-with-fewer-barriers)
record the scope. Complete production preflight remains G60. L102 now qualifies
the subsequent custom floating pointer-table setup removal.

**L100 is the preceding model-throughput improvement** at **814.527 prefill /
37.063 decode tok/s** on A1. Its matched archived L97 control measures
**814.133 / 36.824**: **+0.048% prefill / +0.648% decode**, with nonoverlapping
decode ranges (control 36.810–36.907, candidate 36.931–37.230). All 26 logit
files / 185 rows and twelve natural 256-token sequences match exactly.
[Matched Release ABBA](../../../../parity-results/qwen-flash-next/reducer100-model-comparison.json)
and [byte comparison](../../../../parity-results/qwen-flash-next/reducer100-release-accuracy-r1/logit-byte-comparison.json)
retain original observer/source ancestry; the component-only frozen98 archive
is a separate control. Model, precision, activation arithmetic, FP16 K/V,
Automatic-resolved Four-worker policy, vocabulary-sharded head, RAM ngrams,
no MTP/prefix cache, context 4096, 512/256 workload and 28 physical host cores are unchanged.
No precision change contributes to this gain.

The [recorded GDN bundle ABBA](../../../../parity-results/qwen-flash-next/grouped-reducer100-component-abba-r1/result.json)
reduces TP1/TP2/TP4 latency **5.019% / 5.684% / 3.125%**. Its new grouped
reducer preserves original producers and ordered FP32 partial folds, reducing
the ordinary direct-output decode graph from seven kernels to six. Concurrent
scaling still misses both targets: **1.304× TP2 / 1.710× TP4** in this matched
comparison. [Exact-kernel profile](../../../../parity-results/qwen-flash-next/grouped-reducer100-profile-summary-r1/report.json)
records TP4 grid32/block128, allocated VGPR24/SGPR48 and zero LDS/scratch;
gfx906 ISA inspection confirms zero register spills for all five compiled
geometries. All 776 Unit and 12 focused registrations pass. The subsequent
[all-five scoped-geometry captured extension and slot/bias neighbor](../../../../parity-results/qwen-flash-next/grouped-reducer100-scoped-native-r1/result.json)
also pass, including direct publication, odd tails, empty partial planes and
guard bytes. Complete production preflight remains pending; G60 remains the
last complete checkpoint. The [full captured-decode trace](../../../../parity-results/qwen-flash-next/reducer100-fulltrace-r1/result.json)
has retired cleanly with the original generated sequence. Its
[completed attribution](../../../../parity-results/qwen-flash-next/reducer100-fulltrace-analysis-r1/REPORT.md)
joins all 255 measured forwards per rank, with 131 classified kernel names and
none unclassified. It identifies 100 branch norms and 120 floating pointer-setup
calls per forward. The norm synchronization candidate passes independent
original-arithmetic qualification, all 776 Unit tests and the canonical model
panel with all 185 logit rows byte-identical to L100. Captured native-event
timing improves M1/H2560 latency 2.794% with M512 effectively unchanged; matched
model ABBA is complete with effectively unchanged throughput. Pointer setup is
a separate opportunity. See
[normalization evidence](QFN_TP_COMPUTE_SCALING.md#branch-normalization-original-reduction-order-with-fewer-barriers).
The profiled envelope is 108.883 ms versus 26.981 ms/token unprofiled, so these
intervals cannot establish removable latency. The nine-row TP2 prefill
comparison remains unqualified because its library reduction differs from the
saved TP4 observation. See
[scaling, scope and evidence](QFN_TP_COMPUTE_SCALING.md#grouped-ordinary-m1-reducers-l100).

**L96 floating reductions are qualified** at **796.370 prefill / 36.750 decode
tok/s** on A1. The matched Automatic94 control measures **791.395 / 36.654**:
**+0.629% prefill**, with decode effectively unchanged (+0.260%, overlapping
ranges). All 185 final-logit rows and twelve generated sequences match exactly;
776 Unit and six focused registrations pass. Isolated ROCm profiles verify the
nine-to-three barrier change, unchanged resources and zero spills. See the
[arithmetic, profile and matched comparison](QFN_FLOATING_REDUCTION_TUNING.md).
Complete production preflight remains pending.

**Automatic Four-worker selection is qualified** for the measured ordinary
MI50 TP4 owner gate/up domain. Fresh default A94 measures **791.635 prefill /
36.645 decode tok/s**, effectively matching explicit Four at **791.084 /
36.668**. All **776 Unit** registrations, the focused native gates and the
185-row accuracy panel pass; all twelve benchmark sequences match. See the
[scope, settings and evidence](QFN_ROUTE_OWNED_MOE_TUNING.md#automatic-four-worker-policy-qualification).
Additional domains require their own measurements before default selection.

The [same-Four vocabulary comparison](QFN_TP_COMPUTE_SCALING.md#default-policy-qualification)
confirms sharding at **785.553 prefill / 36.629 decode tok/s**, versus mirrored
**784.992 / 36.179**: **+1.243% decode**, with effectively unchanged prefill,
185 byte-identical logit rows and twelve identical generated sequences. The
planner now defaults the qualified ordinary QFN Q8_0 MI50/RCCL TP4 family to
sharding. Its regression fails against the old resolver and passes after the
change; all **771 Unit** registrations pass. Current-source native parity and
prefix restore pass, with ten exported head/input records byte-identical to
the earlier mirrored run. This repeats the head-policy comparison on the Four archive;
it is not an additional gain to multiply into L90's already-sharded score.
The recorded head-projection concurrent cohort is green: **1.973–1.978× TP2**
and **3.847–3.865× TP4**, meeting both targets across three recorded rows with
every participant byte-checked. These are projection-compute timings with
restore/communication excluded; other kernel families remain pending. Removing
unused KV heads is a parallel priority across every participant count 1–8.
The aligned geometry and compact CPU/CUDA/ROCm consumer tests pass. The cache
origin and retained-snapshot follow-up passes **19/19** focused registrations
and **771/771 Unit**; it retains every participant's original KV interval and
checks original layer/device ownership. The working candidate now propagates
original-source arithmetic and the same head assignment through automatic
planning, PMA, weights and graphs: **3/3** focused planning/admission aliases
and **773/773 Unit** pass. CUDA/ROCm captured slices pass across all 21 native
source formats at M1/3/17 with twenty histories; seven neighboring GEMM checks
also pass. Bulk M512 plus passive geometry checks pass **9/9**, with exact
slice bytes and zero observed local-memory bytes on both backends. The first
model attempt exposed a shared TP configuration lifetime race during parallel
setup. Owning immutable snapshots fix it; the replacement/retirement regression
passes and complete current Unit is **774/774**, zero skips. The corrected
Release model/reset gate passes with **185 byte-identical logit rows** versus
accepted Four90. Same-run geometry evidence confirms compact K/V on all four
participants, and the canonical TP4 parity/prefix-restore cell passes in
91.07 seconds. Matched Release ABBA now measures **L95: 791.519 prefill /
36.759 decode tok/s**, versus **786.251 / 36.746**: **+0.670% prefill** and
effectively unchanged decode. All twelve natural output sequences match. The
first candidate attempt remains rejected for a driver workqueue warning during
teardown; its unchanged repeat and the other three accepted arms are clean. The current equal-width expert down-projection policy independently
excludes full-model degrees 3/6/7 for H2560; head-layout coverage alone does not
advertise those complete topologies.

The [route-owned MoE follow-up](QFN_ROUTE_OWNED_MOE_TUNING.md) now qualifies
**L90: 785.806 prefill / 36.780 decode tok/s**, a **5.142% paired decode gain**
with unchanged prefill. All twelve generated sequences match. Both same-source
Release accuracy panels retain all **185 final-logit rows byte-identically**
to each other, L74 and accepted84 sharded. The immutable model binding passes
**769/769 Unit**, **18/18 Serial focused checks** and **20/20 Four-worker focused
checks**, including real two- and four-device binding and passive descriptor
inspection on both GPU backends. Native graph inventory confirms the worker
templates/resources on all four devices, with per-instance identity explicitly
unavailable in the aggregate exporter. The source-format comparison and genuine
CPU-promoted23 fixture cover all eighteen physical worker decoders with exact
bytes and lower component latency at the tested N640/K2560 geometry. All eighteen
compiled leaves have zero spills/private memory. The new Automatic policy
selects Four for the qualified MI50 TP4 operation and keeps established policies
elsewhere; explicit Serial/Four remain available. Complete preflight remains G60.
The two/eight-worker contest is complete: neither gives a consistent gain over
Four, and its source is retired with evidence preserved. The next
[compute-scaling diagnostic](QFN_TP_COMPUTE_SCALING.md) preserves recorded inputs
and routes while measuring TP1/TP2/TP4 against cumulative targets of 1.85×/3.6×.
The
[KV coverage plan](QFN_TP4_KV_COVERAGE_PLAN.md) records the separate duplication work.

Updated: 2026-10-23; **L104 is the current A1 checkpoint** at **815.121 / 37.460 tok/s**, versus matched L102 **815.181 / 37.420** (**−0.007% / +0.106%**, overlapping ranges). This counted-row/typed-bias repair preserves model throughput; all 185 logit rows and twelve generated sequences remain exact. L102 is the preceding floating projection checkpoint at **815.053 / 37.424**, a matched **+0.781% decode** gain. L101 remains the preceding branch-normalization checkpoint. **L100 is the preceding model-throughput win** at **814.527 / 37.063 tok/s**. Its matched archived L97 control measures **814.133 / 36.824**, giving **+0.048% / +0.648%** with nonoverlapping decode ranges. L97 remains the preceding native-format checkpoint. L96 remains the preceding floating-reduction checkpoint. A94 Automatic remains the preceding policy checkpoint. L95 is the preceding compact-KV checkpoint. Its historical explicit-Four control measures **786.251 prefill / 36.746 decode tok/s**. L90 independently measured the Four-worker gain against same-source Serial at **785.821 / 34.982 tok/s**. L84 remains the preceding vocabulary-sharded checkpoint at **786.235 / 35.056**; the [TP4 computation audit](QFN_TP4_COMPUTE_OWNERSHIP_AUDIT.md#experiment84-measured-ordinary-m1-head-sharding) retains its matched comparison and final-source85 regressions. Automatic terminal ownership now shards the qualified ordinary MI50 TP4 family; the compiled partition policy now defaults to Automatic, selecting Four only within the measured domain. Use the exact settings recorded below. Complete production preflight remains pending. Local KV-head duplication is removed in the qualified ordinary TP domain; short-context indexer-query work remains a separate opportunity.

Goal: beat **all applicable measured reference engines in both prefill and decode**, while keeping each row's accuracy gate green. For each phase, the target is the fastest applicable measured result across **Castagna Veloce, upstream llama.cpp and Strata**. Keep their pinned revisions, topology differences and arithmetic qualifications explicit, including Strata's approved default-tail7 approximation. Supported but unmeasured comparisons remain pending; an unsupported topology is not an automatic win. Collect the applicable baselines before tuning a row. Runtime configurations remain owned by the typed parity inventory; a planned row does not advertise implemented support.

**Priority:** the next focused win is **four-MI50 decode against Castagna Veloce's 42.959 tok/s**, while preserving the prefill lead and numerical acceptance. The four policy pairs below have been exercised: retain `TensorParallel + GateUpOwnedDownColumns` as the fastest numerically green pair. The requested [CPU ngram investigation](QFN_NGRAM_CPU_PATH_INVESTIGATION.md#measured-follow-up-host-worker-placement-2026-10-21) found all four lookup-service workers sharing one physical core. Candidate69 fixes that placement with green regressions and unchanged logits; paired throughput is effectively unchanged. The [compact-import follow-up](QFN_COMPACT_INTERMEDIATE_IMPORT_TUNING.md) is accurate and faster in isolation, but its complete A/B/B/A gives only −0.060% prefill / +0.194% decode with overlapping ranges. It is not promoted. The [fresh accepted-runtime trace and exact collective fixture](QFN_TP4_COLLECTIVE_ATTRIBUTION.md) now identify a native RCCL selector defect: `AllGather:LL` targeted broadcast because the name table disagrees with the enum. The repair now passes 756/756 Unit registrations, 9/9 focused native/packaging tests and the frozen A1 Release accuracy slice, with all 185 retained logit rows byte-identical to L69. Captured traces prove that the selector now dispatches LL; automatic transport defaults remain unchanged. The [ordered short-K TwoTerm candidate](QFN_ORDERED_SHORT_K_PROJECTION_TUNING.md) now passes all 180 backend/tile/shape checks. Its K320 component is about 1.25× faster on MI50 and 1.08× on RTX 3090 with identical bytes. The shared 32-column production dispatch passes 756/756 Unit, 3/3 focused native/geometry and 19/19 surrounding all-format checks. Frozen A1 accuracy retains all 185 logit rows byte-identically. Four clean matched arms measure **785.397 prefill / 33.882 decode tok/s**, a **+1.797% decode** gain with effectively unchanged prefill. The next probe passed exactness and attributes most isolated transaction cost to gathering: about 41 µs versus 2.29 µs for the fold. A broad LL protocol pilot gives **702.850 / 34.853 tok/s**: better decode but a 10.51% prefill loss, so it is not promoted. The fixed-message channel component 76 is exact but slower at 97 transactions: **+25.8% CUDA P2 / +8.89% ROCm P4 latency**, so it is not attached to production. It exposed a HIP kernel-argument sizing defect: the focused original runtime consumes 24 MiB against the unchanged 8 MiB family budget; the isolated repaired runtime fits 8 MiB and passes four variants with three exact replays each. The dependency repair passes the surrounding **24/24** model-free graph/lifetime/packaging gates, and the complete Unit suite passes **757/757**. Its frozen A1 Release accuracy slice is green, with **all 185 logit rows byte-identical to L74**, exact reset replay and clean native/driver retirement; complete qualification remains pending. The batch-only 16 KiB candidate77 passes **757/757 Unit**, **24/24 focused checks** and frozen A1 accuracy with **all 185 logit rows byte-identical to L74**. It improves the paired ROCm transfer component by **3.53–4.09%** at 97 transactions; CUDA remains **24.93% slower than native**. This is a component result with no production TP attachment or whole-model gain. The direct-consumer/fold prototype78 passes 32 focused checks but is **7.84% slower than native on ROCm and 52.70% slower on CUDA** at 97 transactions, winning 0/31 pairs on each backend. Its source is retired and evidence retained; CUDA sanitizer qualification also remains failed/open. The private size-aware LL candidate79 now passes native boundary/loader/byte checks and a complete model A/B/B/A: **785.609 prefill / 34.697 decode tok/s**, versus matched automatic **785.103 / 33.947**, or **+0.064% / +2.209%**. All twelve measured output sequences match. That plugin remains experimental. The native integration80 now passes **759/759 Unit**, **17/17 focused checks**, frozen accuracy with **185 byte-identical logit rows**, and all four matched model arms. Accepted A1 advances to **L80 785.303 / 34.700 tok/s**, versus matched original **785.911 / 33.939** (**−0.077% prefill / +2.240% decode**); all twelve output sequences match. QFN NativeVNNI coverage is the next tuning pass. Complete production preflight remains pending. TP4 names the participant count, not either policy pair. Other topology tuning is parked until this focused pass is settled; then resume A before novel compositions and remaining families in B–D.

Current evidence: all five admitted rows passed checkpoint60 accuracy. **A1–A4 now have uninstrumented Llaminar observations; A4 is ahead of Castagna in both phases, while A1 decode and A2 prefill remain behind their selected references.** A3 is ahead of the recorded upstream samples, whose decode variance remains unresolved, but A2 and A3 both remain below Strata's approved approximate-policy decode reference. Strata's full-routing and default-tail7 controls are complete for A2/A3. Complete checkpoint60 Unit/production preflight and all five model accuracy cells passed, with clean drivers, unchanged runtime metadata, exact reset replay and normal native retirement. The qualified source is committed and pushed to `feat/qwen-flash-next` as `6c06e8efe`; its commit hook authenticated all 2,454 prerequisite registrations without rerunning them.

Accuracy legend: **🟢 GREEN** = all required numerical groups and execution/retirement/reset checks passed; **🔴 RED** = an observed gate failure; **⚪ PENDING** = not tested or prerequisites missing. Hardware, capacity and implementation blocks are recorded separately from numerical failure. `—` means unmeasured. All throughput cells are **tokens/second**.

## Active tuning pass: QFN NativeVNNI dispatch coverage

With native small-AllGather L80 qualified as a focused checkpoint, audit QFN GEMM/GEMV
dispatch on CPU, CUDA and ROCm. Inventory the actual native weight formats,
activation policies and live M/N/K shapes for prefill and decode, including the
TP slices and expert partitions used by every admitted dashboard topology.
Compare those keys with the installed exact overlays and observe the kernels
selected by real inference; an overlay entry alone does not prove selection.
Measure missing or suboptimal entries with the canonical NativeVNNI tuning
workflow, preserve numerical acceptance and arithmetic order, then qualify
end-to-end performance before installing winners. Extend dispatch coverage and
focused regression registrations for any defects found.

The initial source audit finds no QFN entry in the common shape catalog. Nine
physical dense shapes recorded in accepted L74 A1 have no installed CB19 exact
keys on either GPU backend; most full QFN CPU dimensions likewise lack exact
entries. The vocabulary head is already covered. Existing generic policies
still own these shapes, so missing entries alone do not establish a slowdown.
TwoTerm HC-up and floating router/alpha-beta use separate kernel families.

Before collecting new winners, fix two identity gaps. The ROCm dense exact key
omits the prepared-weight family: legitimate native-Q8 and packed-INT8 programs
can share CB19/M/N/K, while using different operand layouts. A source-derived
witness at M64/N512/K5120 collides with an installed packed-INT8 row. The
[original-production regression](../../../../parity-results/qwen-flash-next/nativevnni81-red-regression/result.json)
reproduces that exact exception after the expanded-INT8 control and 64 native
serial-row oracles pass. The expert pair key also omits the
physical down-output width. A1 computes gate/up640×2560 and down640×640;
whole-expert down2560×640 evidence cannot qualify that compact operation merely
because both retain H2560 for their arithmetic contract. No new dispatcher or
performance gain has been installed by this audit.

Candidate81 adds 22 missing QFN geometries to the common catalog, including
source-owned TP2/TP4 slices and the quantized MTP HC-inject matrix. All 606
original resolved records remain unchanged; the inventory now has 628 records,
120 production geometries. This is measurement coverage, not new topology
admission or a set of measured winners. CPU ordinary-prefill collection uses
the explicit large-model evidence budget without asserting a model parameter
total. The real C++ loader and Python loader agree on every resolved field in
the [five focused checks](../../../../parity-results/qwen-flash-next/nativevnni81-inventory-focused/result.json).
The surrounding run passed 62/63 registrations; its stale CPU process-count
assertion is [fixed and green](../../../../parity-results/qwen-flash-next/nativevnni81-inventory-training-plan/result.json),
now deriving equal-M groups from the authoritative shape/format/ISA inventory.

The prepared-family repair passes its focused native and Unit gates. NativeVNNI and expanded
Int8Blockwise now have distinct complete keys, including for CB19. All 576
installed measured configurations retain their original numeric keys and launch
tuples ([540 native, 36 expanded INT8](../../../../parity-results/qwen-flash-next/nativevnni81-installed-policy-preservation.json)); no native-Q8 winner is invented. Runtime,
generator, additive retention and explicit preflight regressions share that
identity. The [five focused checks](../../../../parity-results/qwen-flash-next/nativevnni81-family-focused/result.json)
pass, including three captured replays with every native-Q8 output byte equal
to all 64 serial rows and the unchanged expanded-INT8 selection.
All [16 surrounding registrations](../../../../parity-results/qwen-flash-next/nativevnni81-family-surrounding/result.json)
pass: installed all-format exact entries, actual producer identity, prefill
bucket invariance and grouped prepared-source byte equivalence.
The rebuilt [complete Unit suite](../../../../parity-results/qwen-flash-next/nativevnni81-unit/result.json)
passes **760/760**, with no skips, in **115.61 seconds**. Driver windows and
native retirement are clean. The [archived Release accuracy run](../../../../parity-results/qwen-flash-next/nativevnni81-release-accuracy/runtime-observation.json)
passes the unchanged canonical panel in **101.68 seconds**. All **26 retained
files / 185 final-logit rows** are [byte-identical to L74](../../../../parity-results/qwen-flash-next/nativevnni81-release-accuracy/shortk74-logit-byte-comparison.json),
including reset and diagnostic rows; the ordered short-K path is observed.
This is a correctness and measurement-coverage checkpoint. Throughput remains
the measured L80 result; these focused passes do not replace complete
production preflight.

An exact-table refresh alone cannot improve the current ROCm native-Q8 M1 path:
its numerical contract fixes K partitions, and its launcher fixes TN128/CPT1.
The existing target-waves value is overwritten when the resolved partition
count is applied. Experiment82 now exposes five independent output-column
geometries while preserving each column's arithmetic, the exact partition
boundaries and every ordered partial addition. The existing baseline remains
installed until repeated captured timing, compiler resources, byte equivalence
and the model comparison establish a winner. CUDA and CPU have different
effective dispatch surfaces and need their own measured evidence.

The first [qualified Release sweep](../../../../parity-results/qwen-flash-next/nativevnni82-release-tournament-loader-fixed/result.json)
passes **50/50 candidate cells and all 5,050 timing samples**, with no output or
retained-replay byte mismatches. It uses the real prepared native-Q8 public
projection, including activation quantization, captured at M1 on gfx906; each
candidate has 31 warmups and 101 samples. These synthetic component weights
are not the model accuracy panel. The ordinary unscoped TN128/CPT1 launch is
the single timed control. The [common analyzer summary](../../../../parity-results/qwen-flash-next/nativevnni82-release-tournament-loader-fixed/summary.csv)
is explicitly diagnostic and cannot emit or install a policy.

| Physical N × K | Default median µs | Lowest median µs | Geometry at lowest median | Latency reduction |
|---|---:|---:|---|---:|
| 160 × 2560 | 27.04 | 24.48 | TN256/CPT1 | 9.47% |
| 320 × 10240 | 35.84 | 31.36 | TN256/CPT1 | 12.50% |
| 512 × 2560 | 27.68 | 24.96 | TN256/CPT1 | 9.83% |
| 1536 × 2560 | 30.72 | 26.72 | TN256/CPT1 | 13.02% |
| 2560 × 160 | 23.52 | 22.40 | TN128/CPT2 | 4.76% |
| 2560 × 1536 | 30.88 | 26.88 | TN256/CPT1 | 12.95% |
| 2560 × 2560 | 34.88 | 30.08 | TN256/CPT1 | 13.76% |
| 3072 × 2560 | 37.44 | 32.16 | TN64/CPT1 | 14.10% |
| 10240 × 2560 | 60.48 | 57.76 | TN256/CPT1 | 4.50% |
| 248320 × 2560 | 994.88 | 994.88 | Default TN128/CPT1 | 0.00% |

This first pass uses fixed candidate order and one graph replay per event sample.
Retain every result, including slower candidates. The balanced follow-up below
supersedes these screening values for candidate selection. No component
percentage is a whole-model throughput claim.

The [paired single-projection run](../../../../parity-results/qwen-flash-next/nativevnni82-paired-release-one/result.json)
and [captured batch32 run](../../../../parity-results/qwen-flash-next/nativevnni82-paired-release-batch32/result.json)
both pass all **50 candidate/shape cases and 5,050 raw samples**. The trainer
retains all five actual graphs, rotates and reverses their order through a
shared ten-round schedule, and preserves every original round and position.
Each complete schedule balances positions and both directions of every pair;
the final partial round of the 101-sample run remains included. Batch32 is
one graph containing 32 complete public projections, including quantization,
submitted once per event sample. Actual graph nodes and construction counters
prove that multiplicity. The original unscoped one-operation capture remains
the byte oracle. Both runs retain zero output/replay mismatches, the intended
native loader closure, clean driver windows and normal retirement.

The common analyzer retains all five candidates in each
[single-projection summary](../../../../parity-results/qwen-flash-next/nativevnni82-paired-release-one/summary.csv)
and [batch32 summary](../../../../parity-results/qwen-flash-next/nativevnni82-paired-release-batch32/summary.csv).
The table shows the best paired geometric-mean speedup in each separate regime;
it is a diagnostic comparison, not an installed selection or model result.

| Physical N × K | Single control µs | Best single speedup | Batch32 control µs/op | Best batch32 speedup | Best batch32 geometry |
|---|---:|---:|---:|---:|---|
| 160 × 2560 | 24.640 | 1.0024× | 10.125 | 1.0032× | TN64/CPT1 |
| 320 × 10240 | 31.201 | 1.0000× | 18.320 | 1.0073× | TN64/CPT1 |
| 512 × 2560 | 25.120 | 1.0090× | 10.515 | 1.0195× | TN64/CPT1 |
| 1536 × 2560 | 26.720 | 1.0021× | 12.450 | 1.0000× | Default TN128/CPT1 |
| 2560 × 160 | 21.760 | 1.0014× | 7.330 | 1.0033× | TN64/CPT1 |
| 2560 × 1536 | 26.880 | 1.0014× | 12.615 | 1.0005× | TN256/CPT1 |
| 2560 × 2560 | 30.400 | 1.0020× | 18.130 | 1.0130× | TN64/CPT1 |
| 3072 × 2560 | 32.320 | 1.0079× | 20.630 | 1.0205× | TN64/CPT1 |
| 10240 × 2560 | 58.561 | 1.0000× | 47.705 | 1.0000× | Default TN128/CPT1 |
| 248320 × 2560 | 996.963 | 1.0000× | 980.208 | 1.0000× | Default TN128/CPT1 |

The earlier 4.5–14% reductions mostly disappear when candidate ordering is
balanced. Batch32 reduces submission overhead per operation and changes cache
reuse; its normalized times are not interchangeable with full-model timings.
These results do not justify promoting a new geometry. The full head and large
projection retain the existing winner. Physical-selector groundwork and dynamic
profiling are parked; neither has been installed. Follow the concrete TP4
duplication opportunities before spending more qualification work on these
small component differences.

The [TP4 ownership audit](QFN_TP4_COMPUTE_OWNERSHIP_AUDIT.md) verifies that GDN,
query heads, shared FFN and routed expert columns are sharded. Explicit HC,
PLE, indexer, router and terminal-head replicas remain. The first implementation
candidate is ordinary vocabulary gathering with MTP disabled and zero reserved
capacity: its existing native publication is incorrectly gated on MTP capacity.
Keep the current Auto policy while fixing and measuring the explicit sharded
mode. Consumer-minimal KV ownership and unused short-context indexer query work
are separate next candidates; their typed ownership and replay tests remain
required before any speed claim.

The [Release build](../../../../parity-results/qwen-flash-next/nativevnni82-paired-release-build-r1/result.json)
passes, as do the [seven focused checks](../../../../parity-results/qwen-flash-next/nativevnni82-paired-focused/result.json)
and [complete Unit suite](../../../../parity-results/qwen-flash-next/nativevnni82-paired-unit/result.json):
**761/761, zero failures or skips, 116.45 seconds**. Focused preflight entries
cover the actual shared order and malformed pairing/capture evidence, alongside
the existing native geometry and benchmark loader checks. This change touches
the test/analysis harness only; it does not replace the retained model accuracy
receipt or claim a new complete production-preflight run. The first Release
build attempt also requested a Unit-only target and was rejected before
compilation; its [failed receipt](../../../../parity-results/qwen-flash-next/nativevnni82-paired-release-build/result.json)
is preserved, and the corrected build targets the Release trainer alone.

The [exact compiled gfx906 functions](../../../../parity-results/qwen-flash-next/nativevnni82-isa/resources.json)
have zero spills, private storage and LDS. Producers use 18 VGPRs; reducers use
23 for CPT1 and 27 for CPT2. Actual loaded-function queries agree and prove
resident blocks at each launch width; these are resource limits, not measured
occupancy. Captured kernel-node functions and grids match the recorded launches.
The [initial 13 focused checks](../../../../parity-results/qwen-flash-next/nativevnni82-focused/result.json)
pass, including tail widths, empty partition planes, poisoned replays and
invalid scoped bindings.

The [first timing attempt](../../../../parity-results/qwen-flash-next/nativevnni82-release-tournament/result.json)
was rejected: the benchmark's HIP host link embedded an overriding SDK RPATH
and loaded LLVM OpenMP alongside the compiler-selected GNU runtime. Its 40
partial rows remain rejected evidence. A [device-free original-code control](../../../../parity-results/qwen-flash-next/nativevnni82-loader-negative/result.json)
reproduces the wrong OpenMP binding. The target now uses the C++ host linker
and `OpenMP::OpenMP_CXX`; HIP source compilation and the spill guard remain
enabled. The actual benchmark and application loader controls, mixed-runtime
negative cases, geometry bytes and evidence validation all pass in the
[five-check follow-up](../../../../parity-results/qwen-flash-next/nativevnni82-openmp-focused/result.json).
The successful sweep authenticates its complete loader closure before device
launch, observes the intended core/HIP/GNU runtime, and retires normally with
an unchanged source/runtime snapshot and clean driver window. Its exact build
receipt is [nativevnni82-openmp-release-build](../../../../parity-results/qwen-flash-next/nativevnni82-openmp-release-build/result.json).

The [complete Unit suite](../../../../parity-results/qwen-flash-next/nativevnni82-unit/result.json)
passes **760/760**, zero skips, in **116.28 seconds**. The
[surrounding native run](../../../../parity-results/qwen-flash-next/nativevnni82-surrounding/result.json)
passes all four selected registrations plus their fixture in **19.19 seconds**:
complete GEMV format coverage, grouped prepared-source byte equivalence, the
Qwen GDN output shard, and actual serial-route telemetry. Driver windows and
native retirement are clean.

The [unchanged-default Release accuracy panel](../../../../parity-results/qwen-flash-next/nativevnni82-release-accuracy/runtime-observation.json)
passes in **100.87 seconds**. All **26 files / 185 final-logit rows** remain
[byte-identical to L74](../../../../parity-results/qwen-flash-next/nativevnni82-release-accuracy/shortk74-logit-byte-comparison.json),
including reset and diagnostic rows. This qualifies the infrastructure with the
existing production geometry; it does not qualify an installed faster policy.
The new [actual construction-route observations](../../../../parity-results/qwen-flash-next/nativevnni82-release-accuracy/native-q8-physical-route-observation.json)
show seven ordinary serial shapes and the vocabulary head on TN128/CPT1/KB16.
The head has physical N248320 but arithmetic-policy N62080, which the component
head's equal-N key does not represent. Neither 160×2560 nor 2560×160 appears in
this serial producer's model observations, so their isolated timings cannot yet
be attributed to A1. Physical shape, arithmetic-policy shape and fused/batched
operation identity must remain distinct in any measured dispatch update.

The [actual GGUF descriptor inventory](../../../../parity-results/qwen-flash-next/nativevnni81-source-descriptors.json)
was collected through the canonical bounded header parser, without scanning or
hashing tensor payloads. Main ordinary dense/HC/PLE/head weights are Q8_0;
routed gate/up use Q4_K in47 layers and Q5_K in1, while down uses Q5_1 in43 and
Q8_0 in5. Indexer weights arrive as BF16. The MTP sidecar also has quantized
HC-inject matrices (N4/K10240), unlike the main model's F32 HC-inject weights;
its actual prepared route must be represented explicitly in coverage.

## A. Implemented and admitted — baseline and tune first

`G60` names the last complete qualification checkpoint. `L102†` removes custom floating pointer staging while preserving arithmetic and real table consumers; 776 Unit checks, the focused native regressions, 185 exact logit rows and twelve identical generated sequences pass. Matched decode improves 0.781%; prefill is effectively unchanged. Cold hipBLAS capture and explicit input-event regressions are included. Isolated profiles show unchanged registers/LDS and zero scratch. Full preflight remains pending. `L101†` reduces HIP branch-norm barriers from nine to three while retaining original arithmetic; 776 Unit tests, three focused registrations, 185 exact logit rows, native profiles and four matched model arms pass. Component M1 latency improves 2.794%; whole-model throughput is unchanged. `L100†` groups ordinary ROCm M1 reducers with unchanged producers, source precision and ordered FP32 folds; 776 Unit, 12 final focused registrations, 26 byte-identical logit files / 185 rows and twelve equal natural sequences pass. Matched decode improves 0.648%; the subsequent all-five scoped-geometry captured extension and slot/bias neighbor pass. Full preflight remains pending. `L97†` specializes native FP16/BF16 source encoding without changing arithmetic; 776 Unit tests, six focused registrations, all 185 byte-identical logit rows, isolated profiles and four matched model arms pass. Prefill improves 2.254%; decode is 0.229% lower with overlapping ranges. `L96†` adds the original-tree wave64 reduction tail, with 776 Unit checks, six focused registrations, 185 byte-identical logit rows, isolated profile/resource evidence and four clean matched model arms. Its decode difference is small with overlapping ranges. `A94†` qualifies Automatic selection of the L90 Four winner on top of L95, with 776 Unit registrations, twenty focused CTest registrations (including two native default cases), exact 185-row accuracy and matched ABBA. Its timing is effectively unchanged versus explicit Four. `L95†` adds query-covered K/V storage and projections with original arithmetic, 774 Unit registrations, all-format native slice tests, exact 185-row accuracy, canonical prefix restoration and matched throughput. See the [KV qualification record](QFN_TP4_KV_COVERAGE_PLAN.md#matched-release-throughput-l95), including the rejected driver-warning attempt. `L90†` measures the explicit compiled Four-worker owner gate/up policy against same-source Serial, with 769 Unit checks, 18/20 focused checks, unchanged 185-row accuracy, native family inventory and four clean timing arms. That historical experiment used an explicit build choice; A94 now qualifies the bounded Automatic default. `L62` names the original uninstrumented Llaminar Release measurements. `L84†` measures explicit ordinary `VocabularySharded` output on top of L80; follow-up85 validates the final capacity implementation with complete Unit, focused native checks and byte-identical Release accuracy. L84 timing and final-source85 accuracy are separate retained observations, and neither replaces G60. `L80†` adds the native small-AllGather preference to the ordered short-K L74 checkpoint, with complete Unit, focused dependency/capture checks, Release accuracy and matched timing passed; complete production preflight remains pending. `L69†` remains the lookup-worker placement correctness fix with effectively unchanged paired throughput. `L66†` remains the earlier measured GPU-control affinity improvement. `L64†` remains the earlier transfer-scheduling checkpoint. A4 passes both phase comparisons on the declared short workload; A3's reference variance remains open. Historical `R58†` values remain in the evidence discussion and do not populate current measured rows.

| ID | Topology / actual work placement | Ngrams | Reference engine | Accuracy | Llaminar prefill | Reference prefill | Strata prefill | Llaminar decode | Reference decode | Strata decode | Next action |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| A1 | 4×ROCm: TP4, `TensorParallel` + `GateUpOwnedDownColumns`, all GPU experts; explicit `VocabularySharded` terminal head and Automatic-resolved Four-worker owner gate/up | RAM | Castagna | 🟢 L106† Release required groups; L95 prefix restore; latest complete topology gate [G60][a1] | 815.992 L106† | 467.821 F62§ | — | 39.274 L106† | 42.959 F62§ | — | L106 schedules FP32 loads by captured geometry: +4.676% matched decode with unchanged prefill and exact logits. Next qualify packed FP16 attention/KV sharing. Full preflight remains pending. |
| A1.E151 | Same A1 topology; explicit FP16 attention with `prefill8_decode33`, native FP16 KV; MTP/prefix off | RAM | Castagna | 🟢 Required Release groups; 832 Unit/affected registrations; boundary panel diagnostic | 818.579 E151† | 467.821 F62§ | — | 40.110 E151† | 42.959 F62§ | — | +0.566% decode versus same-binary uniform8. Tune decode head grouping; retain prefill. Experimental arithmetic; no new complete preflight or default promotion. |
| A1.E160 | Same A1 topology; explicit FP16 `prefill8_decode33`, V0/V1 overlap and PV unroll2; MTP/prefix off | RAM | Castagna | 🟢 Required Release groups; 185 exact rows; 832 Unit/affected registrations | 818.413 E160† | 467.821 F62§ | — | 40.111 E160† | 42.959 F62§ | — | Matched model throughput is flat versus frozen151. Native M1/512 pair15.446µs and M16/64K88.597µs. Experimental arithmetic; full preflight/default promotion remain open. |
| A1.E164 | Same A1 topology; explicit FP16 `prefill8_decode33`, K1/V0/V1 overlap and QK64/32; MTP/prefix off | RAM | Castagna | 🟢 Required Release groups; 185 exact rows; 832 Unit/affected plus new K1 oracle registration | 819.682 E164† | 467.821 F62§ | — | 40.265 E164† | 42.959 F62§ | — | Matched model ranges overlap frozen151. Native M1/512 pair15.086µs and M16/64K84.354µs. Experimental arithmetic; full preflight/default promotion remain open. |
| A1.E165 | Same A1 topology; explicit FP16 `prefill8_decode33`, retained K0/K1 row addresses; MTP/prefix off | RAM | Castagna | 🟢 Required Release groups; 185 exact rows; 20 native tests and 832 Unit/affected | 818.073 E165† | 467.821 F62§ | — | 40.190 E165† | 42.959 F62§ | — | Matched model ranges overlap frozen151. Native M1/51215.054µs, M1/64K19.127µs, M16/64K82.285µs. Fresh paired profiles identify merge as next target; default/full-preflight status unchanged. |
| A1.E179 | Same A1 topology; explicit FP16 `prefill8_decode33`, merged GDN inputs and cooperative compact grouping; MTP/prefix off | RAM | Castagna | 🟢 Required Release groups; 185 exact rows; 798 Unit/affected; final native/profile/path proofs | 819.648 E179† | 467.821 F62§ | — | 40.656 E179† | 42.959 F62§ | — | +0.897% matched decode versus frozen177 with non-overlapping ranges; prefill flat. All nine sequences match. Experimental attention arithmetic; default promotion and complete preflight remain open. |
| A2 | 1×ROCm + 1×CPU: expert overlay, 8 GPU / 504 CPU experts per layer | Adaptive VRAM/RAM/SSD, zero-cache control | Castagna | [🟢 GREEN G60][a2] | 156.707 L62 | 159.177 F61-30 | 45.270 S62≈ | 15.164 L62 | 11.600 F61-30 | 24.024 S62≈ | Prefill remains 1.6% below Castagna; decode remains below Strata's 24.024. Both reference policies stay explicit. |
| A3 | 1×CUDA + 1×CPU: expert overlay, 8 GPU / 504 CPU experts per layer | Adaptive VRAM/RAM/SSD, zero-cache control | llama.cpp | [🟢 GREEN G60][a3] | 180.920 L62 | 129.020 U62⚠ | 74.301 S62≈ | 18.231 L62 | 13.059 U62⚠ | 31.253 S62≈ | Ahead of recorded upstream samples; below Strata's 31.253 decode. Resolve upstream variance and tune against the faster applicable reference. |
| A4 | CPU only: one NUMA participant, 28 physical-core workers | SSD mmap | Castagna | [🟢 GREEN G60][a4] | 98.466 L62 | 48.408 F61-30 | n/a | 5.797 L62 | 4.611 F61-30 | n/a | Both phases ahead of Castagna: 2.03× prefill, 1.26× decode. Complete other applicable baselines before closing the broader row. |
| A5 | PP(1×CUDA, 1×ROCm, 1×CPU): 8 / 12 / 28 layers | SSD mmap | Castagna / llama.cpp (unavailable at current pins) | [🟢 GREEN G60][a5] | — | — | — | — | — | — | Exact GPU→GPU→CPU-tail reference cannot be represented; no alternate-topology substitution or automatic win. |

A2/A3 retain most experts on CPU: these are not all-GPU performance results. Their zero-cache adaptive controls do not qualify positive-capacity promotion/eviction. A1 uses Llaminar's existing `GateUpOwnedDownColumns` partition. Castagna TP4 shards each expert's intermediate dimension instead. A same-hardware/model end-to-end comparison is useful now, with that difference recorded; implementing C4 must not delay tuning A1.

Use the exact policy names in subsequent four-GPU records. The dense-compute policy is `TensorParallel` (CLI `tensor-parallel`) or `PrefillTensorParallelDecodeReplicated` (CLI `prefill-tensor-parallel-decode-replicated`). Independently, routed experts use `GateUpOwnedDownColumns` (CLI `gate-up-owned-down-columns`) or whole-expert `Apportioned` (CLI `apportioned`). These are independent choices:

| ID | Dense-compute policy | Expert policy | QFN accuracy | Llaminar prefill | Castagna prefill | Llaminar decode | Castagna decode | Status |
|---|---|---|---|---:|---:|---:|---:|---|
| A1 | `TensorParallel` | `GateUpOwnedDownColumns` | 🟢 L106† Release required groups; L95 prefix restore | 815.992 | 467.821 | 39.274 | 42.959 | Explicit `VocabularySharded` head, Automatic-resolved Four-worker owner gate/up, compact K/V, HC activation/publication fusion and geometry-selected FP32 load schedules; 185 identical logit rows and all twelve matching sequences. |
| A1.1 | `TensorParallel` | `Apportioned` | 🟢 L64† Release required groups | 561.867 | 467.821 | 30.515 | 42.959 | Accuracy, corrected SnapshotAdmission and timing passed; slower than current A1 in both phases. |
| A1.2 | `PrefillTensorParallelDecodeReplicated` | `GateUpOwnedDownColumns` | 🔴 L65† Release short decode row KL | — | 467.821 | — | 42.959 | Admission and native execution passed; one forced decode row KL 1.119 exceeds the unchanged 1.0 ceiling. Diagnose before promotion. Timing pending. |
| A1.3 | `PrefillTensorParallelDecodeReplicated` | `Apportioned` | 🟢 L65† Release required groups | 382.757 | 467.821 | 30.761 | 42.959 | Retained **Ornith** arrangement; admission, accuracy and timing passed, but both phases are slower than the current A1 control and Castagna. |

A1.2's [first complete L65 Release accuracy run](../../../../parity-results/qwen-flash-next/policy65-release-replicated-gateup-accuracy/rocm4-dense-prefill-tp-decode-replicated-routed-gate-up-owned-down-columns-ram-hc-up-two-term/report.json) finished with native exit0, clean drivers, byte-equal reset replay and authenticated production execution, but its numerical gate is **red**. Short-panel prefill/decode mean KL is **0.033535869 / 0.040144556**, top-1 **93.75% / 94.79%**, and top-5 overlap **87.50% / 90.42%**. The sole failed criterion is the unchanged **1.0 per-row KL ceiling**: `prose-cause:decode:4` reaches **1.118726**, versus A1 **0.008986** and A1.1 **0.025321** on the identical forced history. All 24 required short/heldout prefill metric records equal A1's; this is not a claim of byte-identical logits or of equal diagnostic N2047 results. Heldout prefill/decode mean KL **0.009691368 / 0.003807744** passes. Keep this failure and its frozen thresholds intact while tracing the decode difference; do not infer a throughput or promotion result from a numerically red run. The [four-way comparison](../../../../parity-results/qwen-flash-next/policy65-four-way-numerical-audit/REPORT.md) retains every matched row; A1.3 gives KL **0.006106** on this exact outlier with the same executable as A1.2. This does not identify a generic replicated-decode or handoff defect. `L65†` remains subject to complete production preflight.

A1.3's [L65 Release accuracy](../../../../parity-results/qwen-flash-next/policy65-release-replicated-apportioned-accuracy/rocm4-dense-prefill-tp-decode-replicated-routed-apportioned-ram-hc-up-two-term/report.json) passes every required group, driver window, reset replay and native-retirement check. Short-panel prefill/decode mean KL is **0.018669085 / 0.027724139**, top-1 **93.75% / 97.92%**, and top-5 overlap **93.75% / 90.83%**. Heldout KL is **0.010157034 / 0.004906650**, with **100%** top-1 agreement in both phases. The uninstrumented [matched benchmark](../../../../parity-results/qwen-flash-next/policy65-replicated-apportioned-benchmark/result.json) passes source/runtime, driver, resource and retirement checks. After one warmup, prefill samples are **382.757 / 382.995 / 382.170 tok/s**, and decode **30.759 / 30.761 / 30.968 tok/s**. All three 256-token sequences [match A1 and A1.1](../../../../parity-results/qwen-flash-next/policy65-replicated-apportioned-benchmark/output-comparison.json). These are the same declared workload and arithmetic, with the disclosed dense/expert policy changes; they are not an interleaved A/B comparison. This arrangement does not close the decode gap and loses the prefill lead, so A1 remains the tuning control. The common live-state handoff is a plausible extra prefill cost, but no phase attribution has measured its contribution yet.

The [historical Ornith report](../2026-09/2026-09-20-ornith-rocm4-auto-reproduction.md) uses a different model, source and runtime; its speed is not a QFN baseline. All three additional QFN pairs now have corrected admission evidence. A1.1 and A1.3 have canonical Release accuracy and timing; A1.2 has a numerical failure and no accepted timing result. Do not silently change A1's pair or attach G60's badge to a new combination.

A1.2's detailed `prose-cause` reference now covers every layer and six forced decode steps. All seven final-logit rows are [byte-identical to the frozen reference](../../../../parity-results/qwen-flash-next/reference65-full-stage/prose-cause-final-row-comparison.json). The [native full-stage replay](../../../../parity-results/qwen-flash-next/full-stage65-focused-prose-cause-replicated-gateup-staged/focused-observation.json) stopped at the complete physical BOM before logits, with clean drivers and complete retirement. Unlike the final-logit canary, this diagnostic retains all intermediate snapshots. Their allocation still scales with the physical M512 family, even though this history uses nine live rows; projecting live rows after a full copy does not reduce that allocation. This is a diagnostic capacity block, not another ordinary inference failure or an explanation of the numerical outlier. Any repair must bind the actual snapshot destination capacity and copy extent together; lowering an estimate alone is invalid.

The measured affinity66 candidate changes control-worker construction affinity only. The [original-production negative controls](../../../../parity-results/qwen-flash-next/affinity66-negative/result.json) reproduce restricted child masks for both CUDA and ROCm with OpenMP binding enabled; both unbound controls pass. The scoped fix reuses the existing HTTP placement algorithm, widens only the constructing thread while independent GPU-control children inherit the admitted OpenMP partition, and restores the parent's exact mask. CPU/mixed/composed TP pools retain their established placement. New actual backend-worker and TP-pool regressions have explicit preflight entries. The [first archived-L65 A/B/B/A control arm](../../../../parity-results/qwen-flash-next/affinity66-abba-a1-control/result.json) passed at median **779.209 prefill / 31.674 decode tok/s**, with clean drivers, unchanged runtime/assets and normal retirement. The [complete owning Integration build](../../../../parity-results/qwen-flash-next/affinity66-integration-build/result.json) passed in 272.76 seconds, all [11 focused registrations](../../../../parity-results/qwen-flash-next/affinity66-focused/result.json) passed in 19.13 seconds, and [754/754 Unit registrations](../../../../parity-results/qwen-flash-next/affinity66-unit/result.json) passed in 115.73 seconds. Focused coverage includes both full native GPU-context suites, actual bound/unbound workers and the shared HTTP scope. Driver windows and retirement are clean. The [Release build](../../../../parity-results/qwen-flash-next/affinity66-release-build/result.json) passed in 130.95 seconds. The [frozen A1 Release accuracy run](../../../../parity-results/qwen-flash-next/affinity66-release-accuracy-diagnostic/report.json) passes all required groups, path/reset/driver/retirement checks; [all 26 files / 185 rows](../../../../parity-results/qwen-flash-next/affinity66-release-accuracy-diagnostic/phase64-logit-byte-comparison.json) are byte-identical to phase64. This canonical accuracy environment is unbound; the separate ordinary-bootstrap timing arms exercise bound control-worker placement and require identical generated token arrays. The [completed A/B/B/A comparison](../../../../parity-results/qwen-flash-next/affinity66-abba-comparison.json) retains all four clean arms and all twelve identical measured output-token arrays. Six samples per role give control **779.090932 prefill / 31.662366 decode** versus candidate **784.382147 / 33.376371 tok/s**: **+0.679% prefill / +5.413% decode**. Prefill ranges are 777.985–781.499 control and 784.138–785.285 candidate; decode ranges are 31.248–31.704 and 33.306–33.450. The candidate still trails Castagna decode. The current Release executable/core are archived separately before another rebuild, with loader relocation checked and external assets metadata-pinned; no whole-model or whole-core hashing was added. Complete production preflight remains pending.

The three new source cases and their exact paired profiles build and export through the canonical inventory: 112 numerical cells, exactly eight paired profiles (the original five plus three). All four focused definition/profile regressions passed, followed by [754/754 Unit tests](../../../../parity-results/qwen-flash-next/policy64-unit/result.json) in 116.36 seconds. The [first three standalone SnapshotAdmission cases](../../../../parity-results/qwen-flash-next/policy64-three-snapshot-admissions/result.json) all failed physical admission, with clean drivers and complete process retirement. Their test setup reserves **13,018,822,720 bytes per GPU** for all-layer snapshots at 512 retained rows; ordinary nine-row diagnostic history needs a different envelope. Standalone setup enables campaign evidence only after its bucket-selection hook has already returned. The earlier A1 admission used an explicit campaign environment and nine retained rows. This is a diagnostic setup-order defect, not evidence that ordinary production inference exceeds the cards' capacity. The fixture now activates its lifetime before the existing bucket resolver; focused regressions exercise the actual helper and preserve explicit frozen M512 geometry. The corrected rebuild, [six focused regressions](../../../../parity-results/qwen-flash-next/policy64-snapshot-order-focused/result.json), and [754/754 complete Unit registrations](../../../../parity-results/qwen-flash-next/policy64-snapshot-order-unit/result.json) passed. The [real admission rerun](../../../../parity-results/qwen-flash-next/policy64-three-snapshot-admissions-corrected/result.json) passed A1.1 with **363,753,168 bytes per GPU** at nine retained rows. Both replicated-decode pairs reached graph materialization but failed a separate workspace contract: actual serial family **820,752,388 bytes**, admitted **615,364,864 bytes**, on all four devices. The complete batch retired with clean drivers. A1.1 timing passed while that mismatch is investigated. The original failed receipts remain failures.

A1.2's [ordinary M512 Release accuracy attempt](../../../../parity-results/qwen-flash-next/policy64-release-replicated-gateup-diagnosis/rocm4-dense-prefill-tp-decode-replicated-routed-gate-up-owned-down-columns-ram-hc-up-two-term/report.json) also failed before logits: actual workspace **977,624,068 bytes**, admitted **954,447,616 bytes**. The mismatch therefore affects serving as well as the smaller diagnostic bucket. An unchanged-policy [nine-row diagnostic with the existing per-buffer BOM log](../../../../parity-results/qwen-flash-next/policy64-replicated-workspace-bom-diagnosis/result.json) identified the missing contract: six named `GDNLiveStateAllGatherStage` scratch banks per GDN layer, totaling **264,757,248 bytes across 36 layers**. These are distinct workspace pointers from the recurrent kernel-state allocation, even though that allocation happens to have the same byte count. The actual serial workspace plan, rather than a sum of logical aliases, remains the physical allocation authority. The applied fix shares the stage's typed descriptor producer with admission and includes actual handoff layers and policies, preserving original buffers and execution. No anonymous reserve, reduced TP degree or alternate execution policy has been substituted.

The [old-production negative](../../../../parity-results/qwen-flash-next/policy65-gdn-handoff-negative/result.json) reproduces zero admission contribution against the actual stage's 264,757,248-byte contract on both CUDA and ROCm. After the fix, the expanded device-free regression covers TP2/4/8, full and partial global layer ranges, M1/M9/M512, B2×M1, non-native contexts and already-full primary state. All [five focused checks](../../../../parity-results/qwen-flash-next/policy65-gdn-handoff-focused/result.json) passed in 2.66 seconds, followed by [754/754 complete Unit registrations](../../../../parity-results/qwen-flash-next/policy65-gdn-handoff-unit/result.json) in 115.98 seconds. The regression has its own explicit `ProductionTestPreflight` entry. Both original [four-MI50 SnapshotAdmission retries](../../../../parity-results/qwen-flash-next/policy65-replicated-snapshot-admissions/result.json) then passed: A1.2 in 85.04 seconds and A1.3 in the same 156.54-second batch, with clean drivers and complete retirement. Each actual graph still needs **820,752,388 bytes**; admission now records **880,122,112 bytes**, including the exact named handoff contribution. Subsequent Release observations are recorded above: A1.2 is numerically red; A1.3 passes accuracy and timing but is slower. Complete production preflight remains pending. Dynamic full-source execution and retained-MTP runtime qualification are not established by this focused slice.

The separate Release final-logit corpus retains production M512. Its first metadata-only attempt stopped before model loading because manually supplied `HOME` changed an unused disabled-cache directory; the fresh v2 metadata uses the canonical runner environment and preserves reference logits and numerical limits. A1.1's first native attempt completed numerically but failed its driver window on an `amdgpu_amdkfd_restore_userptr_worker` workqueue warning. The [identical clean repeat](../../../../parity-results/qwen-flash-next/policy64-release-accuracy-clean-repeat/rocm4-dense-tp-routed-apportioned-ram-hc-up-two-term/report.json) passed required numerical groups, reset replay, driver and retirement checks. Main short-panel prefill/decode mean KL is **0.018669085 / 0.015707422**, top-1 **93.75% / 98.96%** and top-5 overlap **93.75% / 90.83%**. Heldout KL is **0.010157034 / 0.005208659**, top-1 **100% / 100%** and top-5 **95.00% / 92.92%**. This is local Release accuracy evidence, not complete preflight or image certification. The driver-rejected receipt remains rejected.

A1.1's [uninstrumented benchmark](../../../../parity-results/qwen-flash-next/policy64-tp-apportioned-benchmark/result.json) passed driver, source/runtime, resource and native-retirement checks. After one warmup, its three prefill samples were **561.362 / 562.530 / 561.867 tok/s**, and decode **30.515 / 30.532 / 30.323 tok/s**. All three generated 256-token sequences [match the current A1 policy exactly](../../../../parity-results/qwen-flash-next/policy64-tp-apportioned-benchmark/output-comparison.json). This is the same binary, prompt, arithmetic and 28-physical-core budget, with expert ownership changed; it is not a new interleaved A/B comparison. It retains a prefill lead over Castagna but does not close the decode gap or justify replacing A1.

`S62≈` is Strata's approved BF16 compatibility pack **with default tail7 expert skipping**, measured at the stdio wall boundary. It omits about 29.5% of decode expert contributions. The full-routing controls, sample ranges and exact work counts appear below; Strata has no inherited accuracy badge. Its numbers are an additional reference, with different arithmetic and prefill placement.

`F61` retains the original Castagna graph profiles; disabling graphs substantially reduced decode speed. `F62§` is the newer graphs-enabled 28-worker/NUMA1 observation, paired in A1 with corrected canonical Llaminar `L62` at the same physical host budget. Each engine's two phase medians come from its own single declared run. Earlier profiles remain visible rather than combining their best phase numbers; expert partition and host-control binding still differ.

| A1 Castagna profile | Median prefill | Median decode | Measured prefill range | Measured decode range | Evidence |
|---|---:|---:|---:|---:|---|
| Graphs enabled | 460.320 | 41.264 | 457.272–461.473 | 41.247–41.526 | [F61 enabled](../../../../parity-results/qwen-flash-next/tp4-baseline61-castagna-graphs-default/result.json) |
| Graphs explicitly disabled | 463.898 | 33.393 | 462.698–465.584 | 33.065–33.470 | [F61 disabled](../../../../parity-results/qwen-flash-next/tp4-baseline61-castagna-graphs-disabled/result.json) |
| Graphs enabled, 28 workers / NUMA1 | 467.821 | 42.959 | 466.864–471.550 | 42.866–43.374 | [F62 host28](../../../../parity-results/qwen-flash-next/tp4-baseline62-castagna-host28/result.json) |

Each profile ran one full warmup and three measured requests with 512 prompt tokens, zero cached prompt tokens, 256 output tokens and 255 after-prefill decode steps. Source/runtime identities, driver windows and requested graceful native retirement passed. The synthetic repeated-token prompt produced the same output sequence on every request. These establish throughput, not numerical model qualification. The fork's within-expert partition differs from Llaminar's M512/expert ownership. Later fresh source/native evidence corrects the previously inferred320+192 physical prefill split to checkpointed320+188+4; see the [paired stage comparison](QFN_CASTAGNA_STAGE_COMPARISON.md).

`L62` ([result](../../../../parity-results/qwen-flash-next/tp4-baseline62-llaminar-host28/result.json), [native benchmark](../../../../parity-results/qwen-flash-next/tp4-baseline62-llaminar-host28/benchmark.json)) is the corrected canonical **28-worker/NUMA1** observation with ordinary MPI bootstrap and cores/close. Its three-iteration medians are **779.493454** prefill and **29.679393** after-prefill decode tok/s, ranges **778.155–780.401** and **29.385–29.706**. These are iteration medians, not the different aggregate rate. Normal exit0, source/runtime/driver checks, MTP off and PerfStats disabled all passed. Against F62's 467.821296/42.958858, current medians show about **1.67× prefill throughput** and **31% lower decode throughput**. This is a same-hardware/GGUF/28-core-NUMA1 end-to-end observation, with the disclosed expert partition, physical prefill split and host-control binding differences; it is not an algorithm-matched kernel speedup or a win in both phases.

Historical `L61‡` ([result](../../../../parity-results/qwen-flash-next/tp4-baseline61-llaminar/result.json), [native benchmark](../../../../parity-results/qwen-flash-next/tp4-baseline61-llaminar/benchmark.json)) retains medians 779.815/29.696 tok/s and ranges 779.352–780.204/29.527–29.817. Both original launchers began with physical CPU0–55, but Llaminar's ordinary bootstrap selected NUMA1 CPU28–55/cores-close, then CLI `--threads 56` reinstated a 56-requested-worker/28-place team. The fork used 56 workers without that rebinding. That historical host mismatch cannot establish a controlled ratio and is not substituted for L62.

`F62§` explicitly requested 28 physical workers on NUMA1, with startup CPU28–55 and the fork's NUMACTL compute-mask policy. Median prefill was **467.8213** and decode **42.9589** tok/s. Requested graceful shutdown completed with exit0, no forced kill or surviving session processes, clean driver and unchanged source/runtime identities. Actual thread masks and OS NUMA residency remain attached; no strict memory-bind or identical host-thread-placement claim follows. L62 matches the physical host budget, while the fork's postcompute caller widening and Llaminar's cores/close control-thread inheritance remain distinct.

The explicit [A1 external-MPI host-policy diagnostic](../../../../parity-results/qwen-flash-next/a1-host-control62-llaminar-external/result.json) completed operationally with median **784.371918** prefill and **31.793346** after-prefill decode tok/s. Its three decode samples were **31.816953 / 31.793346 / 31.773872**, a **7.1226%** median increase over canonical L62. All three 256-token ID arrays match the canonical control; normal exit0 and source/runtime/driver checks passed. The changed policy uses unbound external MPI, taskset CPU28–55 and `OMP_PROC_BIND=false`, with the same model, arithmetic and 28-worker budget. **The main A1 row remains canonical L62: 779.493454 / 29.679393.** This diagnostic does not establish a production fix or a new full-accuracy qualification.

Saved per-thread affinity samples during measurement show **37 of 39 native threads allowed only CPU28/84** in canonical L62 (14:44:29–14:44:49 UTC), versus **37 of 39 allowed CPU28–55** in the control (14:47:59–14:48:19 UTC); the other two threads allowed CPU0–111 in both. CPU28/84 are one physical core’s SMT siblings, while CPU28–55 cover the declared 28 physical NUMA1 cores. These observations prove different scheduler eligibility persisted during timing. They do not identify each thread’s compute role, actual CPU execution, memory binding, or how much of the throughput difference is attributable to affinity alone.

The [A1 `NCCL_PROTO=AllGather:LL` control](../../../../parity-results/qwen-flash-next/a1-allgather-ll62-llaminar-host28/result.json) completed with **778.772156 prefill / 29.489001 decode tok/s**, versus canonical **779.493454 / 29.679393**. Decode samples were 29.682409 / 29.489001 / 29.334759; all three generated-token arrays exactly match the canonical run. Ordinary bootstrap, arithmetic and host budget were retained, and native retirement, source/runtime identity and driver checks passed. The override did not demonstrate a gain and is **not promoted**. This receipt records the requested override; it does not independently authenticate each native collective's selected protocol.

`F61-30` ([accepted normal A2 baseline](../../../../parity-results/qwen-flash-next/a2-baseline61-castagna-cpu28-mmap-cleanup30/result.json)) completed with median prefill **159.176510** and decode **11.600097** tok/s. Normal exit0 followed graceful cleanup in **11.6297 seconds** under the declared 30-second allowance; all source, loaded-runtime and driver checks passed. This is the complete normal throughput receipt used in A2's main row, not a promotion of the earlier diagnostic. The fork uses all CPU experts and OS-cached lazy mmap ngrams; it does not implement A2's eight GPU expert IDs or bounded adaptive zero-cache policy.

The original `F61⚠` [10-second A2 receipt](../../../../parity-results/qwen-flash-next/a2-baseline61-castagna-cpu28-mmap/result.json) is retained: all requests completed at 158.058/11.587 tok/s, but the wrapper killed the process during retirement. RSS fell from 73.46 to 22.05 to 0.90 GiB; source/runtime/driver checks stayed clean. The separate [30-second cleanup diagnostic](../../../../parity-results/qwen-flash-next/a2-cleanup61-diagnostic-30s/result.json) exited0 after **11.610 seconds** at 159.8918/11.8404 tok/s. Its scope remains **diagnostic-only, not a qualified timing receipt**, despite the subsequent normal baseline passing.

A4's [complete Castagna baseline](../../../../parity-results/qwen-flash-next/a4-baseline61-castagna-cpu28-mmap-cleanup30/result.json) passed at median prefill **48.4083349** and decode **4.61116463** tok/s. Requested graceful shutdown completed in **5.2446 seconds**, exit0 with clean driver/source/runtime checks. The [fresh Llaminar control](../../../../parity-results/qwen-flash-next/a4-baseline62-llaminar-cpu28-taskset/result.json) passed at **98.465999 / 5.796785**, with prefill samples 98.465999 / 98.119205 / 98.687710 and decode 5.806738 / 5.796785 / 5.773497. PerfStats is disabled; source/runtime/driver checks, normal exit0 and retirement passed. Every measured Llaminar sample exceeds the corresponding reference range in both phases. A4's current comparison is **2.03× prefill and 1.26× decode**, with unchanged G60 arithmetic and accuracy. CPU ISA/worker-affinity and bounded-window versus OS-cached lazy-mmap differences remain disclosed; this is engine throughput on the same hardware/model, not identical kernel algorithms or cold-SSD throughput. Historical instrumented R58 Llaminar medians were 82.806 / 5.127. The fork completed on explicit port19864 after port19863 was unavailable in TCP TIME_WAIT; that earlier probe rejected before native launch or output-directory creation and was not an engine failure or numerical red.

The [actual Castagna CUDA build](../../../../parity-results/qwen-flash-next/castagna-cuda61-build/result.json) configured for Release/SM86, then failed in 15 translation units. [Compiler evidence](../../../../parity-results/qwen-flash-next/castagna-cuda61-build/build.log) includes unguarded AMD intrinsics, wave64 shuffle code and an SM86 static shared-memory limit violation. No Castagna CUDA baseline binary was produced. The original HIP artifacts remain unchanged; an unavailable fork baseline is not a Llaminar win. The approved separate CUDA reference is official upstream llama.cpp pinned to `0a1ef309cdbb9f2024f58a38fb4c42352c5db569`. Its [Release/SM86 build](../../../../parity-results/qwen-flash-next/upstream-cuda62-build/result.json) completed in **113.53 seconds**. [Binary authentication](../../../../parity-results/qwen-flash-next/upstream-cuda62-build/authentication.json) passed the source, compiler and loader checks, including GNU OpenMP with no LLVM OpenMP or missing libraries; device discovery listed both RTX 3090s, with CUDA0 at PCI `0000:3f:00.0`. The unavailable Castagna result is retained separately.

`U62⚠` ([A3 upstream result](../../../../parity-results/qwen-flash-next/a3-baseline62-upstream-cuda-cpu28-mmap/result.json)) completed operationally with normal exit0, clean driver and unchanged source/runtime identities. Median prefill was **129.0201624** and decode **13.0586671** tok/s. The three measured prefill rates were **129.4207 / 129.0202 / 127.6413**, range **127.6413–129.4207**; decode was **17.4166 / 13.0587 / 12.2933**, range **12.2933–17.4166**. **Decode variance review is pending; no stable win is claimed**, and the older instrumented R58 Llaminar rates are not a fresh paired measurement. Placement evidence combines source-declared `cpu-moe`/cache0 with actual props, OS and NVML observations; the **full tensor map remains UNKNOWN**. Default-verbosity model-buffer logs were unavailable, and props does not expose tensor ownership. Incidental GPU contexts are not compute-placement proof. Preserve the intended whole-CPU-expert versus Llaminar 8-GPU/504-CPU and lazy-mmap versus adaptive-zero-cache distinctions without treating source intent as full native tensor-placement authentication.

**A5's exact reference is unavailable in these pins/builds.** The requested ordered placement is CUDA0 layers `[0,8)`, ROCm0 `[8,20)`, then CPU NUMA0 `[20,48)`. Both pinned parsers reject CPU entries in `--device` ([Castagna](https://github.com/benpeterson40/castagna-veloce/blob/3483d462715d61f21ed5877836be35a724446359/common/arg.cpp#L1128), [upstream](https://github.com/ggml-org/llama.cpp/blob/0a1ef309cdbb9f2024f58a38fb4c42352c5db569/common/arg.cpp#L1134)). Their layer allocators expose a CPU **prefix** followed by a GPU suffix, not this CPU tail ([Castagna](https://github.com/benpeterson40/castagna-veloce/blob/3483d462715d61f21ed5877836be35a724446359/src/llama-model.cpp#L1693), [upstream](https://github.com/ggml-org/llama.cpp/blob/0a1ef309cdbb9f2024f58a38fb4c42352c5db569/src/llama-model.cpp#L1670)). CPU tensor overrides only select weight buffers; KV and recurrent state still follow `model.dev_layer` ([KV owner](https://github.com/ggml-org/llama.cpp/blob/0a1ef309cdbb9f2024f58a38fb4c42352c5db569/src/llama-kv-cache.cpp#L216), [recurrent owner](https://github.com/ggml-org/llama.cpp/blob/0a1ef309cdbb9f2024f58a38fb4c42352c5db569/src/llama-memory-recurrent.cpp#L87)). Existing artifacts also enable only HIP or only CUDA; a mixed-runtime process is not established. This is a reference interface/build limitation, not hardware unavailability, numerical failure or a Llaminar win. Do not substitute a CPU prefix, whole-expert offload or another topology. This A5-specific finding does not decide support for the separate B–D rows.

## Candidate69: lookup-worker placement

The [CPU ngram investigation](QFN_NGRAM_CPU_PATH_INVESTIGATION.md) identified four real lookup workers restricted to one physical core. Construction now uses the existing admitted OpenMP partition and restores the caller exactly. Both CUDA/ROCm bound negative controls reproduced the old defect, both unbound controls passed, and all eight focused registrations passed after the fix. The four worker-affinity regressions are explicitly in production preflight. Complete Unit passed **754/754**; the frozen Release numerical panel passed with all **185 retained logit rows byte-identical to L66**. This is a placement correctness fix, not a measured speed win. Complete current-source production preflight remains pending.

The [unprofiled A/B/B/A comparison](../../../../parity-results/qwen-flash-next/ngram69-abba-comparison.json) retained all twelve samples, six per runtime. Control medians are **784.599102 prefill / 33.326504 decode tok/s**; candidate medians are **785.181665 / 33.237744**, or **+0.07425% / −0.26634%**. Control ranges are **784.229200–785.816661 / 33.285583–33.402426**; candidate ranges are **784.420225–786.116590 / 33.226563–33.389609**. The ranges overlap. All twelve 256-token sequences match; every arm passed driver, resource, source/runtime and normal-retirement checks. The workload remains A1: four MI50s, 28 physical NUMA1 host workers, RAM ngrams, N512/D256/M512/C4096, greedy seed42, MTP/prefix cache/PerfStats off, one warmup and three measured requests per process. Decode counts 255 tokens after prefill. No precision, table-locking, source-validation, polling or transport policy changed.

The subsequent [compact packet-import experiment](QFN_COMPACT_INTERMEDIATE_IMPORT_TUNING.md) passed exactness but produced overlapping whole-model timing ranges and remains parked. The [collective selector repair](QFN_TP4_COLLECTIVE_ATTRIBUTION.md) is green with unchanged model logits. The [ordered short-K candidate](QFN_ORDERED_SHORT_K_PROJECTION_TUNING.md) retains the original ascending-K FP32 fold and passes the component geometry/timing sweep. The 32-column production dispatch passed focused qualification and matched whole-model timing. L74 is the accepted A1 observation. The additive component attribution now separates gather, fold and launch-arrival controls; gathering dominates. The broad LL model pilot sacrifices 10.51% prefill for a descriptive 2.87% decode gain, so the next candidate targets small decode messages explicitly.

## Additional Strata reference — full routing and default approximation

The user approved a **separately labeled compatibility baseline**, including Strata's default expert-skipping approximation, on 2026-10-21. Official [Strata](https://github.com/Niko1221/Strata) is pinned at [`011c9e600a93bbc9fd451c563501b5831c32bbca`](https://github.com/Niko1221/Strata/commit/011c9e600a93bbc9fd451c563501b5831c32bbca). Separate Release builds passed for [RTX 3090 / SM86](../../../../parity-results/qwen-flash-next/strata62-build-sm86/result.json) and [MI50 / gfx906](../../../../parity-results/qwen-flash-next/strata62-build-gfx906-explicit-sdk/result.json). Actual [native help/version/families and dependency checks](../../../../parity-results/qwen-flash-next/strata62-native-discovery/result.json) also passed with a clean driver window. Every HIP compile explicitly binds the selected SDK. Its gfx906 wave64 build differs from its wave32 HIP backend.

Strata's [documented UD-Q4 compatibility pack](https://github.com/Niko1221/Strata/blob/011c9e600a93bbc9fd451c563501b5831c32bbca/docs/UNSLOTH_Q4.md) rounds 195 original Q8_0 projections to BF16, alongside exact F32-to-BF16 conversions and a runtime F32-to-F16 PLE convolution. Original GGUF shards remain unchanged. The [local conversion](../../../../parity-results/qwen-flash-next/strata62-pack-compat/result.json) completed in **30.04 seconds**, adding **1,495,822,388 bytes**, with unchanged original shard metadata and no `experts.bin`. Its manifest records **460 conversions: 264 exact and 196 rounded**, including the 195 Q8_0-to-BF16 projections and one PLE convolution narrowing; maximum recorded absolute error is about 0.0144. The local expert index preserves 43 Q4_K/Q5_1 layer pairs, four Q4_K/Q8_0 pairs and one Q5_K/Q8_0 pair. Strata accuracy must be reported under that arithmetic distinction.

Exact A1 **TP4** is unavailable in this Strata pin: its multi-GPU model execution is contiguous layer splitting with per-stage expert caches and CPU misses. A four-MI50 PP observation cannot be entered as TP4. The A2/A3 reference uses the same original expert IDs 0–7 per layer in a populated 384-slot GPU cache, with remaining expert misses computed on 28 physical CPU cores. Original experts and PLE/ngrams use mmap; the application PLE row cache is explicitly zero. Adaptation, speculative drafting, prompt/conversation caching and PCIe expert offload are disabled. Normal prefill still streams experts to the GPU; this is not CPU-only expert prefill. Full native tensor placement remains **UNKNOWN**, and exact arithmetic/partition equivalence is not claimed. Unsupported or unmatched rows retain empty Strata cells. CPU-only rows are outside this added reference request.

The separately selected policies are **full top-10** (`STRATA_ROUTE_TAIL_SKIP=0`) and **default tail7** (`=7`). Tail7 omits missed experts at zero-based rank 7 or later, contributing zero without renormalization; cached experts still execute. Every request must account for all 122,880 logical routed entries as executed lookups plus skipped entries, with zero PCIe-offloaded entries. The approximation is a valid requested performance baseline, not a Llaminar accuracy qualification or equal-work kernel comparison. Its effect is measured with all other settings unchanged.

| Device / CPU overlay | Routing policy | Median prefill | Median decode | Measured prefill range | Measured decode range | Evidence |
|---|---|---:|---:|---:|---:|---|
| MI50 + CPU28 | Full top-10 | 44.235 | 19.750 | 44.033–44.280 | 19.642–19.906 | [S62 full](../../../../parity-results/qwen-flash-next/strata62-gfx906-cpu28-mmap-static8-fullroute-r4/result.json) |
| RTX 3090 + CPU28 | Full top-10 | 73.508 | 24.791 | 72.623–73.698 | 24.520–24.859 | [S62 full](../../../../parity-results/qwen-flash-next/strata62-sm86-cpu28-mmap-static8-fullroute/result.json) |
| MI50 + CPU28 | Default tail7 approximation | 45.270 | 24.024 | 45.174–45.348 | 22.067–24.233 | [S62≈](../../../../parity-results/qwen-flash-next/strata62-gfx906-tail7-cpu28-mmap-static8-routing-r5/result.json) |
| RTX 3090 + CPU28 | Default tail7 approximation | 74.301 | 31.253 | 73.655–74.366 | 30.756–31.329 | [S62≈](../../../../parity-results/qwen-flash-next/strata62-sm86-tail7-cpu28-mmap-static8-routing-r5/result.json) |

All four receipts passed: one warmup and three measured requests per policy/backend, exact 512 input IDs and 256 emitted IDs, clean native retirement, unchanged model/source/runtime identities and clean driver windows. Full routing executed 122,880 expert lookups per request: MI50 2,383 GPU hits plus 120,497 CPU misses; RTX 3090 2,467 plus 120,413. Tail7 executed 86,616 lookups and skipped 36,264 entries on MI50; RTX 3090 executed 86,609 and skipped 36,271. These are **29.512% / 29.517%** omitted contributions across the 256 one-row verification windows. Every request's cumulative skip-counter delta accounts exactly for the logical top-10 work. Native `--spec 2` reserves verifier capacity; actual counters establish ordinary one-row execution with no drafts. All four output sequences agree within each policy/backend.

The [complete comparison](../../../../parity-results/qwen-flash-next/strata62-routing-comparison.json) retains every measured sample. Tail7 raised median decode throughput **21.64% on MI50 / 26.07% on RTX 3090** relative to the full-routing control. MI50's 22.067–24.233 tok/s approximate-policy spread remains visible; no samples were discarded. These runs establish throughput under changed arithmetic, not KL or application quality. Mmap expert reads came from the warm OS page cache; they do not measure cold SSD throughput.

**Timing boundary:** these Strata rates are stdio wall measurements: 512 divided by submission-to-first-token time, and 255 divided by first-to-last-token time. Native `prompt_ms` covers only the first 511 input tokens; native decode timing covers 256 verification windows, including the last input token. Both native counters remain in the receipt and are not renamed as 512/255 native phase timers. The main table identifies the approximation and wall scope as `S62≈`.

The pinned [native-format documentation](https://github.com/Niko1221/Strata/blob/011c9e600a93bbc9fd451c563501b5831c32bbca/docs/QUANTS.md) identifies a native MMQ candidate using the **same original UD-Q4_K_XL shards**: `STRATA_MMQ_KQ=1` enables compressed K-quant prefill instead of the default expert dequantization-to-FP16/BLAS path. S64 exercised both choices successfully on both backends. Actual target definitions and source-bound runtime selection are recorded, including CUDA's loaded `libstrata_kq.so`; the generic `STRATA_PREFILL_MMQ=OFF` cache entry does not describe the special SM86/gfx906 target definitions. This establishes selection, not an individual-kernel trace. No model download or weight requantization was needed. Original routed-expert metadata contains 120,795,955,200 weights in 77,017,907,200 bytes, **5.10069444 stored bits/weight including block scales/mins**, not a whole-model average or nominal four-bit precision. The small compatibility-pack rounding remains as disclosed above.

The separate automatic-cache variants use Strata's complete ranked expert profile, rather than the original 384-entry profile that would cap even `--expert-cache auto` at 384. Actual admitted GPU expert slots and bytes are authenticated from the native run. These larger-cache variants are **additional placements**, not replacements for A2/A3's eight-GPU-experts-per-layer rows. All eight controls use full routing, the same original GGUF, 28 physical host cores, warm OS-cache mmap, one warmup and three measured 512/256 requests. Each completed with clean drivers and normal retirement. Rates below use the same stdio wall boundaries defined above; warmups and instrumented diagnostics are excluded.

| Backend | Prefill expert arithmetic | GPU expert slots / MiB | Prefill median [min, max] | Decode median [min, max] |
|---|---|---:|---:|---:|
| MI50 | Original FP16 BLAS, static profile | 384 / 1,147 | 44.235 [44.033, 44.280] | 19.750 [19.642, 19.906] |
| MI50 | Native MMQ, static profile | 384 / 1,147 | 45.934 [45.610, 46.561] | 19.876 [18.826, 20.033] |
| MI50 | Native MMQ, automatic cache | 8,849 / 26,441 | 61.323 [61.021, 63.382] | 20.388 [20.324, 23.155] |
| MI50 | Original FP16 BLAS, automatic cache | 8,849 / 26,441 | 58.223 [57.503, 59.070] | 23.247 [21.222, 23.622] |
| RTX 3090 | Original FP16 BLAS, static profile | 384 / 1,147 | 73.508 [72.623, 73.698] | 24.791 [24.520, 24.859] |
| RTX 3090 | Native MMQ, static profile | 384 / 1,147 | 71.926 [71.700, 72.241] | 24.044 [23.913, 24.375] |
| RTX 3090 | Native MMQ, automatic cache | 5,916 / 17,672 | 86.594 [85.853, 86.894] | 28.642 [28.594, 28.756] |
| RTX 3090 | Original FP16 BLAS, automatic cache | 5,919 / 17,681 | 88.712 [88.649, 88.858] | 28.883 [28.288, 28.963] |

Automatic caching improves prefill more than the arithmetic switch. Relative to each original static control, MI50 automatic MMQ gains **38.63%** prefill and CUDA automatic FP16 gains **20.68%**. These are descriptive policy comparisons, with changed expert residency. CUDA's FP16 variant admits three more expert slots because it does not load the K-quant MMQ DSO. MI50's two automatic-cache runs have identical GPU ownership, yet both show substantial within-run decode variation; the observed median difference cannot establish an effect of prefill arithmetic on decode. All measured token sequences match within and across each backend's four controls, but this repeated-token workload does not establish a new KL or application-quality gate.

The [eight-arm comparison](../../../../parity-results/qwen-flash-next/strata64-eight-arm-comparison/comparison.json) authenticates all original receipts, work counts, source/runtime identities and actual cache placement, retaining every sample and earlier rejected attempt. Separate [CUDA](../../../../parity-results/qwen-flash-next/strata64-sm86-cpu28-mmap-fullroute-native-mmq-auto-timing/result.json) and [ROCm](../../../../parity-results/qwen-flash-next/strata64-gfx906-cpu28-mmap-fullroute-native-mmq-auto-timing/result.json) built-in timing diagnostics also completed cleanly; they are excluded from throughput statistics. On their three measured prefills, the interval labeled `dequant` accounts for **73.1–76.0% on CUDA / 62.6–65.1% on ROCm**. Source inspection of `PfTimer::mark/fold` and the MMQ branch in pinned `src/prefill/prefill.cpp` shows this label spans compressed gathers, host-fed staging and submission gaps; it does **not** mean FP16 dequantization kernels ran for that duration. Native events include queue idle gaps, and cumulative host staging overlaps those intervals. The evidence locates a dominant staging/submission interval, without establishing GPU saturation, PCIe saturation or an independently removable wall-time amount.

The earlier zero-cache admission failed because this native verifier requires a populated expert tier. A subsequent default-tail7 warmup correctly failed the then-full-routing work-count check; neither failed receipt is promoted. The initial ROCm mmap attempts also exposed SVM workqueue warnings. Both accepted ROCm runs explicitly set `GPU_PINNED_MIN_XFER_SIZE=1000000` so ROCclr uses owned staging instead of pinning the large mapped source range; driver warnings still fail the receipt. No host configuration changed. Strata selects its AVX2 CPU path because its own AVX-512 path requires VBMI, which this Xeon lacks. The ggml dependency is built with `-march=native`; source/compiler evidence also permits applicable VNNI instructions inside those quantized dot products. This is not evidence of a scalar CPU run or runtime instruction-level profiling.

## Fresh A2 and CUDA variance observations

The [fresh A2 Llaminar baseline](../../../../parity-results/qwen-flash-next/a2-baseline62-llaminar-cpu28-taskset/result.json) passed with medians **156.706631 prefill / 15.164408 after-prefill decode tok/s**. All three decode samples were 15.148439 / 15.164408 / 15.171115; prefill 156.706631 / 159.312843 / 156.700734. Native exit 0, clean driver, unchanged model/runtime identities and ordinary retirement passed. Against Castagna 159.176510 / 11.600097, this observation is about 1.55% behind in prefill and 30.73% ahead in decode. Retain the original expert-partition/ngram differences; this is not a matched-kernel claim or a win in both phases.

The preceding A2 launcher attempt exited 1 before engine startup: Open MPI interpreted `--cpu-set` as a binding policy conflicting with its PE/core request. Its [failed receipt](../../../../parity-results/qwen-flash-next/a2-baseline62-llaminar-cpu28/result.json) remains visible. The corrected private launcher uses ordinary external MPI with binding disabled, then `taskset -c 0-27` before native exec; OpenMP remains 28 workers/cores/close. An actual MPI mask probe confirmed the intended 28 physical NUMA0 CPUs. No production code or inference policy changed.

The [A3 light-telemetry diagnostic](../../../../parity-results/qwen-flash-next/a3-variance62-upstream-light-telemetry/result.json) completed cleanly with unchanged model/runtime identities and identical 256 output IDs to every corresponding original request. Its median was 129.179506 prefill / 16.808011 decode, with decode 16.808011 / 17.147634 / 16.562314. Measured sample windows recorded zero major faults, zero storage-read bytes and no GPU clock-throttling event; memory clock stayed 9501 MHz and temperature peaked 49°C. The persistent query consumed 3.65 CPU seconds over its full lifetime. This changed observer is a diagnostic, not a replacement unchanged-throughput baseline; the original 12.29–17.42 decode range remains unresolved and no stable CUDA win is claimed.

The [fresh A3 Llaminar baseline](../../../../parity-results/qwen-flash-next/a3-baseline62-llaminar-cpu28-taskset/result.json) passed with medians **180.920298 prefill / 18.231122 after-prefill decode tok/s**. Prefill samples were 183.449950 / 180.920298 / 178.482777; decode samples 18.207466 / 18.231122 / 18.316814. Its [native benchmark](../../../../parity-results/qwen-flash-next/a3-baseline62-llaminar-cpu28-taskset/benchmark.json) confirms PerfStats disabled, unchanged six-selector arithmetic, 512 fresh prompt tokens and 256 emitted tokens per iteration. Normal exit0, source/runtime/driver checks and retirement passed. This replaces the instrumented R58 row values; the gain over that historical run is not attributed to one cause. Current samples exceed both U62's original range and its light-telemetry diagnostic, but a stable paired reference lifecycle remains necessary before closing A3 as a repeatable win.

## B. Explicitly requested compositions — after A is tuned

Two CPU participants mean the two physical NUMA/socket participants, with distinct physical-core worker sets. These proposed MPI2 expert-domain compositions do not create a mixed-vendor TP group. The requested ngram tier is part of each row's identity.

| ID | Topology / proposed work placement | Ngrams | Reference engine | Accuracy | Llaminar prefill | Reference prefill | Strata prefill | Llaminar decode | Reference decode | Strata decode | Remaining prerequisite |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| B1 | 2×CUDA + 2×CPU: CUDA TP2 continuation; both CPUs own experts | RAM | llama.cpp | ⚪ PENDING | — | — | — | — | — | — | Actual MPI2 admission, owner/path proof and paired accuracy; fork equivalence unproved. |
| B2 | 2×CUDA + 4×ROCm: ROCm TP4 continuation; CUDA expert domain | **Fully GPU VRAM**, sharded on the actual ROCm lookup cohort | Mixed-vendor support unproved | ⚪ PENDING | — | — | — | — | — | — | Original-source capacity and MPI2 proof; no CPU expert or ngram tier substitution. |
| B3 | 2×CUDA + 4×ROCm + 2×CPU: ROCm TP4 continuation; CUDA and both CPUs own experts | **RAM** | Mixed-vendor support unproved | ⚪ PENDING | — | — | — | — | — | — | Actual MPI2 admission and positive ownership/work on every requested domain. |

Rank ownership and initial expert tier caps remain in the [qualification plan](QWEN_FLASH_NEXT_QUALIFICATION_PLAN.md#additional-requested-device-and-ngram-cells). They are benchmark intent, not admitted memory grants. The alternative PP composition of ROCm TP4 and CUDA TP2 is distinct and cannot stand in for B2/B3.

## C. Remaining local baseline topology coverage — queued

These rows preserve the agreed baseline-family coverage; they are not yet qualified full-model configurations. RAM ngrams are the initial control unless stated otherwise. Two-card capacity needs explicit admission: record any necessary CPU offload in the exact configuration and match it in both engines. Never silently change the GGUF or offload policy to make a row fit.

| ID | Topology / configuration | Ngrams | Reference engine | Accuracy | Llaminar prefill | Reference prefill | Strata prefill | Llaminar decode | Reference decode | Strata decode | Remaining prerequisite |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| C1 | CPU only: one NUMA participant, RAM control | RAM | Castagna | ⚪ PENDING | — | — | n/a | — | — | n/a | Qualify this storage variant independently of A4. |
| C2 | 2×ROCm: homogeneous TP2 | RAM | Castagna | ⚪ PENDING | — | — | — | — | — | — | Bind source partition and any declared CPU offload; admit capacity. |
| C3 | 2×CUDA: homogeneous TP2 | RAM | llama.cpp | ⚪ PENDING | — | — | — | — | — | — | Bind source partition and any declared CPU offload; admit capacity. |
| C4 | 4×ROCm: TP4 with the fork's within-expert F partition | RAM | Castagna | ⚪ PENDING | — | — | — | — | — | — | Exact native F slices 128/128/128/256 and live-output reduction; distinct from A1. |
| C5 | 2×ROCm: contiguous PP2, equal layer ranges | RAM | Castagna | ⚪ PENDING | — | — | — | — | — | — | Full-model admission, GPU pipeline and exact range proof. |
| C6 | 2×ROCm: contiguous PP2, unequal layer ranges | RAM | Castagna | ⚪ PENDING | — | — | — | — | — | — | Bind ranges and capacity before timing. |
| C7 | 2×CUDA: contiguous PP2, equal layer ranges | RAM | llama.cpp | ⚪ PENDING | — | — | — | — | — | — | Full-model admission, GPU pipeline and exact range proof. |
| C8 | 2×CUDA: contiguous PP2, unequal layer ranges | RAM | llama.cpp | ⚪ PENDING | — | — | — | — | — | — | Bind ranges and capacity before timing. |
| C9 | 4×ROCm: contiguous PP4, equal ranges, one stage per card | RAM | Castagna | ⚪ PENDING | — | — | — | — | — | — | Singleton-stage control for C12; full-model admission. |
| C10 | 4×ROCm: contiguous PP4, unequal layer ranges | RAM | Castagna | ⚪ PENDING | — | — | — | — | — | — | Bind ranges, source owners and capacity. |
| C11 | 4×ROCm: PP(TP2, TP2) | RAM | Castagna | ⚪ PENDING | — | — | — | — | — | — | Native homogeneous TP cohorts and pipeline boundary. |
| C12 | 4×ROCm: interleaved PP, 12 logical stages, card order 0/1/2/3 repeated three times | RAM | Castagna | ⚪ PENDING | — | — | — | — | — | — | Repeated physical ownership and concurrency-aware admission. |
| C13 | CPU prefix + CUDA suffix: partial-layer offload | RAM | llama.cpp | ⚪ PENDING | — | — | — | — | — | — | Exact layer ranges and original source placement. |
| C14 | CPU prefix + ROCm suffix: partial-layer offload | RAM | Castagna | ⚪ PENDING | — | — | — | — | — | — | Exact layer ranges and original source placement. |
| C15 | CUDA + CPU: whole-expert-tensor offload | RAM | llama.cpp | ⚪ PENDING | — | — | — | — | — | — | Match actual fork tensor overrides; differs from A3's expert-ID ownership. |
| C16 | ROCm + CPU: whole-expert-tensor offload | RAM | Castagna | ⚪ PENDING | — | — | — | — | — | — | Match actual fork tensor overrides; differs from A2's expert-ID ownership. |

## D. Additional hardware — queued

This host has four MI50s and two RTX 3090s. Six mixed-vendor GPUs do not satisfy an eight-card homogeneous case. The CUDA four-card rows retain backend symmetry for the topology families.

| ID | Topology / configuration | Ngrams | Reference engine | Accuracy | Llaminar prefill | Reference prefill | Strata prefill | Llaminar decode | Reference decode | Strata decode | Block |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| D1 | 4×CUDA: homogeneous TP4 | RAM initially | llama.cpp | ⚪ PENDING | — | — | — | — | — | — | Four CUDA devices unavailable. |
| D2 | 4×CUDA: contiguous PP4, equal layer ranges | RAM initially | llama.cpp | ⚪ PENDING | — | — | — | — | — | — | Four CUDA devices unavailable. |
| D3 | 4×CUDA: contiguous PP4, unequal layer ranges | RAM initially | llama.cpp | ⚪ PENDING | — | — | — | — | — | — | Four CUDA devices unavailable. |
| D4 | 4×CUDA: PP(TP2, TP2) | RAM initially | llama.cpp | ⚪ PENDING | — | — | — | — | — | — | Four CUDA devices unavailable. |
| D5 | 8 homogeneous GPUs: TP8 | RAM initially | Bind backend first | ⚪ PENDING | — | — | — | — | — | — | Eight same-backend devices unavailable; bind backend before execution. |
| D6 | 8 homogeneous GPUs: TP2 × PP4 | RAM initially | Bind backend first | ⚪ PENDING | — | — | — | — | — | — | Eight same-backend devices unavailable. |
| D7 | 8 homogeneous GPUs: TP4 × PP2 | RAM initially | Bind backend first | ⚪ PENDING | — | — | — | — | — | — | Eight same-backend devices unavailable. |

## Accuracy detail for the current green observations

Checkpoint60 broad-panel prompt-balanced final-logit metrics: 16 frozen histories, prefill plus six forced decode steps. Eight separate heldout histories also passed on every row. Values below are **prefill / decode**. KL is in nats; top-1 is exact agreement; top-5 is mean set intersection divided by five. N2047/D9 remains diagnostic, not long-context certification.

| Row | Mean KL | Top-1 agreement | Top-5 overlap |
|---|---:|---:|---:|
| A1 | 0.033536 / 0.032070 | 93.75% / 97.92% | 87.50% / 92.29% |
| A2 | 0.024628 / 0.032013 | 87.50% / 98.96% | 95.00% / 92.08% |
| A3 | 0.023410 / 0.020687 | 81.25% / 97.92% | 93.75% / 91.88% |
| A4 | 0.016063 / 0.022886 | 93.75% / 100.00% | 92.50% / 90.21% |
| A5 | 0.032444 / 0.024587 | 93.75% / 97.92% | 90.00% / 90.21% |

Use the frozen [numerical gates](QWEN_FLASH_NEXT_QUALIFICATION_PLAN.md#frozen-short-context-pilot-and-unseen-prompts), including every decode offset, tails and structural checks. A pooled mean alone cannot make a row green. Retain full layer/stage PyTorch parity and CSV diagnostics for failures. Accuracy must bind the same original quantized GGUF, arithmetic policy and topology as the timed Llaminar candidate.

## Measurement contract

- **Model:** the same four original `Qwen3.8-Flash-Next-UD-Q4_K_XL` shards for Llaminar and the llama.cpp references. No requantization or smaller-model substitution. The separately approved Strata compatibility pack is the explicit exception described above; its BF16 arithmetic must remain labeled and cannot inherit Llaminar's numerical qualification.
- **Initial workload:** 512 actual prompt tokens, 256 generated tokens, context capacity 4096, FP16 KV, greedy sampling with seed 42, MTP off, prompt-cache misses. Freeze ` test` repeated 512 times and authenticate its token IDs in both engines. Longer contexts receive their own accuracy/performance entries.
- **Llaminar arithmetic:** HC-up TwoTermQ8, FP16 router operands with FP32 accumulation, FP16 MoE alpha/beta; ordinary wide-projection Q8 activation math retained. Record all six selectors. Precision changes cannot inherit another policy's green result.
- **Timing:** unprofiled Release, initialization separate. Explicitly match warmup/repetitions and retained graph state. Compare Llaminar `decode_after_prefill` to the corresponding fork interval: 256 emitted tokens contain 255 after-prefill decode steps. Retain every sample, dispersion and token sequence; a noise-sized median lead is not a demonstrated win.
- **Placement:** match hardware, CPU ISA/affinity/workers, expert/layer placement, ngram residency and cache policy. Different internal partitions must be disclosed as a same-hardware/model end-to-end comparison; they cannot establish a kernel-only speed advantage.
- **Fork pin:** Castagna Veloce `3483d462715d61f21ed5877836be35a724446359`. Qualify explicit graph-enabled and graph-disabled candidates separately where applicable; never recover by silently changing the topology or execution policy. Mixed-vendor/expert-ID support remains unproved until exercised. Unsupported fork configurations need evidence, not invented scores or automatic wins.
- **Row completion:** current green accuracy and repeatable wins in **both** phases against every applicable measured engine on the declared workload. A supported but unmeasured engine comparison remains pending. Record prefill/decode verdicts separately when only one wins. Preserve failures and earlier measurements instead of selecting favorable samples.

`R58†` values are TwoTerm medians of six measurements across two process lifecycles in an ABBA policy experiment, with one warmup and three measurements per lifecycle and 28 physical-core workers. Native records show `perf_stats.enabled=true`: these are prior policy measurements, not the final uninstrumented cross-engine baseline. Fresh baseline work also reauthenticates per-request cache misses; process-wide cache counters can include setup/warmup. Replace these numbers only after the fork baselines are collected.

## Measured performance checkpoints

Checkpoint each accepted win on `feat/qwen-flash-next` with its settings, speeds,
accuracy scope and remaining qualifications. The user authorized commit and push
with `--no-verify` for these incremental checkpoints. This does not turn focused
accuracy evidence into complete production-preflight or image certification.
Keep experimental candidates separate until their own comparisons pass.

| Checkpoint | Change | Paired control prefill / decode | Candidate prefill / decode | Change prefill / decode | Evidence |
|---|---|---:|---:|---:|---|
| L64 | Individual prefill and grouped-verifier transfers; batched decode endpoints | 780.089 / 29.532 | 779.529 / 31.128 | −0.072% / +5.406% | [Four clean arms, twelve equal output sequences](../../../../parity-results/qwen-flash-next/a1-phase64-abba-comparison.json) |
| L66 | Construct independent GPU-control workers on the admitted OpenMP partition | 779.091 / 31.662 | **784.382 / 33.376** | **+0.679% / +5.413%** | [Four clean arms, twelve equal output sequences](../../../../parity-results/qwen-flash-next/affinity66-abba-comparison.json) |
| L74 | Parallel short-K contributions with the original ordered TwoTerm fold | 784.699 / 33.283 | **785.397 / 33.882** | **+0.089% / +1.797%** | [Four clean arms, twelve equal output sequences](../../../../parity-results/qwen-flash-next/shortk74-abba-comparison.json); [settings, accuracy and resources](QFN_ORDERED_SHORT_K_PROJECTION_TUNING.md) |
| L84 | Explicit ordinary vocabulary sharding with MTP disabled and zero retained capacity | 785.752 / 34.627 | **786.235 / 35.056** | **+0.061% / +1.237%** | [Four clean arms, twelve equal output sequences](../../../../parity-results/qwen-flash-next/vocabulary84-model-abba-comparison.json); [settings, final-source85 accuracy and regressions](QFN_TP4_COMPUTE_OWNERSHIP_AUDIT.md#experiment84-measured-ordinary-m1-head-sharding) |
| L90 | Explicit compiled Four-worker execution of original owner gate/up K partitions | 785.821 / 34.982 | **785.806 / 36.780** | **−0.002% / +5.142%** | [Four clean arms, twelve equal sequences](../../../../parity-results/qwen-flash-next/moe-model90-release-abba-comparison-r1); [all-format resources, accuracy and settings](QFN_ROUTE_OWNED_MOE_TUNING.md#matched-release-model-throughput-l90) |
| L95 | Query-covered K/V storage and original-output-slice projections | 786.251 / 36.746 | **791.519 / 36.759** | **+0.670% / +0.034%** | [Four accepted arms, twelve equal sequences](../../../../parity-results/qwen-flash-next/query-kv95-model-comparison.json); [settings, numerical proof and rejected attempt](QFN_TP4_KV_COVERAGE_PLAN.md#matched-release-throughput-l95). Decode is effectively unchanged. |
| A94 | Automatic selects the qualified Four-worker program on top of L95 | 791.084 / 36.668 | **791.635 / 36.645** | **+0.070% / −0.064%** | [Four clean arms, twelve equal sequences](../../../../parity-results/qwen-flash-next/four-automatic94-model-comparison.json); [default scope and qualification](QFN_ROUTE_OWNED_MOE_TUNING.md#automatic-four-worker-policy-qualification). Performance is effectively unchanged. |
| L96 | Preserve the original floating reduction tree with a wave64 tail | 791.395 / 36.654 | **796.370 / 36.750** | **+0.629% / +0.260%** | [Four clean arms, twelve equal sequences](../../../../parity-results/qwen-flash-next/reduction96-model-comparison.json); [accuracy and isolated profiles](QFN_FLOATING_REDUCTION_TUNING.md). Decode ranges overlap. |
| L97 | Specialize native FP16/BF16 source encoding outside the K loop | 796.421 / 36.812 | **814.368 / 36.727** | **+2.254% / −0.229%** | [Four clean arms, twelve equal sequences](../../../../parity-results/qwen-flash-next/x16-97-model-comparison.json); [accuracy, components and profiles](QFN_FLOATING_REDUCTION_TUNING.md#native-source-format-specialization-candidate97). Decode ranges overlap. |
| L100 | Group ordinary direct-output ROCm M1 reducers; retain producers and original ordered FP32 fold | 814.133 / 36.824 | **814.527 / 37.063** | **+0.048% / +0.648%** | [Four clean arms, twelve equal sequences](../../../../parity-results/qwen-flash-next/reducer100-model-comparison.json); [185 exact rows](../../../../parity-results/qwen-flash-next/reducer100-release-accuracy-r1/logit-byte-comparison.json); [components and profile](QFN_TP_COMPUTE_SCALING.md#grouped-ordinary-m1-reducers-l100). Decode ranges do not overlap; G60 remains the latest complete preflight. |
| L101 | Original-tree HIP branch norm: nine barriers to three | 814.645 / 37.103 | **814.610 / 37.088** | **−0.004% / −0.041%** | [Four clean arms, twelve equal sequences](../../../../parity-results/qwen-flash-next/branch-norm101-model-comparison.json); [185 exact rows](../../../../parity-results/qwen-flash-next/branch-norm101-release-accuracy-r1/logit-byte-comparison.json); [component and profile evidence](QFN_TP_COMPUTE_SCALING.md#branch-normalization-original-reduction-order-with-fewer-barriers). Component M1 latency improves 2.794%; model throughput is unchanged. |
| L102 | Immutable custom floating addresses; retire pointer-table setup | 814.156 / 37.134 | **815.053 / 37.424** | **+0.110% / +0.781%** | [Four clean arms, twelve equal sequences](../../../../parity-results/qwen-flash-next/pointer102-model-comparison.json); [185 exact rows](../../../../parity-results/qwen-flash-next/pointer102-release-accuracy-r1/logit-byte-comparison.json); [component/profile and cold-capture regressions](QFN_TP_COMPUTE_SCALING.md#floating-projections-immutable-captured-bindings-l102). Both roles carry the same cold hipBLAS workspace fix. |

Each percentage uses its own interleaved A/B/B/A comparison. Do not multiply
percentages or compare unmatched process lifecycles to claim a cumulative gain.
Each arm runs one warmup and three measured requests; the table uses six samples
per role. Runtime/model identities, driver windows and native retirement passed.
Raw result directories stay ignored; this table and the settings below are the
versioned checkpoint record.

The L64/L66/L74 benchmark contract is:

| Setting | Exact measured selection |
|---|---|
| Hardware / build | Four MI50 32 GiB, unprofiled Llaminar `Release`; build targets CUDA SM80/86/89/90 and ROCm gfx906 with spill guards enabled |
| Model | Original four `Qwen3.8-Flash-Next-UD-Q4_K_XL` GGUF shards; shared `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` supplied but MTP disabled |
| Prompt / generation | UTF-8 bytes `b" test" * 512`, no newline; 512 copies of token ID 1228; 256 emitted tokens, 255 `decode_after_prefill` steps |
| Prompt identity | 2,560 bytes; SHA-256 `d17afaa1069c1800ba2db70c90cc82ad85c213ec245d07e6916bf089fe06fe21` |
| Context / graph bucket | `-c 4096 --prefill-max-bucket-size 512`; retained production graphs |
| Topology | One ordinary self-bootstrapped MPI process, rank-local RCCL, original ROCm devices 0/1/2/3, equal 0.25 participant weights |
| Dense / expert policies | `tensor-parallel` + `gate-up-owned-down-columns`; uniform routed phase, ordinal owner order, static-by-ID residency, static-owner assignments for prefill and decode |
| Domains / tiers | Continuation, base and shared domains all `qfn_rocm_full_tp`; one all-GPU expert tier, priority 0, automatic memory, no expert-count cap |
| Ngram / caches | `--ngram-storage ram`; mmap window flags 65,536 bytes × 1,024 retained in the command; `--no-prefix-cache` |
| Precision | `--activation-precision fp32 --kv-cache-precision fp16`; six projection selectors listed below |
| Sampling / maintenance | `--temperature 0 --top-k 1 --top-p 1 --seed 42`; `--no-mtp --mtp-depth-policy fixed --moe-residency-maintenance off` |
| Host budget | Outer `taskset -c 28-55`, `--threads 28`; NUMA1 physical cores. The canonical MPI bootstrap installs GPU-local `OMP_PLACES=cores`, `OMP_PROC_BIND=close`; actual worker masks are retained. No strict memory-bind claim |
| Timing | `LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1`, `LLAMINAR_BENCHMARK_ITERATIONS=3`; native phase timing, no profiler or PerfStats instrumentation in the timed run |

L84 retains the same workload, arithmetic and worker settings. It uses the
accepted HIP76/RCCL80 dependency closure and adds explicit
`--mtp-graph-capacity-draft-tokens 0` to both arms. The sole comparison variable
is `--mtp-terminal-head-policy mirrored-full-vocabulary` versus
`--mtp-terminal-head-policy vocabulary-sharded`, with `--no-mtp` retained.
The latter is the current A1 tuning configuration; it does not change Auto.
The final-source85 run repeats the canonical 185-row accuracy panel after the
capacity/ABI follow-up; the throughput table retains the actual Release84
archive observations rather than attributing a new timing run to source85.

L90 retains the L84 candidate settings, including explicit vocabulary sharding
and zero MTP capacity, on both sides. The sole compiled comparison variable is
`LLAMINAR_ROCM_MOE_M1_OWNER_PARTITIONS=serial` versus `four_worker_planes`.
Separate Release archives from the same source retain their original native
metadata/dependency receipts. The option affects only the owning MoE C++
translation unit; it is not a serving environment variable. Complete Unit and
focused device checks pass, and all 185 retained logits remain byte-identical.
The explicit build qualified the measured QFN workload. A94 subsequently
installs Automatic selection for that bounded operation/domain, retaining
explicit modes and established policies elsewhere.

L95 retains the L90 Four candidate configuration and dependencies on both sides.
It compares the retained Four90 source/runtime with current QueryInterval KV95;
this is a cross-revision layout comparison. The [KV record](QFN_TP4_KV_COVERAGE_PLAN.md#matched-release-throughput-l95)
retains the complete settings and failed-attempt disposition.

The six arithmetic selectors were explicitly frozen for both arms:

```text
LLAMINAR_QFN_FA_OUTPUT_ACTIVATION=single_q8
LLAMINAR_QFN_GDN_OUTPUT_ACTIVATION=single_q8
LLAMINAR_GDN_PROJECTION_ARITHMETIC=fp16_moe_alpha_beta
LLAMINAR_QFN_GDN_QKV_ACTIVATION=single_q8
LLAMINAR_QFN_HC_UP_ACTIVATION=two_term_q8
LLAMINAR_MOE_ROUTER_ARITHMETIC=fp16
```

L66 passed **754/754 Unit registrations**, all **11 focused registrations**, and
the unchanged frozen A1 required accuracy groups. Short-panel mean KL is
**0.033535869 prefill / 0.032070057 decode**, top-1 **93.75% / 97.92%**, top-5
**87.50% / 92.29%**. All **26 final-logit files / 185 rows**, including diagnostic
and reset rows, are byte-identical to L64. The canonical accuracy run is unbound;
the separate timing comparison proves bound-bootstrap output-token equality.
The checkpoint also contains the tested named GDN handoff workspace admission
repair, focused fixture fixes and explicit policy-comparison regressions.
**A1.2 remains numerically red; full current-source production preflight remains
pending.** The private compact-import and scheduler-ticket experiments are not
part of this checkpoint.

## CPU ngram follow-up

The [N68 host diagnostic](../../../../parity-results/qwen-flash-next/ngram68-host-service-diagnostic-global/result.json)
used the unchanged archived L66 runtime and passed driver, retirement and
source/runtime checks. Actual worker TIDs joined to passive scheduler masks
prove that all four RAM lookup services share physical core 28 (`{28,84}`),
despite the earlier GPU-control placement repair. The trace accounts for every
live gather, including explicit prefill preparation; measured decode gathers
have a 45.006 µs median and 146.648 µs p95 under instrumentation. These timings
do not isolate the model critical path and do not enter the throughput table.

Candidate69 corrected construction-time eligibility of these I/O workers using
the existing OpenMP partition authority. Worker/lifecycle regressions and exact
logit comparisons passed. Its [paired timing](../../../../parity-results/qwen-flash-next/ngram69-abba-comparison.json)
is effectively unchanged, so retain the placement correction without claiming a
throughput win. Polling, metadata authentication, table locking, native bytes and
inference arithmetic remain fixed. See the [investigation follow-up](QFN_NGRAM_CPU_PATH_INVESTIGATION.md#measured-follow-up-host-worker-placement-2026-10-21)
for component timings, counts and the rejected initial probe attachment.

## Compact-import follow-up

The [candidate67 investigation](QFN_COMPACT_INTERMEDIATE_IMPORT_TUNING.md) retains
all failed and corrected CUDA evidence, 21/21 focused passes, 754/754 Unit passes,
Release compiler resources and unchanged numerical results. All 26 files / 185
final-logit rows are byte-identical to checkpoint69. The captured pack/import
component takes 28.26% less time on ROCm and 37.07% less on CUDA, but the complete
[model comparison](../../../../parity-results/qwen-flash-next/compact67-cuda-fixed-after69-abba-comparison.json)
is effectively unchanged: control **785.062 / 33.325 tok/s**, candidate
**784.594 / 33.389**, with overlapping sample ranges and twelve identical output
sequences. This experiment did not replace L69; the later independent short-K
L74 improvement is now the accepted dashboard record. The compact-import
experiment is preserved without claiming an end-to-end win. Fresh attribution must use the accepted
runtime; historical L62 traces cannot rank current exposed costs.

## Remaining qualification work

The optional all-format performance sweep stopped after 19 completed CUDA M1 formats because its Q8_1 fixture called the generic tensor factory instead of the existing typed raw-byte Q8_1 factory. Production Q8_1 correctness coverage passed; this was a separate performance-fixture defect. The narrow saved-source construction repair is now **applied and tested**: both focused fixture targets rebuilt, the new explicit preflight alias passed in **0.29 seconds**, and [all five Q8ResidualAllFormats tests](../../../../parity-results/qwen-flash-next/q8-original-source62-focused.xml) passed in **10 milliseconds**, covering all 21 formats and exact saved-source bytes. The subsequent [complete Unit run](../../../../parity-results/qwen-flash-next/unit62-fixture/result.json) passed **754/754**, zero failures or skips, in **116.43 seconds** ([JUnit](../../../../parity-results/qwen-flash-next/unit62-fixture/junit.xml)). Preserve the earlier component observations; the optional full component performance sweep remains pending. These focused and Unit passes do not imply that the sweep or a new complete production-preflight gate ran.

The shared projection fixture now includes mixed original Q4_K/Q5_1 and Q4_1/Q5_1 sources, plus explicit ordinal membership and A1's P4/E512/H2560/F640/top-10/M1 geometry in a separate synthetic performance target. The [six focused native/counted projection regressions](../../../../parity-results/qwen-flash-next/a1-projection62-focused/result.json) passed on CUDA2, ROCm2 and ROCm4 in **194.90 seconds**, with a clean driver window. These retain all-format byte checks and adversarial ownership/replay coverage. The first [exact-geometry performance invocation](../../../../parity-results/qwen-flash-next/a1-projection62-q4k-q51/result.json) exhausted its existing 120-second budget during setup, before producing timing samples. Parallel source generation reduced that phase to **13.441 seconds**, but the [second invocation](../../../../parity-results/qwen-flash-next/a1-projection63-q4k-q51/result.json) still timed out during cold GPU preparation, without timing samples. Each of 2,048 matrices per participant was constructing a separate loader, pool and three staging streams. The applied fixture-only batch retains every original matrix, prepared handle, full-down oracle and 20 adversaries while using one existing production loader/pool lifecycle per device. All eight projection variants passed in the affected follow-up below. The [renewed exact-geometry performance invocation](../../../../parity-results/qwen-flash-next/a1-projection63-batched-setup-q4k-q51/result.json) passed in **22.08 seconds** with the original 120-second timeout, clean drivers and normal retirement; matrix preparation took about 0.6 seconds per device. Its synthetic counted pipeline measured **192.161 microseconds median**, p10/p90 **182.080 / 210.401**, versus **246.401 microseconds** for its unsharded compute oracle. Bytes and useful packet extents passed. This is not an old/new endpoint-batching comparison, and synthetic source weights cannot certify model accuracy or model throughput. The other exact-geometry source-format performance cases remain pending.

The first production A1 candidate batches up to three independent mapped-transfer endpoints per native launch, retaining each original channel, live extent, cursor and publication. For A1 this targets **288 → 96 endpoint launches per GPU per model graph**, with unchanged endpoint CTAs and payload bytes. The [initial focused run](../../../../parity-results/qwen-flash-next/batch63-transfer-routing-projection-focused/result.json) completed **32/35 passes** in 180.30 seconds with clean drivers and no surviving native processes. All five new batching aliases, existing transfer checks, routing tests and counted projections passed. Three legacy native projection fixtures failed before execution because parallel source setup relied on CPU inventory publication performed only by counted transport. Their common entry now acquires the existing canonical inventory. The [affected follow-up run](../../../../parity-results/qwen-flash-next/batch63-fixture-and-mixed-retest/result.json) passed **13/13** in 146.43 seconds, with clean drivers, exit0 and complete native retirement. It includes additional mixed-capacity batch replay proving unequal per-endpoint copy grids and a final fused partial packet on all four backend orders. Together these focused runs resolve the observed failures; they are not a new complete preflight gate.

The [A1 model diagnostic](../../../../parity-results/qwen-flash-next/a1-batch63-accuracy-diagnostic/report.json) passed the unchanged required numerical groups, native retirement, driver checks and reset proof. All **26 final-logit files / 185 rows** are [byte-identical to checkpoint60](../../../../parity-results/qwen-flash-next/a1-batch63-accuracy-diagnostic/checkpoint60-logit-byte-comparison.json): 178 measured rows across short, heldout and N2047/D9 histories, plus seven reset rows. This is a supported local diagnostic with current prerequisites explicitly unavailable, not a new complete-preflight or image qualification. Production arithmetic is unchanged.

The rebuilt [Release A1 diagnostic](../../../../parity-results/qwen-flash-next/a1-batch63-release-accuracy-diagnostic/report.json) also passed required groups, reset, driver and retirement checks. Its same 26 logit files are [byte-identical to checkpoint60](../../../../parity-results/qwen-flash-next/a1-batch63-release-accuracy-diagnostic/checkpoint60-logit-byte-comparison.json). Short-panel prefill/decode mean KL remains **0.033535869 / 0.032070057**, top-1 **93.75% / 97.92%** and top-5 overlap **87.50% / 92.29%**. This qualifies the focused numerical observation on the production build; the complete prerequisite gate remains pending.

The candidate built for CUDA SM80/86/89/90 and gfx906 with the spill guard enabled. Static inspection found no new spill/private storage; fused register use increases from 46–48 to 54 on [CUDA](../../../../parity-results/qwen-flash-next/batch63-static-resources/cuda/REPORT.md) and from 48 to 51 VGPRs on [gfx906](../../../../parity-results/qwen-flash-next/batch63-static-resources/rocm/REPORT.md) (52 allocated). These are Integration-object compiler-resource observations, not measured occupancy or authenticated Release code-object measurements. The original Release executable/core are archived with verified loader relocation, and the [candidate source snapshot](../../../../parity-results/qwen-flash-next/batch63-candidate-source/receipt.json) is retained. Release rebuild, loader checks and the numerical diagnostic completed successfully.

The [complete model timing comparison](../../../../parity-results/qwen-flash-next/a1-batch63-abba-comparison.json) records A/B/B/A plus an explicit replacement control after the original closing control failed its driver check. Each process ran one warmup and three measured 512/256 requests under unchanged model, arithmetic, MPI bootstrap and host budget. All **15 measured output-token arrays are identical**, including the rejected control. All samples and failures remain attached.

| Batch63 timing lifecycle | Median prefill | Median decode | Status |
|---|---:|---:|---|
| [A1 archived control](../../../../parity-results/qwen-flash-next/a1-batch63-abba-a1-control/result.json) | 778.427 | 29.883 | Passed |
| [B1 batching](../../../../parity-results/qwen-flash-next/a1-batch63-abba-b1-candidate/result.json) | 755.948 | 31.429 | Passed |
| [B2 batching](../../../../parity-results/qwen-flash-next/a1-batch63-abba-b2-candidate/result.json) | 755.081 | 31.047 | Passed |
| [A2 original closing control](../../../../parity-results/qwen-flash-next/a1-batch63-abba-a2-control/result.json) | 780.130 | 29.606 | Rejected: driver workqueue warning; inference and normal retirement passed |
| [A3 explicit replacement control](../../../../parity-results/qwen-flash-next/a1-batch63-abba-a3-control-retry/result.json) | 779.493 | 29.567 | Passed |

Across all six accepted samples per policy, control medians are **778.812130 / 29.768184**, versus batching **755.514294 / 31.233908**: **2.991% lower prefill and 4.924% higher decode**. Decode ranges are 29.552–30.002 control versus 30.810–31.898 batching; prefill ranges are 777.142–780.378 versus 754.056–757.315. This establishes the observed trade-off on this workload, not a statistical confidence bound or a new full-preflight qualification. The rejected A2 warning named `amdgpu_amdkfd_restore_userptr_worker`; it occurred at approximately 17:23:44 UTC during graph materialization, before warmup at 17:23:46.929 and timed iterations at 17:23:57.133. No host configuration changed, and the warning is not relabeled as a pass.

The requested next candidate freezes scheduling from the graph's existing semantic phase: **Prefill → individual endpoints; Decode → batched endpoints; GroupedVerifier → original individual scheduling until separately tuned.** It retains the same channel owners, event edges, exact live byte extents and arithmetic. Phase64 source and four focused preflight aliases are applied. The [complete Integration build](../../../../parity-results/qwen-flash-next/phase64-integration-build/result.json) passed in **631.20 seconds**, rebuilding the Unit/preflight target closure and model canary. All [20 focused regressions](../../../../parity-results/qwen-flash-next/phase64-transfer-routing-projection-focused/result.json) passed in **129.56 seconds**: equal-row phase selection and actual captured-node proof on CUDA2/ROCm2/ROCm4, existing routing and counted projections, and all four backend endpoint directions. Drivers and native retirement were clean. The [Release build](../../../../parity-results/qwen-flash-next/phase64-release-build/result.json) passed in **161.08 seconds**. The completed paired timing below supplies L64†; the complete G60 qualification remains the latest full prerequisite checkpoint.

The [phase64 Release numerical diagnostic](../../../../parity-results/qwen-flash-next/a1-phase64-release-accuracy-diagnostic/report.json) passed the unchanged required groups, actual path/admission checks, reset proof, driver window and retirement. All [26 final-logit files / 185 rows](../../../../parity-results/qwen-flash-next/a1-phase64-release-accuracy-diagnostic/checkpoint60-logit-byte-comparison.json) are byte-identical to checkpoint60, including diagnostic boundary and reset rows. Short-panel prefill/decode mean KL therefore remains **0.033535869 / 0.032070057**, top-1 **93.75% / 97.92%**, top-5 overlap **87.50% / 92.29%**. This directly tests the Release candidate being timed; a duplicate Integration model run was deferred after the complete Unit and focused native checks passed. Complete production preflight remains required before a new certified checkpoint.

The [phase64 A/B/B/A comparison](../../../../parity-results/qwen-flash-next/a1-phase64-abba-comparison.json) completed all four arms successfully, with clean drivers, unchanged runtime/model metadata and normal retirement. All twelve measured 256-token arrays are identical. Six samples per role give control **780.089 prefill / 29.532 decode** versus candidate **779.529 / 31.128 tok/s**: **−0.072% prefill / +5.406% decode**. Prefill ranges overlap (control 778.578–780.555, candidate 778.039–781.262); decode ranges separate (29.261–29.843 versus 30.709–31.433). The candidate recovers the earlier unconditional-batching prefill loss while retaining the decode gain. This is a measured improvement, still below Castagna's 42.959 decode target. No arm was excluded or replaced, and the result does not imply complete preflight or image qualification.

The [complete batch63 Unit invocation](../../../../parity-results/qwen-flash-next/unit63-endpoint-batch/result.json) finished with **741/754 passing and 13 failures**, with a clean driver window. The failed executables predate the added backend virtual method and load the newly rebuilt core: inspected old MockBackend vtables have 0x450 bytes versus 0x458 in the rebuilt target. After the complete target rebuild, the [fresh phase64 Unit suite](../../../../parity-results/qwen-flash-next/unit64-phase-aware-transfer/result.json) passed **754/754** in **120.00 seconds**, with clean drivers and normal retirement. This resolves the stale executable/core ABI mixture without relaxing assertions or inventing a production repair. The failed receipt remains retained; a new complete production preflight is still pending.

## Evidence and update rules

- [Accuracy checkpoint60](../../../../parity-results/qwen-flash-next/accuracy-checkpoint60/result.json). Each green cell links its original report and exact canonical case identity.
- [Release checkpoint58 comparison](../../../../parity-results/qwen-flash-next/hc-default58-model-abba-r2/comparison.json): all samples, token equality, resource observations and native retirement. Raw results stay ignored, outside versioned source payloads.
- [Qualification plan](QWEN_FLASH_NEXT_QUALIFICATION_PLAN.md), [implementation record](QWEN_FLASH_NEXT_IMPLEMENTATION.md), [ngram tier plan](QFN_NGRAM_TIERED_STORAGE_PLAN.md).
- Canonical topology/pair authority: [Qwen4ExpLocalPairedCases.h](../../../../tests/v2/integration/parity/qwen4exp/Qwen4ExpLocalPairedCases.h). This dashboard is a reporting view, not a runner registry.
- Stable procedure: [llama.cpp comparison skill](../../../../.agents/llama-cpp-comparison/SKILL.md).

Keep applicable RAM, SSD mmap, bounded RAM/SSD, replicated VRAM, sharded VRAM and pooled VRAM/RAM/SSD variants visible. VRAM has highest placement priority; ngrams have lowest VRAM admission priority. Test positive-capacity hot-row promotion/eviction/reset. Add a separate variant row before measuring a different storage policy, context, ISA or arithmetic policy; do not combine them in one timing cell.

TP/PP coverage retains even/skewed/legal-empty ownership, reversed device order, singleton/full-cohort boundaries, equal/unequal ranges and actual live communication checks. Unsupported combinations reject admission explicitly; family coverage does not authorize an invalid Cartesian product.

After every complete baseline, accuracy run and tuning experiment, update the row with the exact case ID, model/build revisions, effective configuration, corpus/workload, grants, commands and artifact links. Replace historical `G60`/`R58†` only with identified newer evidence. If a candidate fails accuracy, mark it red immediately rather than keeping an older green badge next to its new speed.

[a1]: ../../../../parity-results/qwen-flash-next/accuracy-checkpoint60/rocm-tp4-allgpu-ram-hc-up-two-term/report.json
[a2]: ../../../../parity-results/qwen-flash-next/accuracy-checkpoint60/rocm-cpu-static8-adaptive-hc-up-two-term/report.json
[a3]: ../../../../parity-results/qwen-flash-next/accuracy-checkpoint60/cuda-cpu-static8-adaptive-hc-up-two-term/report.json
[a4]: ../../../../parity-results/qwen-flash-next/accuracy-checkpoint60/cpu-mmap-hc-up-two-term/report.json
[a5]: ../../../../parity-results/qwen-flash-next/accuracy-checkpoint60/hybrid-cpu-tail-mmap-hc-up-two-term/report.json
