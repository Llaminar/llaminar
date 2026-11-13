# QFN QSA preparation fusion

## Component screen, 2026-10-25

The candidate combines independent pooled-key and query-head preparation into
one captured launch. Both roles retain their original arithmetic, reduction
order, representation rounding and addresses. It does not remove query CTAs:
M1 still launches five 128-thread blocks. Scores and selection are unchanged.
Cold validation rejects cross-role write/read aliases before recording work.
No buffer, admission ledger, transfer or synchronization is added.

The coherent Release build covers gfx906 and CUDA sm80/86/89/90. The paired
screen retains frozen old and candidate executable/core/source closures and
runs each cell control/candidate/control on the same MI50. Each arm checks
membership before and after native-event timing, then exports actual pools,
prepared queries, scores, selected IDs and counts for exact comparison.
Both AllLive and RankingOnly demand are tested, including poisoned skipped
query rows. These are synthetic component measurements, not model throughput.

All **26 cells / 78 accepted native arms** have identical terminal bytes.
Coverage is M1 at 512, 768, 2051, 2052, 4096 and 65536 tokens; M16 at 768, 2052,
4096 and 65536; and M512 at 512, 4096 and 65536, with both demand policies.
The complete indexer retains three kernels per transaction instead of four.

| Rows / context / demand | Original, µs | Fused, µs | Time reduction |
|---|---:|---:|---:|
| 1 / 512 / RankingOnly | 12.0411 | 9.195635 | 23.63% |
| 1 / 768 / RankingOnly | 12.0936 | 9.26031 | 23.43% |
| 1 / 2052 / RankingOnly | 48.540475 | 43.2241 | 10.95% |
| 1 / 65536 / RankingOnly | 636.30075 | 629.6075 | 1.05% |
| 16 / 768 / RankingOnly | 12.749675 | 9.81375 | 23.03% |
| 16 / 65536 / RankingOnly | 1053.810 | 1046.960 | 0.65% |
| 512 / 512 / AllLive | 47.017525 | 48.5799 | −3.32% |
| 512 / 65536 / AllLive | 22456.375 | 22462.3 | −0.03% |

The short prefill loss motivates retaining the original prefill preparation
program and selecting fusion for grouped decode. That follow-up still needs
its own dispatch, correctness and economy qualification. Neither this screen
nor the prepared-Q/K GDN component result changes checkpoint196's model score.

The [aggregate](../../../../parity-results/qwen-flash-next/qsa-preparation-fusion-matrix-summary-r1/result.json)
retains all accepted arms and the two failed collection attempts. One arm was
interrupted by the quiet-host guard when a formatter ran. Another native pass
was rejected by a reader's fixed 0.02 µs normalization allowance, which did not
cover independently printed six-significant-digit values at large durations.
A separate reader uses print-rounding intervals and FP32 division bounds,
checks both transaction and row normalization, and requires exact complete
native signatures. Its positive/negative tests pass; it also authenticates
the previously accepted logs. Original receipts are preserved. Only unfinished
cells were rerun, with fresh outputs.

## Compiler and isolated profiler evidence

Both new gfx906 programs use 20 VGPRs, 512 bytes of LDS and 128 threads, with
zero private storage and zero register spills. RankingOnly uses 40 SGPRs;
AllLive uses 29. CUDA's final linked programs retain the original 32-byte
`STACK` allocation and `LOCAL:0`; sm80/86/89 use 26 registers and 512 bytes of
shared memory. Sm90 uses 28/30 registers and 1536 bytes of shared memory.
Object-only CUDA shared/stack fields are unresolved before device linking;
the retained final-linked metadata owns those figures. Static resources do
not establish achieved occupancy. `STACK` is not a register-spill counter or
proof of call/return-only storage: retained SASS includes local accesses without
call instructions. The canonical compiler spill guard remains authoritative.

All **16 isolated ROCm counter launches** and their strict joins pass. Four
shapes/policies each collect Wavefronts, SALUInsts, VALUInsts and FetchSize in
separate processes. Each uses one complete operation per graph, the initial
checked replay, five warmups and twelve samples. The seventh preparation
execution is measured; all 18 graphs and 54 dispatches are authenticated.
This R1 diagnostic schedule is separate from amortized timing above.

| Rows / context / demand | Waves | SALU/wave | VALU/wave | FetchSize, KiB |
|---|---:|---:|---:|---:|
| 1 / 512 / RankingOnly | 10 | 171.6 | 83.8 | 9.4375 |
| 1 / 512 / AllLive | 10 | 132.4 | 111.8 | 13.625 |
| 1 / 4096 / RankingOnly | 10 | 239.6 | 162.2 | 9.875 |
| 512 / 512 / AllLive | 4354 | 123.609554 | 106.249885 | 1281.375 |

These counters describe one physical dispatch, not request latency or logical
tensor traffic. Profiled event values are excluded from economy results.
Evidence is in `qsa-preparation-fusion-resources-r3` and
`qsa-preparation-fusion-profiles-r1`. CUDA hardware profiling remains stopped
after the previously recorded driver assertions; this screen uses final
linked resources and SASS inspection, not an unqualified counter claim.

Full Unit, affected native regressions, the canonical model panel and matched
Release throughput remain required before promotion.

## Selected production candidate

The applied follow-up selects preparation from immutable graph row capacity:
the existing grouped-verifier range, M1–16, uses the fused three-node indexer;
larger capacities use the original four-node prefill program. Both CUDA and
ROCm consume the same typed selection. Live row counts cannot change this
choice during replay. Boundary tests include M16/M17, partial and empty rows,
and dirty large-to-small replay. This preserves the original prefill launch
layout after the all-row fusion screen measured a short-prefill regression.

The same qualification slice moves attention defaults into participant
admission. Eligible single-request ROCm graphs with native FP16 KV, head
dimension 256 and rotary dimension 64 select the already measured FP16
program with eight prefill partitions and 33 decode partitions. Explicit
choices retain their semantics; unsupported combinations reject admission.
The factory consumes the exact admitted device and PP layer interval. Its
policy enters the graph, wire and prefix identities without a second memory
ledger or a graph-time environment read.

The first complete build pair passed. The focused qualification ran all 778
Unit registrations and 59 affected registrations: **833/837 passed**, with a
clean driver window and no CTest skips. All new QSA/default-policy gates and
both native prepared-Q/K ownership/byte gates passed. Four failing registrations
represent three GDN issues, including the same graph fixture registered twice.
Two fixtures lacked an explicit producer or compact bank. The production
defect was independent rounding of QKV and V widths: at legal uneven TP5,
`ceil(10240/5) - ceil(6144/5)` is 819, which is not a valid two-plane Q/K width.

The applied repair derives the new bank's capacity from the admitted query-head
count through canonical GDN head-partition math. Its normalized origin is used
only for counts; actual tensor/state origins remain unchanged. Schema and BOM
use the same checked capacity helper, replicated-decode schema unions retain
full head counts, and native eligibility explicitly requires DK64/128 and
convolution width four. A direct TP1–8/uneven-TP5 capacity regression is
registered in preflight. This repair does not claim to correct older raw-GDN
arena terms that independently round projection widths; that separate
uneven-partition accounting concern remains open.

The repaired Release and Integration builds passed, followed by **836/838**
registrations in the next complete Unit/affected run. All selected native
CUDA/ROCm regressions passed with a clean driver window. The remaining failures
were test infrastructure: the 122B arena fixture omitted GDN metadata and used
a 16,384-column merged projection where the actual GGUF declares 12,288, and
the new capacity alias named the wrong executable. The latter was correctly
rejected as an empty test selection. The fixture now uses actual metadata,
corrects the two raw FP32 bank terms by 67,108,864 bytes, and expects an
8,192-byte TP2 prepared bank. The alias names the owning activation-estimator
binary. Focused requalification passed the complete owning Unit registration,
the corrected capacity alias, the MTP terminal-gather accounting alias, and
compiler-flag validation after CMake regenerated its compilation metadata.
The source delta is exactly those two test files; every production binary is
unchanged. An explicit composite retains the original 836/838 result and the
four fresh passes, including both actual process exits and the unchanged 95
historical Unit omissions. It covers the required 838 registrations without
claiming a new 838-test run or complete production preflight. Evidence is in
`qsa-selected-defaults-qualification-composite-r1/result.json`.

A naturally quiet window allowed the selected QSA component refresh:
**seven cells / 21 arms**, all with exact terminal bytes. RankingOnly M1 at
512/768 tokens improves **23.64%/23.46%**; M16/768 improves **23.00%**. Preserved
M512 prefill differs by **−0.058%** at 4096 and **+0.011%** at 65536 tokens.
The M512/512 samples transition from about 54 to 47 microseconds within both
control and candidate processes; their aggregate median difference is
inconclusive. All samples are retained rather than discarding the slower
interval. These are component results, not model throughput. See
`qsa-selected-preparation-component-r1/summary.json`.

The first paired RTX 3090 screen also passes exact terminal-byte comparison:
M1/512/RankingOnly measures **6.868 / 5.310 / 6.868 µs** in
control/candidate/control order, a **22.685%** component-time reduction.
This is one shape, not a CUDA model-throughput or long-context result. The
original CUDA attempt stopped before timing because its adapted caller treated
`cudaFuncAttributes.localSizeBytes` as ROCm private storage. The corrected
caller and independent reader join every actual symbol and loaded architecture
to the frozen core's exact linked REG/STACK/SHARED/LOCAL metadata. Unknown or
changed resources still reject; there is no blanket 32-byte allowance. The
failed attempt remains in `qsa-cuda-selected-preparation-component-r1`, and
the accepted screen is in `qsa-cuda-selected-preparation-component-r2`.

The model gate uses the unchanged canonical 26-file/185-row panel and
requires actual defaults with diagnostic arithmetic selectors absent. Its
first selected-default run stopped during startup, before any logits: the
prepared-Q/K activation was declared only by the dense schema override, while
the MoE schema used by QFN bypassed that declaration. The inherited resolver
and GDN builder still selected the prepared program, correctly rejecting its
missing buffer. The driver window was clean. A shared schema-completion fix
and actual resolved-schema regressions are required before another model run;
manually supplying the buffer in a kernel fixture cannot certify this wiring.
The failed run is retained in `qsa-selected-defaults-model-fp16-r3`. The new
public-schema regression reproduced the missing MoE/QFN owners and incorrect
global PP indexing against the original implementation; its five cases took
191 ms, with the dense and foreign-width controls passing. The applied shared
schema completion preserves the original buffer geometry and MoE projection
schema replacement, and removes QFN's duplicate PP remapping. Rebuild and
positive/model qualification follow below.

Both coherent rebuilds passed: Release in 126.96 seconds and Integration in
219.47 seconds. The fresh selection contains all 778 Unit registrations and
26 affected schema/default/graph/GDN ownership aliases. The inherited-schema
regression and all selected native checks pass. The full selection finishes
**799/804**: five source-audit/tooling tests reach their original 30/60-second
deadlines without diagnostic output. An exact five-test repeat on unchanged
source passes in 32.05 seconds overall, retaining each original command,
timeout, affinity and CPU-budget declaration. No timeout was extended and the
separate CPU inference job was not stopped. The original failed run remains
`qsa-inherited-schema-gate-r1`; the focused pass is
`qsa-inherited-schema-timeout-retry-r1`. This establishes a passing observation
for every selected registration, not a clean fresh 804-test run or complete
production preflight. The timeout cause is not proven solely by this retry.
The explicit reconciliation in
`qsa-inherited-schema-qualification-composite-r1/result.json` preserves both
actual process results, checks unchanged source and native assets, and retains
the exact 95 historical Unit omissions. It does not invent a combined native
exit status.

Real-model numerical qualification now passes in
`qsa-inherited-schema-model-fp16-r3` (116.88 seconds). The canonical required
groups, captured execution/reset proofs and all four participants' actual
attention/GDN policies pass with all ten diagnostic selectors absent from the
real canary's environment. Every one of the **26 files / 185 final-logit rows**
is byte-identical to accepted 196shape, 196trial, 190, 180 and 151. Short-group
prefill/decode mean KL remains **0.025613694 / 0.028276396**, top-1 agreement
**93.75% / 95.833%**, and top-5 overlap **92.50% / 92.292%**. Heldout-short
mean KL is **0.014910553 / 0.004552875**, with **100% / 100%** top-1 agreement
and **95.00% / 91.667%** top-5 overlap. Threshold histories remain diagnostic;
this does not broaden the existing absolute-pilot acceptance scope.

The preceding r2 run also completed numerically but failed the outer process
observation: its observer reopened every descendant's executable link before
filtering for the actual canary. The private observer now filters from the
permission-aware process snapshot and checks stable PID/start/session/parent
and executable identity around the required environment read. Twenty focused
tests cover inaccessible ancillary processes, a required canary that cannot
be observed, and identity races. The r3 run records inaccessible driver-log
`sudo`/`dmesg` helpers separately while authenticating the actual canary; r2
remains failed rather than being retroactively relabeled.

The separate CPU inference job was never stopped or exempted: correctness can
proceed under contention, while matched throughput requires the existing strict
quiet-host guard. The fresh current-source norm-inclusive build and coherent
Unit/affected gate now pass **828/828 registrations** in **311.138 seconds**, rc0.
The actual native norm terminal proof retains **193 cases / 1,546 replays**;
all 95 original Unit omission identities/reasons remain explicit, with no new
omissions. Evidence: `gdn-norm-single-q8-unit-affected-{integration-build,gate}-r1`.
This clean result supplements rather than relabels the earlier 799/804 timeout
and unchanged-source retry. It is not complete ProductionTestPreflight.

## Historical bundle matched throughput

The retained pre-norm prepared-QK/QSA/default bundle completes an unprofiled
Release control196/candidate/control196 run with its original qualified runtime,
`/tmp/qfn-qsa-inherited-schema-qualified-runtime-r3`. The corrected observer
proves the controlled root's exec into the pinned MPI launcher and one actual
rank0/world1 worker; all driver, loader, source-archive and 30-second retirement
checks pass. The strict whole-batch guard passes in **510.338 seconds**, with
**496 passive observations and no external-work overlap**. Original failed r1
receipts remain failed and preserved. This authenticates a historical frozen
runtime, not the mutable current checkout.

Both roles retain 512 prompt / 256 natural outputs, context 4,096, M512, TP4 MI50,
28 physical NUMA1 workers, RAM ngrams, sharded vocabulary, FP16 KV,
greedy seed 42, MTP/prefix off and warm1/measure3. All **nine 256-token trajectories
are identical**. Medians and retained ranges are:

| Role | Prefill median (range), tok/s | Decode median (range), tok/s |
|---|---:|---:|
| Frozen196 controls, six samples | 820.637629 (813.345217–821.657358) | 42.243967 (42.146684–42.360168) |
| Earlier bundle, three samples | 820.973634 (820.864819–821.078960) | 42.464423 (42.397817–42.512337) |

Changes are **+0.04094% prefill / +0.52186% decode** within this matched run.
Evidence: `qsa-inherited-frozen-throughput-{a1,b1,a2}-r2`,
`qsa-inherited-frozen-host-r2`, and
`/tmp/qfn-qsa-inherited-frozen-benchmark-r2/summary.json`.

The separate norm-inclusive A/B/A measures 820.172503 / 42.335471 versus its own
821.056620 / 42.280710 controls, a neutral **-0.1077% / +0.1295%** result with
nine identical trajectories. See the [norm evidence](QFN_GDN_RECURRENCE_TUNING.md#output-normalization-and-singleq8-preparation-2026-10-25).
**The two trials are not a paired norm comparison:** differences between their
medians do not prove a norm-fusion regression or recovery. No new throughput
score is promoted; checkpoint196's 821.541 / 42.229 remains the qualified headline
and the 42.959 decode target remains open.

## Next bounded experiment

Prioritize the user-requested **online rescale exponential elision**: only
first-tile and exact equal/winning-maximum scale cases, preserving the original
FP32 update and reduction order. Do not infer finite-input equivalence for
signed zero, infinities or NaNs. Require exact scalar/grouped/partial-tail and
nonfinite byte regressions, plus ISA/resource evidence that the redundant
exponentials are removed without spills. Native component timing and a matched
model comparison follow correctness; no gain or policy change is claimed yet.
