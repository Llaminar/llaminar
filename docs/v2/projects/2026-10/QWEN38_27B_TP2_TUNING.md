# Qwen3.8 dense 27B: local TP2 tuning

Investigation started 2026-10-03 on `feature/qwen38-27b-tp2-tuning`, from
`develop` at `68e45ec359c50d18e33e1923d59b97975c809dee`.

The user accepted the CUDA compute-scaling effort as complete on October 5,
with the slower-participant diagnostic near 1.9x. This is acceptance of the
measured result, not a claim that installed production reaches 1.9x.
The end-to-end TP2 throughput target remains separate;
decode remains a measured tuning objective. Each topology must pass the canonical Release
HTTP E2E cell. Kernel measurements precede communication tuning; GDN reduction
order remains fixed. Terminal-head placement and other measured winners become
production defaults only after correctness and matched workload validation.

## Current compute tuning, October 5

**Accepted for publication.** The latest completed external normalization /
quantization fusion cohort measures **471.390 ms Single / 249.118 ms slower
TP participant = 1.892237x**, or **1.905831x** using the participant mean.
An unchanged-control repeat measures **1.896786x** for the slower participant.
These prototypes remain outside the production tree. Publication retains the
qualified r2 implementation, whose representative-embedding, all-collective-
elided control measures **475.756 / 256.516 ms = 1.854685x** for the slower
participant. All elision measurements deliberately invalidate model arithmetic;
the separate native HTTP and throughput runs below establish correctness and
production performance. Further tuning stopped at the user's request.

The source-bound r2 receipt covers all 152 source, test and skill files, with
1,370 passing prerequisites and four passing Release HTTP cells. Before
publication those hashes matched exactly. Final edits only record acceptance,
clarify diagnostic procedures and correct the terminal-head policy comment.
The unrelated standalone MI50 attention experiment is excluded from the commit.
The user authorized `git commit --no-verify` using these existing pass receipts.

### Earlier cohorts and attribution

The best completed **external candidate**, with all NCCL work removed and
real embeddings available to both participants, measures **473.456 ms
Single / 250.631 ms slower TP participant = 1.889057x**. The participant
mean is 249.367 ms (1.898629x); neither establishes the 1.9x target. The
remaining slower-participant gap is 1.443 ms against the candidate's own
Single baseline. This combines register-owned H5120 normalization and
byte-proven smaller/core GEMM tile choices; it is not installed production.
A later paired control repeats that same candidate at **473.886 / 249.914 ms
= 1.896198x**, still below target. Its participant mean exceeds 1.9x; that
does not establish the slower-participant target. Keep both cohorts visible
instead of selecting the favorable repeat as a new kernel improvement.

The latest installed policy passes all **1,370 prerequisites**. Its fresh
same-source complete-parent cohort measures **475.795 ms Single / 261.839 ms
TP2 = 1.817133x** with native allreduces elided; slower-participant accounting
is **1.815888x** at **262.018 ms**. Native TP2 takes **445.925 ms**. Both
Single and TP2 improve in absolute time over the preceding grouping-only
cohort, but Single improves more, so the scaling target remains unmet. The
current TP2 threshold is **250.419 ms**, requiring another **11.420 ms**.
Every process retains the actual 448+64 row schedule, native compute inventory
and all measured parent submissions. Elided outputs are intentionally invalid
and do not certify correctness. Raw per-projection attribution is complete;
current-source Release HTTP and canonical throughput gates now pass.
Receipts: `cuda-compute-scaling-captured-policy-r2/summary.json` and
`captured-policy-r2-qualification/status.json` under the artifact root below.
Earlier cohorts follow to preserve their distinct sources and baselines.

All four current-source Release HTTP cells pass: dense Qwen3.8 CUDA2/ROCm2
TP and Qwen3.6 MoE CUDA0/ROCm0, with FP32 activations, FP16 KV and dynamic
MTP. HTTP receipt `captured-policy-r2-http.json` has SHA
`cf6a1ceba386dfd9a0847873e08d612b11565249bbfb80c99fbcf397b5c9aea1`.
The uninstrumented 512-prompt/256-decode Single/TP/TP/Single brackets also
pass, including all twelve greedy iterations matching Single per backend,
stable repeated TP MTP work, exact placement, and clean complete driver windows.

| Backend | Single prefill tok/s | TP2 prefill tok/s | Prefill scaling | Single decode tok/s | TP2 decode tok/s | Decode scaling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA | 1,012.426 | 1,015.853 | 1.003385x | 63.8828 | 66.8816 | 1.046942x |
| ROCm | 297.919 | 471.802 | 1.583659x | 37.2630 | 46.4633 | 1.246902x |

These canonical throughput values include communication. They are means of
paired process medians, with three measured iterations and one warmup per
process. Receipts: `captured-policy-r2-model-validation/{cuda,rocm}-timing/`.


The native inventory reveals that these interventions remove **allreduces
only**: one vocabulary allgather remains in each TP2 prefill parent. It sends
one live row of 124,160 FP32 logits, or 496,640 bytes per participant. The
1.817133x result therefore must not be described as fully communication-free.
A separate paired cohort under `cuda-compute-scaling-all-collectives-r1/`
measures native, allreduce-only elision, and all-collective elision. It completes
with **476.091 ms Single**, **259.966 ms allreduce-elided TP2**, and
**254.241 ms mean / 260.910 ms slower-participant all-collective-elided TP2**.
The resulting ratios are **1.872595x mean / 1.824732x slower participant**.
The last
intervention replaces that gather with two device-local copies of the local
logits, completely defining the consumed output while deliberately invalidating
model arithmetic. It retains those copy costs and verifies that every other
compute launch remains unchanged. This is diagnostic evidence, never an
inference mode or a correctness certificate.

**Elision also changes the input distribution.** Vocabulary-parallel embedding
returns zero for tokens outside the participant's shard. Removing its sum leaves
that participant without the real embeddings, so matching kernel inventories
does not prove representative arithmetic activity or power use. The completed
`cuda-card-role-cohort-r2/` reverses the TP member order: the slower role moves
from physical CUDA0 (**260.729–261.663 ms**, CUDA1 **247.681–248.738 ms**) to
physical CUDA1 (**256.872–257.681 ms**, CUDA0 **251.962–252.930 ms**). This
excludes a fixed card/PCIe difference as the explanation for the larger gap.
Both cards retain the same 390 W default power limit; read-only telemetry shows
power-limited clocks during prefill. No clock or power setting is changed.
The first role-cohort attempt is retained as failed because its reversed config
specified both an explicit primary and TP members; r2 corrects only that config.

A follow-up under `cuda-embedding-preserved-cohort-r1/` retains exactly the main
and shifted-MTP embedding sums while eliding the 128 layer sums and terminal
vocabulary gather. It authenticates the retained stage names through the setup
annotation ABI. This is a conservative comparison with two collective nodes
per parent still present, not fully communication-free timing. It measures
**475.944 ms Single / 264.342 ms TP2 mean / 264.944 ms slower participant**:
**1.800491x mean / 1.796396x slower participant**. Both participants now take
about 263–265 ms, supporting the input-distribution explanation. Neither the
faster zero-input lane nor participant-mean elision timing establishes the
1.9x target for representative production computation.

The new raw projection cohort passes **34/34 exact format/geometry/M cases**,
including full-output byte certificates, zero local-memory resource checks and
a complete clean driver window. Applying the actual 448+64 schedule to the
400 dense projection occurrences gives **395.899 ms Single / 212.993 ms local
TP2 = 1.858741x**. This sum excludes recurrence, attention, elementwise stages
and communication; it is not an observed whole-model critical path.

| Projection role | Schedule scaling |
| --- | ---: |
| FFN gate/up | 1.906809x |
| FFN down | 1.939821x |
| GDN QKV | 1.707995x |
| GDN Z | 1.783821x |
| GDN output | 1.919564x |
| Full-attention Q/gate | 1.747496x |
| Full-attention K | 1.208471x |
| Full-attention V | 1.280427x |
| Full-attention output | 2.031199x |

The raw receipt is `cuda-raw-schedule-scaling-captured-policy-r2/status.json`,
SHA `272f58e31d742f3a5dc04e3255ca095ab461c42349ecf1eef30e8bd4d09d0f8e`.
The retained parent DAG also proves each of the sixteen full-attention QKV
stages already has one shared quantizer and three independent tensor-core
branches; it has no GDN-style grouping/serialization defect.

A diagnostic BM32 tournament tests two physical tiles and four staging
schedules, with bracketed production controls, on eight full/local projection
geometries. All **80 cases** pass complete output-byte checks. Smaller tiles
lose on GDN QKV/Z; narrow attention gains are too small to close the model gap.
No policy is installed from this experiment. Its first attempt failed during
host-wrapper compilation, before GPU execution; r2 fixes explicit host ABI
types and binds private IQ table initializers to canonical source bytes.
Artifacts: `cuda-small-row-tile-prototype-r1/` (failed compile) and
`cuda-small-row-tile-prototype-r2/raw-diagnostics/` (completed evidence).
The GDN independent-column/input-staging diagnostic completes all **14 cases**,
retaining the complete fixed eight-part recurrence arithmetic. Every candidate
passes 32 retained correctness graphs covering full/local model head counts,
two requests, live-row changes, normalization, in-place/out-of-place state,
snapshots and guards. A 64-thread/eight-row candidate takes **327.168 us** for
the local 448+64 recurrence schedule versus production **339.584–341.886 us**,
but slows Single to **659.648 us** versus **619.840–647.744 us**. It is not a
production winner. The diagnostic also omitted production's relocatable-device
compilation flag: even its unchanged 128-thread/eight-row geometry uses 76
registers versus production's 112. Compiler mode must be controlled before
attributing a difference to tiling. All variants have zero local-memory bytes;
reported occupancy is a static ceiling, not achieved occupancy. Artifacts:
`cuda-gdn-column-prototype-r1/evidence/`.
The production-matched RDC control now passes five paired correctness/timing
cases: production is **652.608/341.888 us** Single/local, closing at
**652.544/341.632 us**; the unchanged 128-thread/eight-row clone is
**642.176/341.760 us**, closing at **652.416/341.760 us**. The 64-thread
candidate takes **668.992/342.014 us**, slowing Single without a local win.
The apparent earlier recurrence gain does not survive matched compiler mode.
No recurrence kernel changes are promoted. Receipt:
`cuda-gdn-rdc-control-r1/evidence/status.json`; the r2 runner updates its
prerequisite to the completed row-tail r2 after the original host-build failure.

The exact-divisor GEMM experiment keeps the same full-K arithmetic body and
production relocatable-device compilation mode. BM224/BN32 fails the spill
guard for codebook17 under its original two-CTA register budget; a one-CTA
budget clears that guard but exceeds the static shared-memory ABI. Neither
failed build reaches GPU execution. The completed r4 removes BM224 and tests
BM112/BN64 with two warp partitions (224 and 448 threads). All **80 cases**
pass byte and resource checks, but local QKV loses (502.464 vs 472 us), as
does local Z (270.656 vs 267.136 us). A 3.2% full-Z win is a Single-only
result; no table is installed. r3 catches a stale diagnostic thread-count
array before substituting a GPU kernel. All failed evidence remains in
`cuda-divisor-row-tile-prototype-r{1,2,3}/`; r4 owns the complete cohort.

The M64 extension retains the original full-K or canonical-part arithmetic
mode, including the ordered reducer. The first incomplete cohort correctly
rejects an attempted full-K substitution for canonical-part IQ4_XS, after
proving all 20 QKV cases. Its Q5_K local QKV winner takes **107.968 us vs
128.160 us** with BM32/BN64 and cooperative staging. Isolated ISA/resource
evidence binds that exact symbol: 108 registers, zero local bytes, 17,152
shared bytes and four blocks/SM as a static ceiling, with no local load/store
instructions. The corrected mode-aware cohort passes 60 remaining cases,
and an extension passes 90 more; local Q5_K GDN output also improves to
**66.880 us vs 78.784 us**. BM16 passes 40 cases but does not beat the best
BM32 choice. Most FFN and attention cells retain production. These are
diagnostic candidate measurements, not installable training evidence.

A row-tail experiment preserves the installed kernels and original M448
activation-sum stride while splitting output rows into independent 384+64
launches. Its first host-only build fails on CUDA 13 graph-edge API arity;
r2 corrects that ABI and preserves the failed receipt. All **28 cases** pass full output-byte checks. The best local
QKV tail takes **477.120 us vs 473.536–473.664 us** production; local Z
loses substantially (**349.504 us vs 267.008–269.184 us**). Full-width cases
also lose. The extra launch is not promoted merely for eliminating padded
row work. Artifacts: `cuda-row-tail-prototype-r{1,2}/`.

The retained native DAG exposes another independent scheduling hypothesis:
in all 48 GDN stages, FP32 alpha/beta pointer publication and projection
wait for both native QKV/Z projections. Their common normalized input and
distinct named workspace buffers permit a controlled overlap experiment.
`cuda-gdn-subgroup-overlap-r1/` authenticates actual captured physical ranges,
common input/quantizer identity, unchanged kernel geometry, and the complete
downstream join before editing edges. Only M448 full-K native pairs are in
this prototype. Three repetitions pass all 23 affected M448 captures
(21 source formats and full/local production widths), including dirty-output
byte checks. The eight-case model cohort also passes native inventory and
complete-request timing checks. Production is **475.484 ms Single / 264.059 ms
slower TP participant**; overlap is **476.780 / 263.514 ms**. It slows Single
by 0.27% and saves only 0.21% locally. The resulting ratio increase is not a
meaningful production win; the edge change is not promoted. Receipt:
`cuda-gdn-subgroup-overlap-r1/model/summary.json`.

Filtered native stage timing now attributes the remaining non-TC work under
the same source identity. It adds **4.4% Single / 8.4% TP** host-prefill time,
so these are diagnostic last-replay intervals, never uninstrumented service
times. Summed 448+64 intervals show normalization at **12.213 ms Single /
14.014–14.169 ms per TP participant**, tiny FP32 projections at
**23.050 / 12.073–12.276 ms**, and recurrence at **30.013 /
16.565–17.027 ms**. The TP control retains the two embedding collectives;
their event spans also contain inter-participant waits. Artifacts:
`cuda-side-stage-timing-r2/`, including per-stage HTML/JSON/CSV reports.

A complete GDN-bundle microbenchmark captures sixteen dependent copies of
the actual projection stage per sample, with 31 samples and independent
output-byte checks. Production is **1,941.504–1,956.670 us Single /
1,078.080–1,080.704 us local** for 448+64. QKV tile3/staging0 reduces the
local bundle to **1,045.376 us**, while QKV+Z tile3/staging3 takes
**1,043.388 us**. Their Single results do not beat the best Single control.
All nine cases pass, but these bundle observations are diagnostic, not
installable per-projection trainer evidence. Receipt:
`cuda-gdn-pair-tiles-r1/status.json`.

Static inspection of the qualified normalization kernels finds 40 registers,
a 64-byte stack frame and local load/store instructions for the dynamically
indexed sixteen-value cache. This is deliberate private storage, distinct
from register spills. The external H5120 prototype retains all **1,024
logical lanes**, their five-element partials, original warp slots/XOR tree,
and runtime divisor while explicitly scalarizing cached values. All six
candidate/control cases pass 28 geometries and eight poisoned/aliased replays
per geometry. Candidate local memory is zero. Best fused normalization for
M64 uses 1,024 physical threads (**4.096 vs 6.528 us**); M448 uses 512 threads
(**45.824 vs 56.448 us**). The combined schedule improves about **20.7%**.
R3 owns the complete evidence under
`cuda-normalization-register-prototype-r3/evidence/`. Earlier attempts retain
a missing driver-link failure and a terminated malformed host compilation;
neither is a numerical failure or a hidden replacement path.

The r4 prototype applies the best physical width only at M64/M448 and leaves
other rows unchanged. Its three control/candidate cases pass all 28 byte and
alias geometries. Isolated SM86 assembly for all four selected symbols has
zero stack bytes and no local load/store instructions. The eight-process
complete-parent A/B measures **476.094 → 474.000 ms Single** and
**263.012 → 262.561 ms slower TP participant**. That is a 0.44% Single win
but only 0.17% locally; the microbenchmark's 20.7% cannot be extrapolated to
the parent. The conservative ratio changes **1.810162x → 1.805294x**.
No normalization production change is installed from this result yet.
Receipts: `cuda-normalization-register-prototype-r4/{evidence,isa,model}/`.

The embedding-preserved diagnostic still launches **256 FP16 conversion
kernels per parent** for the 128 elided layer sums. A separate six-process
cohort removes those communication-only conversions after authenticating
that each original FP32 producer and reverse-cast consumer use the same
pointer, live-row owner, stage and stream. Both embedding sums and their
codecs remain intact; every other compute launch is unchanged. It measures
**476.621 ms Single / 264.098 ms TP with codecs / 258.632 ms TP without
layer codecs** using the slower participant: **1.804713x / 1.842852x**.
The difference is **5.466 ms**, or 2.07% of the prior TP diagnostic. This
separates communication formatting from model compute but still includes
two real embedding collectives per parent and two local vocabulary copies.
It remains numerically invalid and does not establish a fully
communication-free 1.9x result. All six cases and the driver window pass.
Receipt: `cuda-communication-codec-elision-r1/model/summary.json`.

Combining the normalization and tile candidates in eight real-parent processes
reduces **475.993 → 474.037 ms Single** and **259.539 → 254.221 ms slower
TP participant**, with layer codecs removed but embedding collectives retained.
The first combined cohort is rejected by its inventory audit: the Single GDN Z
shape also matches TP attention Q. The r2 audit explicitly authenticates both
consumers of that shared geometry; it does not introduce topology-specific
dispatch. All eight r2 cases pass. Receipt:
`cuda-combined-compute-candidate-r2/model/summary.json`.

To remove the last two collectives without a zero-input participant,
`cuda-replicated-embedding-diagnostic-r1/` replicates both immutable EmbedQ8
shards during setup. Both devices verify all **1,430,323,200 bytes** against
the concatenated production tables before graph instantiation. Their SHA256 is
`df3952dda852c2d13c538f4c99d865f7a8ba5f01f5ba1290e9005627337434ae`.
The unchanged native lookup kernel retains token IDs, launch geometry, outputs
and edges; only its table pointer and vocabulary descriptor change. Main/MTP
share the same verified backing. Existing FP16 embedding codecs and two local
vocabulary copies remain. Every executable retires before storage release, and
allocated/released byte counts match. Source inspection confirms the original
embedding uploads complete through their exact setup-stream event before
preparation returns; see `source-publication-audit.json`.
The eight-process cohort measures production tiles at **475.756 / 256.516 ms**
(**1.854685x**) and the combined candidate at **473.456 / 250.631 ms**
(**1.889057x**), always using the slower participant. All native NCCL inventories
are empty. Later layer sums remain intentionally invalid, so these are compute
diagnostics, not model-correctness or serving-memory-policy certificates.

The M448 narrow-projection BM16 extension passes **40/40** captured byte/resource
cases. Local IQ4_XS K improves to **93.824 us**, and local Q5_K V to
**116.288 us**; full-width projections retain production or the earlier BM32
winner. Receipt: `cuda-row16-m448-narrow-r1/status.json`.
An additional **240-case** FP32/FP16/BF16 tiny-projection tournament varies
four/eight columns per CTA and 256/512/768-element shared staging while retaining
every original arithmetic partial. It passes, but no variant beats the TP
width24 warp kernel. Width48 still benefits from four-row shared tiles.
Receipt: `cuda-tiny-staging-r1/evidence/status.json`.
Nine complete-parent FFN cases test existing core tiles under actual concurrent
gate/up execution and serial down projection. Controls take **250.609 and
252.327 ms**; no tested alternative establishes a reliable improvement, and
the larger gate/up tiles regress substantially. No FFN policy is promoted.
Receipt: `cuda-ffn-context-r1/model/summary.json`.

The H128 gated-normalization prototype passes **4,200** captured byte/alias
cases across five runs, each with eight poisoned replays. A warp owns the
four original 32-element groups and preserves their XOR reductions and serial
cross-warp addition order. Four heads per CTA reduce the TP M448+64 kernel
schedule from **29.056 to 24.064 us**. All three candidate warp counts have
zero local storage and no local-load/store ISA instructions; occupancy values
remain static resource ceilings. The eight-process model cohort does **not**
establish a TP win: Single improves **473.886 to 472.765 ms**, but the slower
TP participant worsens **249.914 to 250.854 ms**. The candidate ratio is
**1.884623x**, versus **1.896198x** for its paired control. Do not promote
this isolated-kernel winner as a TP improvement. Receipts:
`cuda-gated-norm-warp-r2/model-r2/summary.json` and
`cuda-gated-norm-warp-r2/evidence/status.json`.
The first model attempt is retained as an audit failure: it assumed six
materialized buckets on Single. Production admits 448 rows there and 512
on TP2; r2 authenticates the actual production bucket inventory instead.
Both topologies execute 448+64 because `PrefixHarvestSchedule` selects the
448-token reusable checkpoint before the terminal 64-token suffix. That
prefix-cache obligation remains part of every compared workload.

A two-iteration native GEMM K-loop unroll fails the unchanged spill guard for
the 512-thread physical tile. That failed build remains recorded under
`cuda-prefill-unroll2-r1/`. Revision r2 removes that entire physical family,
keeps every codebook for the 128/256-thread families, and compiles without
spills. All **40** captured byte/resource/economy cases pass. M448 IQ4_XS
gate/up improves from **563.776–567.040 to 545.600 us** locally and
**1095.104–1100.800 to 1068.416 us** at full width. Down projections and
most Q5 cases do not benefit. Revision r3 changes only the diagnostic
environment names so this candidate can compose with the fixed baseline.
Its selected FFN symbol has **127 registers, 32,768 shared bytes, zero stack
and local memory**, and no local ISA accesses. The resource occupancy ceiling
is two CTAs per SM; achieved occupancy is unavailable. The eight-process
complete-parent cohort passes: Single improves **474.656 to 470.391 ms**
and slower TP **250.627 to 248.992 ms**. These are useful absolute gains,
but Single benefits more: scaling moves **1.893876x to 1.889184x**.
The participant mean gives **1.900187x**, which does not establish the
slower-participant target. No unroll policy is installed. Receipt:
`cuda-prefill-unroll2-r3/model/summary.json`.

The non-power-of-two column experiment passes **60/60** captured byte/resource
cases. Its 96-column body replaces two power-of-two column-owner masks with
exact modulus, preserving every K operation and partition. The 192-thread
BM64/BN96, staging2 candidate improves local GDN Z from **267.136–268.672
to 258.368 us**. All other tested full/local QKV, Z, and FFN gate/up
geometries prefer the installed tiles. The subsequent four-process concurrent
GDN bracket rejects that isolated winner: the local M448 bundle rises from
**853.824–854.976 to 895.232–895.424 us**, while the unchanged Single bundle
stays near 1611 us. Its complete 448+64 schedule scaling falls from
**1.8700–1.8736x to 1.7996–1.8040x**. Both projection outputs remain byte
identical. This is not a production candidate; physical grid balance and an
isolated projection win do not establish a stage win. Receipts:
`cuda-column96-bundle-r1/status.json` and
`cuda-column96-prototype-r1/evidence/status.json`.

The installed public-M1 schedule also explains a real source of local GEMM
overhead: at the 82-SM policy input, Q5_K QKV uses **23 ordered partitions**
for N5120 and **11** for N10240. Its ordered accumulator cannot simply be
removed. A further diagnostic specializes only authenticated immutable
alpha/beta, bias/existing-output pointers, and the ordered-mode flag, keeping
the complete partition count/span and arithmetic tree. The SM86 all-codebook
build passes the spill guard, and all **30** captured byte/resource/economy
cases pass. Specialization gives no clear improvement on any tested geometry:
local QKV's best 472.128 us lies inside its production bracket of
470.528–473.920 us, and the other candidates tie or regress. No policy is
installed. Receipt: `cuda-fixed-scalars-r1/evidence/status.json`.

The tiny-projection tournament passes **180** bracketed captured cases across
FP32/FP16/BF16 weights, all six capture buckets and widths 24/48. Each shared
candidate retains the independent legacy tree and twenty dirty-output replays.
At M448, FP32 width48 improves from **382.528 to 327.168 us** with a four-row
shared tile; width24 keeps the existing warp kernel (**189.984 us**).
FP16/BF16 width48 have smaller wins, while width24 again keeps production.
The first executable accidentally linked a second static CUDA runtime and
failed before any measurement; the one-shared-runtime revision passes with
clean driver evidence. Artifacts: `cuda-tiny-projection-448-r2/evidence/`.
The eight-process model cohort now proves the FP32 four-row candidate's
absolute Single benefit: **473.598 to 470.991 ms**. TP retains its original
width24 path and measures **251.185 versus 251.144 ms** on the slower
participant. Scaling therefore falls **1.885457x to 1.875380x**. Retain the
Single benefit as a promotion candidate rather than preserving a slower
baseline to inflate TP scaling. Production gates remain required. Receipt:
`cuda-tiny-rows4-model-r1/model/summary.json`.

A new normalization/activation-quantization fusion diagnostic preserves the
original 1024-logical-lane RMS tree, normalized FP32 and residual outputs,
32-value quantization tree, row-major scales and optional block-major integer
sums. Its SM86 build passes the spill guard. All **1,008** primitive cases
compare all five output byte ranges, poisoned inactive capacity,
residual/output aliases, large-to-small captures and zero/subnormal/random
inputs against the original production bridges. A subsequent **504-case**
capture-hook cohort proves fusion of one exact unpublished normalization leaf:
all pointers and edges survive, quantization destinations are disjoint, and
the stream frontier remains unchanged. The first hook fixture failed its
kernel-count assertion because it did not preload the override; its corrected
loader selection passes, with both driver windows clean.

For the selected 1,024-thread body, residual normalization plus quantization
with sums takes **52.864 us at M448 / 6.464 us at M64**, versus about
**74.7 / 10.8 us** for the original production pair. The comparison to the
existing register-normalization candidate is smaller and needs the complete
model cohort; do not extrapolate those raw savings. The four selected symbols
use **48–50 registers, 128 shared bytes, zero stack/local memory** and no
local ISA accesses. Achieved occupancy remains unavailable without attachment.
The eight-process paired model cohort now passes: Single improves
**473.714 to 471.390 ms**, mean TP **249.204 to 247.341 ms**, and slower
TP **250.975 to 249.118 ms**. Slower-participant scaling improves from
**1.887492x to 1.892237x**, still **1.018 ms** above its own 1.9x target.
The **1.905831x** participant mean does not establish the target. Each
measured bucket replaces exactly 128 adjacent norm/quant pairs while
retaining the other four H5120 normalizations, all GEMMs, all fixed trees
and all actual parent submissions. Receipt:
`cuda-norm-quant-fusion-r2/model/summary.json`, source status SHA
`22519267cb14b070fccb93a21db149b7ad121759928225c2e3dfc2e35787e931`.
This remains external research; no production graph or arithmetic changes
have been made from it.

The remaining narrow-attention candidates also fail to show a reliable model
gain. A new eight-case byte sweep proves that private CUDA table symbols from
BM16 and BM32 libraries coexist; local M448 K improves **100.224 to 93.568 us**
and V **142.848 to 116.224 us** in isolation. The full model cohort additionally
uses the faster full-width V tiles at M64/M448. Its first attempt fails the
substitution-count audit because the full-width V geometry appears **17** times:
sixteen main layers plus replicated `MTP0_kv_proj`. The corrected cohort applies
that full-width choice to the replicated MTP projection on both topologies and
passes all eight processes. Single is effectively unchanged
**471.726 versus 471.742 ms**; slower TP rises **248.697 to 249.051 ms**.
The apparent mean-participant improvement is only 0.086%, and slower scaling
falls **1.896786x to 1.894163x**. Keep the unchanged fusion control rather than
promoting isolated narrow-tile wins. Receipt:
`cuda-row16-mode-prototype-r2/model-r2/summary.json`, source status SHA
`8c6aeef8f0f99a5e9c3935b58d929f063a2aa50b871d7b5554c5c92ee2909cd8`.

A further vector-memory fusion prototype passes all **672** primitive cases.
Each physical thread owns four adjacent logical lanes; three inter-thread XOR
steps and two component steps preserve the original logical warp reduction.
The final 32-warp tree and all five output representations remain unchanged.
At 256 physical threads, M448 residual normalization plus quantization with
sums takes **49.280 us**, versus the scalar fusion's approximately 52.9 us;
M64 takes **5.696 us**, versus approximately 6.46 us. Both builds retain the
same production math/compiler settings and pass the spill guard. Capture
ownership subsequently passes all **504** captured byte/lifecycle cases under
`cuda-norm-quant-vector-r2/`. The r3 complete-model comparison was stopped when
the user accepted the scaling result; its partial measurements do not qualify
the vector candidate. The cancellation receipt is
`cuda-norm-quant-vector-r3/publication-stop.json`.


A preallocated event pair per native parent submission now observes every
Single replay, including 384+64+64. It inserts no graph nodes, synchronizes
nothing during inference, and reads distinct events only at graph retirement.
The complete allreduce-elided CUDA baseline is **1.76645x**: Single **501.087 ms**
versus mean TP2 participant **283.669 ms**. Slower-rank accounting gives
**1.76556x**. All native non-collective launch inventories match their controls.
These are diagnostic GPU graph spans, not hardware-counter kernel service time
or numerically valid model outputs from the collective-elided intervention.

Explicit cap 448 removes the 64 padded rows in TP2's 448+64 live split. With
that cap, Single **501.631 ms** versus TP2 **269.985 ms** gives **1.85799x**
(**1.85686x** using the slower rank). Disabling concurrent projections loses
performance: **276.530 ms**, only **1.81926x**; concurrency remains enabled.
The shared CUDA/ROCm inventory previously omitted 448 on both backends, even
though an explicit cap could request it and kernel choices already covered it.
The shared source inventory now includes 448. Four focused host registrations
and both native CUDA/ROCm retained-family registrations pass, including
128+64, 448+64 and 512+64 repeated request resets. The native driver window
is complete and clean. Both complete builds pass with the spill guard:
Integration SM80/86/89/90 plus gfx906 and Release SM86 plus gfx906. The first
complete gate passed 692/693 Unit registrations; its retained failed receipt
identifies only a stale literal 37,926-cell trainer-matrix expectation. The
check now derives cardinality from shared inventories and explicitly checks
all 21 formats and M=448. Its 38 focused cases and the fresh complete 693-test
Unit gate pass. The fresh complete prerequisite gate passes all **1,366**
registrations with no failures or skips: 693 Unit, 289 host, 134 CUDA, 162 ROCm
and 88 combined multi-device checks. It finishes at 22:04:45 UTC in 2,125.816
seconds under `/tmp/qwen38-tp2-prerequisites-shared-448-r2/`; receipt SHA
`4d5ce649dfabb56a9c1acce25552a0e6bd3e54ac54520a432579cf81781f1d15`.
The source/build identity SHA is
`023a2606dda36e48ec0a88f8d1ba42d2fd1533794aa8c8fadf061f2d77813692`.
All four Release HTTP cells also pass: Qwen3.8 dense CUDA2/ROCm2 TP and
Qwen3.6 MoE CUDA/ROCm single-device, FP32 activations, FP16 KV, dynamic MTP.
The receipt is `/tmp/qwen38-27b-tp2-tuning/shared-448-r1-http.json`, SHA
`64554841aa66a96c0dda5270b6a689a25e5010d2d78c6aa31bc204afd97c4802`.
Maximum serving capacity remains 512. Fresh performance analysis derives the
actual admitted Single/TP2 row schedules rather than assuming capacity stays
unchanged when 448 becomes an admission candidate.

The shared-default paired Release cohorts are complete. Both backends retain
identical greedy tokens across all twelve measured iterations and no padded
prefill rows; each result averages two process medians of three iterations.

| Backend | Single prefill | TP2 prefill | Prefill scaling | Single decode | TP2 decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| CUDA | 978.211 tok/s | 1004.096 tok/s | 1.026462x | 63.247 tok/s | 66.426 tok/s |
| ROCm | 297.144 tok/s | 470.186 tok/s | 1.582348x | 37.408 tok/s | 46.217 tok/s |

The Single prefill brackets drift by only 0.1463% on CUDA and 0.00384% on ROCm.
These are canonical communication-included timings, not compute scaling.
CUDA now admits 448 on Single too. Fresh complete GPU-parent accounting gives
**493.632 ms Single / 268.826 ms TP2 = 1.836252x** with allreduces elided;
slower-rank accounting is **1.834699x**. The same TP2 parents with native NCCL
take **452.387 ms**. The earlier 1.85799x explicit-cap result kept Single at
384; it cannot be presented as the new shared-default scaling. The fresh target
is at most **259.806 ms** for TP2, roughly 9.02 ms below the installed policy.
Receipts live under `shared-448-r1-model-validation/` and
`cuda-compute-scaling-shared448-r1/`.

An external setup-only substitution of registered full-K gate/up candidates
in the exact TP2 parent finds tile 3 / all-operands staging promising at
**264.314 ms** mean-rank and **264.714 ms** slower-rank versus controls
**267.895–271.095 ms**. It retains all pointer/scalar arguments and graph edges,
authenticates the compiled ABI/resources and changes exactly 128 gate/up nodes
per card. It is not installed, does not certify model correctness, and has not
robustly reached 1.9x. More paired evidence and production certification remain.

The fresh shared-default candidate cohort repeats this win: installed controls
are **269.470 / 270.270 ms**, while gate/up tile 3 / staging 3 takes
**265.594 / 265.039 ms**. Adding down tile 4 / staging 3 reaches **264.621 ms**
(**264.777 ms** slower-rank). Down tile 0 / staging 0 and standalone GDN tile
changes lose to gate/up alone. These are external exact-node substitutions,
not installed policies or model-correctness certificates. See
`cuda-prefill-tile-incontext-r2/status.json`.

The pre-change captured DAG revealed an additional grouping defect: GDN QKV and Z
executed serially, with two activation quantizers. `GDNProjectionStage` rejected
different NativeVNNI codebooks even though CUDA and ROCm fused implementations
select each projection's own decoder and already own disjoint persistent
scratch. FFN gate/up already overlaps. Any earlier suggestion that the GDN pair
already overlaps was incorrect. The temporary source-bound DSO and shared native capture regression now pass
on both CUDA and ROCm under `gdn-mixed-projection-diagnostic-r2/`. Each backend
retains 644 native graphs and replays each four times, alternating larger and
smaller live extents over poisoned output capacity. All 21 source formats,
ordinary/verifier row boundaries and full/sharded dense model shapes preserve
every live output byte and leave inactive capacity untouched. The installed
negative control fails only the missing grouping/concurrency assertions; its
output bytes remain correct. The first DSO setup failed before GPU execution
because the preload lacked its explicit core-library dependency; that failed
receipt is retained separately. The grouping change is now built into production
and passes the registered native capture regressions on both backends.

The paired whole-parent comparison with grouping gives **487.435 ms Single /
264.851 ms TP2 = 1.840412x** (slower-rank **1.839899x**), versus installed
**493.753 / 268.901 ms = 1.836187x**. Thus both sides improve; the new 1.9x
threshold is **256.545 ms**, not the threshold from the old Single baseline.
Captured DAG assertions prove one shared quantizer and independent QKV/Z
branches for all 48 GDN blocks. Grouping changes neither projection geometry
nor the fixed reduction tree. See `cuda-compute-scaling-gdn-mixed-r2/`.

Combining grouping with gate/up tile 3/staging 3, down tile 4/staging 3,
QKV tile 3/staging 0 and Z tile 3/staging 3 reaches **258.518 ms** mean-rank
(**258.554 ms** slower-rank), approximately **1.8855x** against that improved
Single baseline. Grouping-only controls bracket **266.278 / 266.043 ms**;
gate/down controls bracket **260.988 / 261.049 ms**. These source-bound
exact-node substitutions are diagnostic evidence only; they are not installed
or numerical certificates for the tile changes. Additional choices and fresh
Single controls remain necessary. See `cuda-prefill-tile-incontext-gdn-r1/`.


The follow-up tile cohort (`cuda-prefill-tile-incontext-gdn-r2/`) finds no
additional winner. Its best-recipe controls are **256.637 / 258.337 ms**
(mean-rank), so the roughly 1.9x boundary is inside the observed process
variation; no target achievement is claimed. Lower-register QKV alternatives,
additional staging schedules and alternative gate/down/Z tiles all lose.

The production GDN grouping change also removes a failed homogeneous verifier
retry. The focused device-free regression reproduces both defects against the
old core: mixed native descriptors split, and a failed verifier bundle submits
twice. The fixed core passes that contract. All **694 Unit registrations**
pass after complete Release and Integration builds. All nine focused
registrations, including native captured GDN, existing floating-format/serial-row
checks and the trainer evidence contracts, pass in 96.32 seconds with a complete,
clean driver window under `gdn-production-promotion-r1/`.

Installable dense CUDA/ROCm tournaments previously timed direct launches.
They now measure retained graphs of sixteen complete projection operations per
event sample, normalize per operation and authenticate `captured_operations`
in every raw row. Sweep-plan v4 binds native-graph submission and latency basis;
old direct-launch evidence cannot enter this generation. The shared evidence
regressions pass, including rejection of direct or partial-batch records.
Fresh CUDA collection covers all 21 formats and all 13 distinct full/local
projection geometries observed in the model at M=448. ROCm checks all formats
at M=64/448 for one matching narrow projection; all 42 cells pass, validating the
symmetric trainer change without installing a new ROCm policy. CUDA finishes
all **273/273 cells** at 23:48:08 UTC with zero byte mismatches and a complete,
clean driver window. The common generator authenticates **208 exact keys**,
replacing 192 measured keys and adding 16. Of those existing choices, 53 change;
25,440 unrelated keys remain byte-identical. The installed include SHA is
`add28ec9390f34d4dcffaa1df500e88f6f0a7c4d23612b18560cd6a316981351`.
The additive receipt is `gdn-production-promotion-r1/cuda-prefill-captured-retention.json`.
Both complete builds pass. The first prerequisite attempt stops during
discovery because the new focused Unit registration incorrectly carries the
Integration-only preflight label; no model timing is admitted. The corrected
inventory retains that Unit entry and adds a separate device-free
`V2_Integration_GDNProjectionGroupingContract`, which passes in 0.37 seconds.
The fresh gate under `captured-policy-r2-qualification/` passes all **1,370**
registrations without failures or skips: 694 Unit, 290 host, 135 CUDA, 163 ROCm
and 88 shared/multi-device tests. It finishes at **00:35:32 UTC** in **2,168.954
seconds**. The prerequisite receipt SHA is
`862738ffadc03a461b8c6ffe02f93ccff7613862c9963e5a7b7fe07112400fe7`.
Its source/build
identity SHA is `0ae68546996c4311746c3eb4db094901aac3dfc4e027d7e6ea8edf1db416768e`.
The source-bound parent cohort completes at **1.817133x**, below the required
1.9x; the following per-projection cohort reaches **1.858741x**. Neither the generated
candidate nor focused checks replace same-source Single/TP2 target proof and
Release HTTP gates before publication. No commit, push, PR or image publication
has occurred.

A separate existing-kernel floating crossover probe completes all **36 cases**:
FP32/FP16/BF16, M=64/128/256/384/448/512, N=24/48, K=5120 and two gates per
group. Every candidate preserves complete output and guard bytes over twenty
poisoned replays; timings rotate the production and shared row-2/4/8 schedules
over sixty native-event samples. At the model's TP2 FP32 M=448,N=24 geometry,
the installed warp path takes **186.176 us**, while the best shared candidate
takes **219.072 us**. No floating policy changes. The N=48,M=448 row-4 result
is an isolated Single-side lead, not a model-level win or a reason to lower the
threshold for all shapes. Receipts are under `tiny-projection-crossover-r1/`.

Memory A/B keeps maximum 512 and adds 448 to the retained ladder. CUDA ABBA
agrees exactly on **26 MiB per card / 52 MiB TP2** additional resident GPU memory;
ROCm AB measures **4 MiB per card / 8 MiB TP2**. Activation, KV and workspace
plans remain unchanged. The planner adds 22 MiB CUDA / 2 MiB ROCm per graph to
its family reservation; these are admission units, not measured allocation
deltas. Both backends retain six rather than five prefill graphs per card,
and the three-request padded-token ledger drops from 192 to zero. The complete
native driver window is clean. No instrumentation or graph substitutions were
loaded for this memory cohort.

Artifacts under `/tmp/qwen38-27b-tp2-tuning/`:

- `cuda-compute-scaling-baseline-comparison-r1.json`: paired baseline and cap
  448 cohorts, including complete request/rank sums.
- `cuda-compute-scaling-cap448-ordered-r1/status.json`: the rejected ordered
  projection cohort and its complete per-submission native timings.
- `cuda-prefill-tile-incontext-r1/status.json`: candidate runs and immutable
  launch substitutions/resource receipts.
- `shared-448-memory-r1/comparison.json`: complete-family memory comparison,
  source/binary/log bindings and driver receipt.

The measurements below are retained historical evidence for the earlier source
state; the shared-default results above supersede their current performance.

## Current status, October 4, 19:47 UTC

The remaining flat CUDA scaling is dominated by host-memory communication over
the shared PCIe uplink. Matched bulk FP16 allreduces take **1.250 ms on CUDA
versus 1.021 ms on ROCm**. A communication-only three-phase copy surrogate,
with neither NCCL nor RCCL, takes **1.272 ms versus 1.039 ms** on the same
devices and payloads. The cross-vendor ratios are **1.224x and 1.225x**.
The actual model uses SHM/direct on both backends; CUDA's native SHM pages are
entirely local to NUMA 0. Explicit placement, host backing and ordinary
fixed-count API controls find no prefill win. Detailed evidence and limits
follow the parent-time comparison below. Production source/defaults and the
canonical **1.014497x CUDA / 1.739176x ROCm** prefill results are unchanged.

The CUDA fast/slow prefill swing is causally attributed to native pinned RAM
checkpoint allocation in timed harvest. Before the fix, each TP2 prompt made **44 native pinned
allocations totaling 983,162,880 bytes**, including four 235,339,776-byte recurrent
images. The two-event-per-parent observer sees the main GPU parent remain about
**389 ms** in both fast and slow processes. Its final matched requests take
**523.951 ms versus 741.885 ms**; enclosing pinned-allocation totals rise from
**479.98 ms to 723.18 ms**. Those host spans overlap GPU execution and must not be
added to the parent durations. Native-call tracing shows only about 0.13–0.16 ms
between `cudaHostAlloc` and enclosing backend totals, excluding an engine-lock
explanation for the hundreds of milliseconds of allocation cost.

An isolated allocator-only A/B/B/A diagnostic preserves tokens, MTP work and
chunks: original allocator cohorts take **676–752 ms**, while a temporary
huge-page registration interposer takes **523–554 ms**. This intervention
establishes causality; it is not installed. The production implementation now
materializes a single PMA-admitted pinned arena per GPU RAM tier during setup.
Shared archive ranges survive eviction, restore and disk-writer aliases and
pending producer events. MoE runtime metadata owns independent ranges within
the same budget. CUDA and ROCm share this implementation. The qualification and
fresh timing below establish correctness and eliminate hot pinning; CUDA's
best-case prefill scaling remains far below 1.7x.

The final focused gate passes **13 registrations with no failures or skips**
and a clean driver window. Five lifecycle gates also pass **20 repetitions
apiece (100 registrations)** on CUDA, ROCm and the device-free arena. Coverage
includes out-of-order producer completion, consumer aliases retained across
logical eviction, pending final-owner teardown, asynchronous disk publication
and hydration, poisoned large-to-small reuse, and independently retained MoE
runtime metadata. Production counters prove exactly one native pinned
allocation per arena across these repeated operations.

The arena and its test translation units additionally pass **8 ASan/UBSan cases
repeated 20 times (160 case executions)**, including 80,000 randomized placement
operations and 32,000 concurrent leases across eight threads. There are no
sanitizer or leak reports; the linked production core is not fully instrumented.
Two focused before-fix regressions reproduce stale handles mutating/retiring a
newer same-key archive and overflowing section arithmetic passing admission.
Ownership identity and checked extents now reject both. Unpublished producer
events also have a device-free fatal-state regression.

Both complete builds pass with spill guards: Integration SM80/86/89/90 and
gfx906; Release SM86 and gfx906. The unchanged source and binaries pass the
complete canonical gate: **1,366 registrations, no failures or skips**, comprising
693 Unit and 673 production preflight (289 host, 134 CUDA, 162 ROCm, 88 combined
multi-device). The gate completes at **18:35:48 UTC in 2,113.913 seconds**. Its
receipt is `/tmp/qwen38-tp2-prerequisites-prefix-arena-r1/prerequisites.json`, SHA
`3f1d287743de421117128b2dfa5943d9fce7ba240b247c52a9b71adea6cb10bd`.

Canonical Release HTTP passes **45/45 checks in each of four cells**: dense
CUDA2/ROCm2 and single-device Qwen3.6 MoE on both backends. All server-log,
shutdown, memory-retirement and complete driver-window checks pass. The report
is `prefix-arena-r1-http.json`, SHA
`92129d67b75363c7468edd7372f9e09a49728b8fe7ddc155f97b7269e0fd1760`.

An additional source-bound PerfStats audit proves each inference GPU materializes
exactly one **4 GiB** configured arena, then acquires **477 payload leases** through
the full HTTP sequence. Both real MoE cells additionally capture **34** and
restore **7** portable **914,395-byte** runtime-state images. HTTP uses the
canonical automatic-placement policy (observed CUDA MoE GPU1, ROCm MoE GPU3,
CUDA dense GPUs0/1, ROCm dense GPUs0/2); the separate performance cohorts retain
the original frozen CUDA0/1 and ROCm1/2 TP placements. The arena proof is
`prefix-arena-r1-http-arena-proof.json`.

Exclusive Single/TP2/TP2/Single Release timing cohorts pass on those frozen
plans. Each process has one warmup and three measured 512-token prefill / 256-token
decode iterations. All twelve measured token streams per backend match Single;
the TP2 MTP work and chunk ledgers match their paired runs. Both driver windows
are complete and clean. Each value below averages two process medians:

| Backend | Single prefill tok/s | TP2 prefill tok/s | Prefill scaling | Single decode tok/s | TP2 decode tok/s |
|---|---:|---:|---:|---:|---:|
| CUDA | 963.373 | 977.340 | **1.01450x** | 62.044 | 66.435 |
| ROCm | 248.477 | 432.145 | **1.73918x** | 37.413 | 46.190 |

CUDA's two TP2 process medians are 523.791 and 523.951 ms, and Single drift is
only -0.0435%. The arena eliminates the historical slow-state swing in this
cohort but does not improve its best-case throughput. ROCm still meets 1.7x.
The source-bound timing evidence is `prefix-arena-r1-model-validation/`.

The sparse CUDA parent/host observation also passes its matched token, MTP and
chunk checks with a clean driver window. All ten observed prefills contain
**zero native pinned allocations**. The last fast request is 522.745 ms:
host submission is 15.629 ms, while main-parent GPU spans are 390.270/391.114 ms
and tail-parent spans are 75.603/75.625 ms. Host and GPU intervals overlap and
cannot be added. Approximately 56 ms remains outside the parent spans; its
transfer attribution is not yet measured. These parents have only two timing
events each, no leaf timing events, and all five submissions/replays observed.
Evidence is `cuda-prefill-prefix-arena-parent-r1/status.json`.

A prefill-only, external NCCL identity diagnostic now passes a complete
Single/native-TP2/identity-TP2/identity-TP2/native-TP2/Single cohort. Its outputs
are deliberately invalid and cannot qualify production correctness or throughput.
Omitting exactly **130 native allreduce kernels per chunk (260 per participant
per prompt)** reduces the combined GPU parent spans from approximately **465 ms
to 285 ms**. Every non-reduction native kernel name and launch geometry matches,
the native vocabulary gather remains, and all TP2 submissions/replays are
observed. This is a measured intervention, superseding the earlier interval-DAG
estimate. Allreduce and its dependencies account for approximately **180 ms,
39% of native GPU parent time**, or a **63% overhead over the identity graph**.
The source-bound evidence and exact aggregation are
`cuda-prefill-reduction-identity-r3/status.json` and
`cuda-reduction-isolation-proof.json`; the driver window is complete and clean.

Two earlier diagnostic failures are preserved. R1 confused logical participant
0 with physical CUDA device 1; the corrected join uses frozen-plan membership.
R2's MPI helper overwrote the inference process's intervention receipt with an
empty one, even though the native graph confirmed that allreduces were omitted.
R3 gives each process its own receipt and joins it to the actual trace PID.

The separate sparse-copy observation also passes matched tokens, MTP work,
chunks, loader identity and a clean driver window. Each TP2 prefill submits
**983,154,688 explicit D2H bytes** across both devices. Its four 235,339,776-byte
recurrent archives dominate. Simultaneous copies on the two participants take
approximately **47–48 ms per frontier**, while their same-size device-hot D2D
copies take **0.57–0.60 ms**. The first frontier completes before the terminal
result; the final archive is still pending then and completes asynchronously.
Consequently back-to-back prefill-only requests can inherit the previous
archive's remaining transfer time. Their 566/386-ms native/identity medians
must not be substituted for the canonical prefill-plus-decode cohort's 524 ms.
The sparse observer changes no copy, allocation, model value or engine wait;
an initial diagnostic incorrectly required archive completion at the terminal
boundary, and its clean failed driver window is preserved. R2 retains timing
events until the existing native device-retirement boundary without adding a
wait. Evidence is `cuda-prefix-copy-timing-r2/status.json`.

At the user's request, the same native/identity parent measurement now passes
on frozen ROCm1/2. The external HIP observer preserves all three measured token
streams, MTP ledgers and aggregate prefill chunks against its full 256-token
decode control. Its prefill median overhead is **0.186%**. Both backends retain
130 native **FP16 Ring/Simple** allreduces per parent and one vocabulary gather;
the identity intervention removes only those reductions. All non-reduction
kernel symbols and launch geometries match, including the actual 512/64 row
buckets. All parent replays are observed, and both complete driver windows pass.

| TP2 GPU parent time, 512-token prompt | CUDA | ROCm |
|---|---:|---:|
| Native allreduces | 465.464 ms | 1,138.408 ms |
| Identity reductions | 284.598 ms | 1,001.151 ms |
| Allreduce-attributable increment | **180.866 ms** | **137.257 ms** |
| Increment / native parent time | **38.857%** | **12.057%** |
| Increment / identity parent time | **63.552%** | **13.710%** |

Each table entry averages per-process/per-participant medians of the three
measured per-request parent sums in an A/B/B/A cohort. Participant times are
never added across devices. Prefix-copy time is outside these parents. The
counterfactual includes the effect of native reduction dependencies/scheduling;
it is not a summed kernel-service-time claim or valid model output.
CUDA's absolute reduction penalty is **1.318x ROCm's**, while its relative
overhead over the identity graph is **4.635x**. CUDA's remaining graph is about
**3.518x faster**, so even comparable PCIe communication consumes much more of
its execution time. The measured communication cost explains most of the flat
CUDA scaling without duplicate whole-model compute or a host launch bottleneck.
It does not establish the 1.7x production target.

ROCm evidence is `rocm-prefill-reduction-identity-r1/status.json`, SHA
`c4774f3321e47626867ff203388013698ffa7fe5cbf2755ee1f15c59dd362733`.
Its `observer-control-and-loader-proof.json` authenticates the actual repaired
HIP, canonical RCCL and GNU OpenMP DSOs. The complete comparison is
`cuda-rocm-reduction-isolation-comparison.json`, SHA
`5e455d9089dc07e6a89d61ca841c85072ff43aafb0462da45f44919838a49c84`.
The external observers and identity intervention remain diagnostic artifacts;
no reduction-elision mode is installed in production.

### Why CUDA communication costs more

The new transport cohort uses the actual frozen pairs: CUDA0/1 and ROCm1/2,
FP16, hidden width 5120, bulk capacity/live rows 512/448 and tail 64/64.
Earlier exploratory ROCm collective probes used GPUs0/1 and are not the
source for this comparison. Each native record has 31 samples of 16 captured
operations, terminal byte comparisons and poisoned inactive output. Counted
production timing omits the passive byte receipt; separate receipt runs prove
the same useful outgoing volume. Bulk medians below average two process
medians per backend. Tail native timings have one process per backend.

| Matched TP2 operation | CUDA | ROCm | CUDA / ROCm |
|---|---:|---:|---:|
| Native bulk allreduce, 4,587,520 B/rank | 1.250176 ms | 1.021379 ms | 1.2240x |
| Native tail allreduce, 655,360 B/rank | 0.240384 ms | 0.173740 ms | 1.3836x |
| Plain simultaneous bulk host reads/writes | 0.887328 ms | 0.720554 ms | 1.2315x |
| Plain ordered ring transfer phases | 1.272448 ms | 1.038842 ms | 1.2249x |

The plain-copy rows use a common **8 blocks x 256 threads** per participant,
16-byte cache-bypassing host loads, persistent buffers and exact-stream graph
events. They average two process medians, with 21 samples of 16 operations
per process. The three-phase surrogate sends half a message, reads and forwards
half, then receives half, following the native two-rank ring's ordered data
volumes. It omits peer progress and reduction arithmetic and introduces native
graph boundaries between phases. It is neither an allreduce implementation nor
a hard lower bound. The full 8/32/80-block sweep is retained: CUDA's faster
32-block three-phase result is about 1.162 ms, so these measurements do not
prove that the installed native kernel is mathematically optimal. They do
reproduce the bulk vendor gap without either communication library.

Both pairs use host-memory SHM/direct, with **no P2P transport**. Real-model
native logs prove eight CUDA channels and four ROCm channels, their physical
bus IDs and their exact loaded dependencies. Both PCIe paths have the previously
audited shared Gen3 x16 CPU uplink and a Gen4 x8 participant branch. Raw copies
show nearly twice the single-card time when both cards transfer together;
the shared uplink does not provide twice the aggregate host bandwidth.
CUDA's best unidirectional SM copies reach about **12.2--12.5 GB/s** with
one card and **12.1--12.4 GB/s aggregate** with two. ROCm reaches about
**13.1--13.7 GB/s aggregate**. For the common-grid simultaneous read/write
case, aggregate bandwidth in each direction is **10.340 GB/s CUDA versus
12.733 GB/s ROCm**. Slot-width symmetry therefore does not imply equal
effective host-transfer performance.

The following interventions narrow the attribution:

- **NUMA:** every CUDA SHM payload mapping in the real model is on node 0.
  Explicit local binding leaves bulk allreduce near 1.25 ms. Forcing all of
  those pages onto node 1, verified in `numa_maps`, adds only **1.34%** versus
  the bracketed local runs. Mapped endpoint counts include aliases and are
  not a physical allocation ledger. RCCL uses native driver-owned pinned
  backing; its payload's physical page nodes are not exposed by `numa_maps`,
  so the process memory-policy experiment alone does not prove their placement.
- **Buffer attributes:** registered CUDA mappings and native cacheable pinned
  backing both give about **0.889 ms** for the common-grid bidirectional copy.
  Write-combined backing takes **0.908 ms** and supplies no candidate win.
- **Device counts and observation:** CUDA bulk counted/native-fixed timings
  are **1.250/1.258 ms**; ROCm is **1.021/1.055 ms**. Removing the passive
  receipt does not improve bulk timing. Making all 448 capacity rows live also
  does not help. Mutable device-owned counts therefore do not explain prefill's
  gap. Fixed live-one decode is faster than capacity-16/live-one on both
  backends (CUDA 16.768/24.896 us; ROCm 20.240/28.890 us), but a host-fixed
  diagnostic is not a production replacement for dynamic MTP counts.
- **Small-message cost remains:** the tail has a larger native vendor gap
  than the bulk. Plain three-phase tail copies take roughly **190/159 us**
  at the common grid, versus native **240/174 us**. Bulk bandwidth evidence
  must not be used to declare every protocol/scheduling cost optimal.

This attributes the main loss to the measured host-transfer route and its
ordered communication phases. It does not independently separate the GPU's
PCIe/cache transaction implementation from CPU-root and switch details; no
hardware registers, firmware, ACS policy or physical placement were changed.
The previously observed **3.518x faster** CUDA non-reduction graph makes this
similar absolute communication burden consume a much larger fraction of CUDA
execution. Reducing communication on the critical path and prefix-state copy
volume remains necessary; the ordinary channel/protocol/placement controls
do not provide the 1.7x production target.

The real-model diagnostic again preserves every measured generated token
stream, MTP ledger and prefill chunk against its matched control. CUDA prefill
is **522.108/522.305/521.929 ms** and ROCm is
**1183.390/1184.404/1184.358 ms**. These are observed diagnostic runs, not a new
canonical throughput cohort. All completed cohorts have complete clean driver
windows and byte checks; the unchanged source retains its existing 45/45 HTTP
certification for both dense topologies. The external copy kernels pass CUDA
SM86 and HIP gfx906 spill guards. The bidirectional kernels use 24 CUDA
registers / 17 HIP VGPRs, no local storage, with native occupancy APIs admitting
6/8 blocks per SM respectively at 256 threads. These are compiler/resource
facts, not profiler-derived achieved occupancy. CUPTI remains unattached.

Evidence is `/tmp/qwen38-27b-tp2-tuning/transport-cause-r1.json`, SHA
`c9305501f881fd841a8bc2a87c3976a9be70eb0c3bd72a93e9a2b1314c0409c8`.
It joins the exact source/build identity, loaded DSO hashes, model controls,
driver windows and full timing records from `matched-transport-r2`,
`raw-host-path-r2`, `raw-ring-phases-r1`, `cuda-host-backing-r1`,
`native-policy-isolation-r1` and `model-transport-placement-r1`.
`transport-probe-resource-audit-r1.json` records all three HIP kernel spill
counts and object identities. The first NUMA wrapper rejected node 1 before
GPU initialization because its nodemask width was too small; corrected R2 uses
the complete machine-word mask. The first HIP raw-load probe emitted four
scalar transactions; its timings are superseded by the explicit vector-load
R2. Neither failed/superseded diagnostic is used as evidence of a production
defect or a backend performance limit.

Evidence under `/tmp/qwen38-27b-tp2-tuning/`:

- `cuda-prefill-causal-parent-r1/status.json`, SHA
  `2b0f94f9209517e785633a1150ffeb8c6c6256cc127f73233cb7519be1815bd6`.
- `cuda-prefill-causal-parent-r2/status.json`, SHA
  `a4f61a19c88b4fede71e55d7f13b712aefb1fc54a247daab388078178ddc1a38`.
- `cuda-pinned-hugepage-causal-r1/status.json`, SHA
  `196df7a1c6b21d6eb101af26ec8fcc281ea3406138b37406ceeb00e40512470e`.
- `prefix-arena-before-hardening.xml` preserves the two failing API regressions.
- `prefix-arena-r3-repeat20/status.json`, SHA
  `8f318457ddda86ead63405751b7d17db9cccafdb33ff6d67997f7e4b073ba354`.
- `prefix-arena-final-asan-ubsan.log`, SHA
  `03915dc58ac8020f1a0d8abaf6e23baaaf85018450416922d1fe74a74319ef0f`.
- `prefix-arena-r4-focused-final/status.json` records the final native allocation
  counter proof and all 13 focused passes.
- `prefix-arena-r1-build-identities.json`, SHA
  `7092cbe245c1c95d6d566a4b35bbc5c07133a2ee197cb1e7411b0468a810384b`.


The request-row collective correction separately passes **10 focused
registrations with no failures/skips**, clean driver state, after rebuilding
all affected test executables against the changed core. Its receipt is
`request-rows-r2-focused.status.json`, SHA
`df26c3440f422a4fed2daa7399034b5c733724506a08ec10d783440a1c2fad44`.

## Previous fully qualified source slice

The last qualified overwrite slice passes both complete builds, the full gate,
canonical default Release HTTP and matched default/explicit-FP16 model cohorts.
The newer request-row and persistent-arena changes pass the fresh builds, full
gate, four HTTP cells and matched timing described above. The
older receipts below do not certify those changes. The completed
canonical transaction passes all
**1360 registrations: 693 Unit and 667 production preflight**, with zero failures
or skips. Preflight comprises 287 host, 132 CUDA, 160 ROCm and 88 multi-device
registrations. The complete gate finishes at 14:38:59 UTC in 2105.181 seconds.
Its receipt is `/tmp/qwen38-tp2-prerequisites-workspace-overwrite-r1/prerequisites.json`,
SHA `d1e0de0bf7e9b7e65de20a9a6a1f69079d1bae49f58e5239d266c89d0f4856f2`.
The final source/build identity receipt is `workspace-overwrite-r1-build-identities.json`,
SHA `8958a8beb57cf7c1828e75598793d2ba79775188d603136253060978eb81eae5`.
The new before-fix CUDA gate detects **65 captured redundant clears**, with no
poisoned-output byte failures and a clean driver window. A checked non-owning
scratch contract now orders complete dense partial writers before readers;
CUDA and ROCm grouped MoE use it for static/runtime, quantized/floating output
publications. Shared/routed gates now poison every mutable projection bank.
Required empty-participant zero output and accumulation/protocol initialization
remain. The R3 receipts below certify the previous frozen source, not these
new changes. The focused build first exposed a standalone device-free MoE
target missing its explicit C++20 requirement; that requirement is corrected.
All **14 affected registrations** pass: contract admission/failure rules,
all-format dense capture, shared/routed dirty scratch, original-row boundaries,
projection phases, compact publication and runtime grouping on both vendors.
The dense case covers 44 format/shape modes, three retained row geometries
(65/17/64) and 20 poisoned replays per mode. Its 21 empty-partition diagnostics
use the exact retained core producer/reducer symbols, changing their partition
count together without modifying installed arithmetic. Every consumed empty
plane is exact +0, output capacity beyond live M remains poisoned, and all
native projection captures contain zero clears. The initial diagnostic changed
an unused cached span and failed its empty-plane assertion; the corrected
diagnostic passes in 9.64 seconds. Both complete focused driver windows are clean.
The complete Integration (SM80/86/89/90 + gfx906) and Release (SM86 + gfx906)
builds pass with spill guards enabled. Fresh canonical default Release HTTP passes
**45/45 checks on CUDA2 and ROCm2**, including production stochastic/dynamic MTP,
prefix cache, long recall and terminal-only GPU D2H. The source-bound report is
`tp2-http-workspace-overwrite-r1.json`, SHA
`5df2966d105335e35ecf00a033d14f7f9cec1dc496eebc5f6e533ab277f9d9d6`.
Both driver windows and complete request/process retirement checks pass.

The fresh six-row, 18-iteration cohorts per backend pass all greedy token and
MTP-work checks. ROCm defaults reach **431.498/46.141 tok/s** prefill/decode,
against **248.128/37.421** Single: **1.739015x prefill scaling**. CUDA Single
averages **967.336/62.070**; explicit FP16 averages **977.908/66.764**. CUDA's
two default process medians are **979.247 and 764.209** prefill tok/s, while
decode remains **66.766/66.848** and all token/chunk/MTP evidence matches.
The default mean is **871.728/66.807**, or **0.901164x prefill**. Preserve that
process variation; there is no matched macro A/B proving the clear removal's
throughput gain. The authenticated cohort hashes are
`321048299f643e8f1768270c84aa578d202acea7daeaeac9b720827375a7db44`
(CUDA) and `6831d50dfd5ae7e4067e2a12220409fada982f1e129ae020f3b7fb96b9507f86`
(ROCm).

Actual native parent probes correct the initial estimate: Single M64 falls
**1852 to 1625 nodes (-227)**, Single M384 falls **1400 to 1399 (-1)**, and
TP2 M64 falls **2116 to 1953 per card (-163)**. TP2 M512 remains 1790. The
old trace contains exactly those removed memset nodes, including fused GDN,
QKV and KV projections; the two standalone FFN-down microbenchmarks do not
represent the full affected model inventory. Weighting the exercised captures
removes **455 clear nodes per Single prompt and 163 per TP2 participant**.
The first current stage diagnostic rejected a changed NVCC anonymous-namespace
reducer symbol. Exact SM86 disassembly proves all **208 instruction/control
words identical**; only its static function-name header differs. The identity
receipt is `canonical-reducer-rebuild-symbol-identity-r1/identity-receipt.json`,
SHA `07164317b9a791db8823786241cbb5e839a1fcdce53abe38f5b04ba4a3871f2d`.
A follow-up host-only observer now passes compiled-ABI and actual-work checks.
Each prefill parent has **400 main tensor-core projections**, all sharded on
exactly one axis: 272 halve N and 128 halve K. GDN has 24 local value heads
instead of 48; full attention has 12 Q/2 KV heads instead of 24/4. The terminal
head has 124160 local vocabulary rows. The main-work audit is
`cuda-captured-main-work-audit-r2.json`, SHA
`7c09ad04886928c18878893e7618db80830e05fec6478c7f751c98cc1cc10b0c`.
Single executes 1200 main matrix nodes over 384+64+64 physical rows; each TP
participant executes 800 over 512+64. Its MAC count is 0.5625 of Single,
or 1.125 combined, because TP processes 576 physical rows against Single's
512. This is padded work, not a duplicated whole-model forward. Tiny FP32
projections, elementwise operations and MTP sidecars remain separate costs.
Event spans still do not prove simultaneous compute across device clocks.
The corrected r2 stage, raw-kernel, bucket-cap and candidate HTTP jobs pass;
their first attempts stopped on the diagnostic symbol rejection before inference.

The current-source cap cohort retains all 18 token comparisons and all 12 TP
MTP-work comparisons. Single averages **968.451/62.012 tok/s** prefill/decode.
Cap 448 averages **1005.358/66.362**, or **1.038109x** prefill scaling and about
2.93% over fast cap 512. Cap 512 again has a slow process (754.111 tok/s) beside
its fast process (976.756); neither is discarded. Explicit cap 448 now passes
the complete **45/45 CUDA Release HTTP** workload and clean driver/retirement
checks. Its receipt is
`cuda-http-cap448-workspace-overwrite-r2/status.json`, SHA
`f8ca2b925668b01f6f1e20d9fcba6c8adb200aaaf7097914e35535e09a0f9053`.
No bucket default is promoted by these diagnostics.

### NCCL and actual launch-count audit

Metadata-only paired Release runs authenticate the installed native library,
source-compiled work layout, retained arguments and real host submissions.
Main FP16 allreduces use **RING/SIMPLE, eight channels and 288 threads**, with
SHM direct transport. All 128 projection sums and the main embedding sum bind
the same device row-count pointer as all 48 GDN recurrences. A K-sharded output
still requires all H=5120 columns; it cannot reduce only H/2. Native passive
receipts prove **4,587,520 useful outgoing bytes per participant/operation**
for cap 512/live 448 FP16, rather than the 5,242,880-byte capacity.
The BOM's `capacity_bytes` is storage evidence, not a wire-byte counter.
The native work/submission proof is
`cuda-nccl-native-setup-audit-r1/native-work-and-submission-proof.json`, SHA
`991c9defbc822fb806088fcb713191fd28e4d960e4566abe284f6e9af08b56df`.
Typical peer host submission skew is about 0.9–1.0 ms. It does not establish
GPU compute overlap. Pinned SHM page locality remains unproven.

The recursive, read-only launch audit passes both focused retained-parent/WHILE
gates and its paired model control: **522.041 vs 521.604 ms** prefill, with
identical tokens, MTP work and chunks. It enumerates native conditional children
without modifying any node or reading a GPU pointee. Each measured prefill
submits each of its two chunk graphs once per participant. The four total
prefill submissions are one graph-readiness request, one warmup and two measured
requests. Captured unused buckets have zero launches. Per participant/request,
the main parents contain **3711 unconditional kernel dispatches**, 800 main
matrix dispatches, six MTP prefill projections, 96 GDN recurrences, 260 native
allreduces and two vocabulary gathers. Conditional inventory is potential work,
not observed body multiplicity; full attention retains guarded direct and split
alternatives. The proof is `cuda-native-launch-inventory-r2/launch-count-proof.json`,
SHA `4c3aedcfe7c207fcd2e4acddcf374528e804734deede845626bfcdcb3db49e28`.
The first expanded diagnostic confused global stage IDs with vector indices
and stopped during setup; corrected metadata lookup passes with clean drivers.

Two economy defects are now explicit. Each GDN input is quantized twice into
the same INT8 and scale banks: the first Q5K quantizer also writes block sums,
while the second IQ4 quantizer repeats activation/scale work and omits sums.
This gives **48 repeated quantizers per chunk, 96 per prompt per participant**.
There are no identical complete matrix launches within one stage. A separate
graph-edit diagnostic removes exactly those verified second quantizers, lowering
the two parent sizes from 1790/1953 to 1742/1905 nodes. Its A/B/B/A cohort preserves
all **12 measured token streams and MTP ledgers**, with clean driver/retirement
checks. Fast prefill means are **521.003 and 522.898 ms** for controls and
**521.881 ms** for the candidate: no macro gain is established. The other
candidate process reproduces the slow state at **686.802 ms**. All four active
fast/slow candidate graphs have identical normalized symbols, launch geometry,
dependencies and native RING/SIMPLE eight-channel work scalars. That does not
prove identical pointee values, clocks, memory locality or device overlap.
The candidate receipt is `cuda-gdn-quant-dedup-diagnostic-r1/model-runs-r1/status.json`,
SHA `4c8f495a22a8adc1e9ab48f3f1a06f82b12c287dd4dbd95d2705147d51ccba18`;
its geometry proof is `fast-slow-geometry-proof.json`, SHA
`1343c17a02471654ce0ddabfb42be74314eaff189e612621d98475c5124f4bfe`.
No shared-quantization production change is installed. Dynamic-MTP admission
also executes **15 physical sidecar forwards**
before launching the complete parent on every request, even when the selected
depth is 3 and all executables are retained. The source calls this capacity
preparation and replaces its physical draft counter with the device logical
ledger. The 14 chained launches recur in both measured requests, not only warmup.
The retained parent itself contains one verifier per transaction and conditional
guards for later draft slots. Admission/materialization must be separated from
selected execution before claiming economical request reuse.

The shifted-MTP prefill embedding sum is a third distinct defect: it still
communicates fixed capacity instead of its own append-row extent. The main
prompt count is not valid for shifted rows, and a ragged batch cannot use
request 0's count as a global prefix. A symmetric typed request-row contract
and explicit native payload/preflight regression are required. That correction
now uses checked `NativeAllreduceRequestRows` banks through the ordinary native
precision/cast implementation and one final exact-stream publication. Captured
CUDA/ROCm tests exercise three ragged requests with passive payload receipts,
poisoned padding, both transport precisions and twenty retained replays. Dense
and MoE graph declarations share an explicitly registered preflight regression.
The focused build is running; no new correctness or performance pass is claimed.

### PCIe path verification

Read-only sysfs and PCI capability checks find a real CUDA last-hop asymmetry:
GPU 0 (`0000:3f:00.0`) reaches **Gen4 x16**, while GPU 1 (`0000:42:00.0`)
reaches **Gen4 x8** at its adjacent PEX880xx port (`0000:41:00.0`). Both endpoints
and ports advertise x16 capability. All intervening switch links are Gen4 x16.
The common switch uplink `0000:3b:00.0` and CPU root `0000:3a:00.0` negotiate
**Gen3 x16**, matching the CPU port's maximum speed.

The frozen benchmark plan selects ROCm **GPU 1/2** (`0000:91:00.0` /
`0000:97:00.0`), giving **Gen4 x16/x8** branch paths, matching CUDA's widths.
ROCm GPU 0/1 are an available x16/x16 pair, not the measured pair. The
common switch/CPU uplink `0000:86:00.0` / `0000:85:00.0` is also Gen3 x16.
ROCm GPU 2/3 report x16 at
their GPU functions but have **Gen4 x8** links above their onboard bridges
(`0000:95:00.0` / `0000:98:00.0`); endpoint width alone is incomplete evidence.
Both systems use equivalent switch families on their respective NUMA nodes.

A 20-second native CUDA collective probe samples all ten relevant nodes 200
times. Both GPU links rise from idle Gen1 to Gen4 and retain x16/x8 widths;
switch internal links and the Gen3 x16 uplink remain unchanged. All available
correctable/nonfatal/fatal AER counter files remain unchanged and the driver
window is clean. The theoretical one-direction capacity before packet overhead
is 31.508 GB/s for Gen4 x16 and 15.754 GB/s for either Gen4 x8 or Gen3 x16.
Both measured pairs therefore share the same width/speed ceilings. Bus width
alone does not establish a cause for their scaling difference or CUDA's
fast/slow process states. The initial read-only summary assumed ROCm GPU 0/1;
checking the authenticated frozen plan corrects that interpretation. The ROCm
single-card plan selects GPU 3, also behind an x8 switch branch; its GPU endpoint
still reports x16. Evidence is in `pcie-topology-idle-r1.json`,
`pcie-link-capabilities-r1.json` and `pcie-topology-loaded-r1.json` under the
local result root. No PCI configuration, driver or device placement was changed.

An isolated sealed FP16 NCCL dispatch candidate passes seven focused gates and
all useful-payload/byte checks. Bulk allreduce improves less than 1%; it remains
uninstalled. Its receipt is
`cuda-nccl-uniform-fp16-candidate-r1/validation-r2/status.json`.
A fresh **14-case FP16 protocol/thread screen** also passes all output, poisoned
tail, actual-payload and driver checks. Tree/Simple reduces cap 512/live 448
latency from about 1246 to 1228 us (about 1.5%), but one-row latency rises from
25.344 to 27.456 us (8.33%). Full-capacity improvement is only 0.32%.
Simple/128 threads, Simple/512 threads, LL and LL128 do not improve bulk latency.
No protocol default is promoted. The screen receipt is
`cuda-nccl-fp16-protocol-screen-r1/status.json`, SHA
`8857ffe49464c774f367132ed18f0a1028719f6a56ea7ed905e36e3315610dc1`.

The audit keeps required initialization separate from dirty scratch. ROCm's
legacy multi-K atomic accumulators require zero starts, its inverse route map
requires invalid sentinels for absent routes, and the GDN arena's blanket clear
owns only live convolution/recurrent state. CUDA's no-local-expert branch must
still define its live output as zeros; it now uses the checked output span's
bytes. Ordinary ROCm KV-pool and both TurboQuant cache startup clears are
potential later removals, pending poisoned append/reset/restore proofs; none
has been removed by this slice.

The user approved one global FP16 native GPU TP sum default for all dense and
MoE models. That implementation removes the per-layer precision map, schema
FP32 counts and forced full-attention overrides. The FP16 threshold is now zero
by default, including serial decode rows. Explicit selectors remain supported;
graph-declared lossless assembly and canonical rank-order folds retain their
FP32 contracts. Both complete Release/Integration builds and 18 focused checks
passed. The complete gate passed 692 Unit and 664 of 665 preflight registrations;
`CapturedCollectiveMaintenanceProgress` deterministically failed because its
fixture omitted the setup FP16 scratch reservation formerly avoided by the
default threshold. All eight CUDA/ROCm subcases reproduce the same admission
failure with a clean driver window. The fixture now reserves the canonical
collective BOM through PhysicalMemoryAuthority before warmup/capture. Fresh complete Release/Integration builds, all 19 focused checks and the
complete r3 gate pass, including all eight maintenance subcases. The full gate
passes **1357 registrations: 692 Unit + 665 preflight**, with no failures or
skips, in 2101.071 seconds. The preflight lanes are 286 host, 131 CUDA, 160 ROCm
and 88 multi-device registrations. The qualified source/build identity SHA is
`69e0d332b8cbfa877c83511f7ad330bfb1afbe08c7454c8a612f983bd85e3116`;
the gate receipt SHA is
`a7520a5bdb33e7347bfb35af741cfe789c02be3786f9c5074aa9ab5f41037fb2`.
The receipt is `global-fp16-stage-timing-r3-prerequisites/`. Fresh default Release HTTP passes **45/45 checks on both CUDA2 and ROCm2**,
with complete evidence validation and clean driver/retirement windows. The
report `tp2-http-global-fp16-stage-timing-r3.json` has SHA
`9ba7b99eeff51e4762cfdf0b31991c511eeb8443daaa762cb3c9a44c4056cd82`.
Paired default/explicit-FP16 model timing now passes all 18 measured greedy
iterations per backend, with identical MTP work between both precision selectors
and clean complete driver windows. ROCm defaults reach **431.738/46.206 tok/s**
prefill/decode, or **1.737183x prefill**, against **248.528/37.428** single-card.
CUDA explicit FP16 reproduces **972.722/66.742** against **944.572/62.078** single
card. The two CUDA default prefill row medians differ substantially,
**741.853 and 975.178**, while token, MTP and chunk evidence agrees; their mean
is **858.516/66.917**. The single-card prefill bracket drifts only -0.291%. This
TP2 timing variation remains unresolved and cannot be discarded as a policy
win or arithmetic change. The named real-model CUDA stage observation now passes before the exclusive raw kernel cohorts.
The failed r2 gate stopped every dependent model job before admission.

The opt-in native CUDA observer now joins each canonical compute-stage capture
scope to its physical native nodes. A self-contained HTML/JSON/CSV report shows
GPU spans, covered interval unions and exclusive node ownership; concurrent
and nested intervals cannot be summed as elapsed time. Conditional bodies
remain opaque. The focused real-capture gate requires named RMSNorm/residual
stages and the existing captured-snapshot byte oracle. Its device-free report
checks pass. Fresh paired CUDA1/TP2 model observation also passes with exact
tokens, MTP ledgers and prefill chunks. Every native node is attributed in all
six exercised prefill parents: 657 stage scopes for each single-card 384/64
graph and 788 for each TP2 participant's 512/64 graph. Standalone
`cuda-native-stage-timing-global-fp16-r3/cuda{1,2}-stages/stage-timing.html`
reports retain JSON/CSV, driver and trace provenance. Native parents take
353.934/102.529 ms on CUDA1 and approximately 423/112 ms on CUDA2 under
instrumentation; these are final-replay GPU intervals, not whole-request or
canonical benchmark timing. The TP2 zero-collective DAG interval model remains
approximately 314.6 ms, or 330.1 ms with observed readiness gaps; this is not a
communication-disabled execution. The stage cohort receipt SHA is
`7c73c8fe5b7a410f9ce207d13388a12fc679a32a4a96240c1eaa10594f0e1fb2`. The initial combined
build stopped before any GPU gate because the standalone observer needed an
explicit C++20 target requirement. That requirement is fixed, and the fresh
combined build/gate receipt is retained as `global-fp16-stage-timing-r2-prerequisites/`.
CUDA, ROCm, testing and MTP skills now document the workflow and its CUDA-only
observer boundary. All new model and kernel cohorts consume only the successful
r3 receipt. The communication-free projection sweep passes all **35 exact
format/shape/row cases**, including each isolated physical kernel. Weighting
nine projection roles by the actual single-card 384+64+64 and TP2 512+64
schedules gives **422.936 ms / 230.867 ms = 1.831945x**. Every pipeline and
restored isolated graph passes its byte oracle. This projection aggregate
excludes attention, GDN recurrence, elementwise operations, tiny FP32 GDN
projections, replicated MTP sidecars and communication; it does not establish
whole-model kernel scaling. Standalone projection dispatch also does not prove
the real model's fused projection path selects the same partition policy; the
current native work audit checks that distinction. FFN gate/up is the largest weak contributor at
1.684619x; small full-attention K/V projections reach 1.491901x/1.633268x.

The model-free canonical-partials clear-removal experiment passes eight
alternating control/candidate cases. Poisoning partials with NaNs before first
warmup retains every output byte while removing all 16 captured clears. Full
FFN-down M64 takes **234.944 to 213.440 us**; the TP2 shard takes **130.112 to
110.048 us**. These are isolated diagnostic wins, not installed defaults.
The follow-up implementation removes the CUDA canonical partial clear under a
strict complete-live-extent overwrite contract. Focused capture/poison tests,
including explicit empty K partitions, pass. Current complete qualification
and both canonical Release HTTP cells pass as described above.
ROCm's corresponding NativeVNNI partial fold is register-resident and already
has no clear. Cache startup initialization remains under audit.

The unobserved eight-row bucket-cap cohort passes all **24 measured greedy
iterations**, with complete clean driver/retirement evidence. Single-card mean
of bracket medians is **945.000/62.039 tok/s**. TP2 caps 512, 448 and 384 reach
**978.329/66.893**, **1002.856/66.531** and **962.219/66.307**, respectively.
The 448 cap improves prefill 2.507% over 512 and reaches **1.061224x** single-
card scaling; the single-card prefill bracket drifts -0.414%. Tokens match in
every run. A follow-up artifact-bound audit verifies that all 18 TP iterations
have identical MTP work across 512/448/384 caps (141 accepted drafts, 276 draft
steps, 91 rejects, 114 verifier runs and 390 verifier rows). It also rechecks
all 24 actual token streams against Single. No bucket default is promoted yet;
current-source candidate HTTP qualification remains necessary. The audit is
`cuda-prefill-cap-mtp-work-audit-r3.json`, separate from the original receipt.
The **38-case raw decode/head sweep passes**
under the same frozen R3 source identity, with restored byte oracles, zero
reported local storage and clean driver/retirement evidence. Main-model
projection sums scale **1.697590x at M1** and **1.971946x at M16**; LM-head
sharding scales **1.948590x/2.562380x**. These exclude communication, nonprojection
work and replicated MTP draft execution, so they do not prove whole-decode
scaling. Small full-attention K/V still regress at M1 (0.787/0.870x).

The last qualified live-prefix binaries passed 690 Unit + 660 preflight tests
and all 45 canonical Release HTTP checks on each topology. Explicit FP16 with
threshold zero also passed all 45 HTTP checks on CUDA and ROCm, with native
FP16 transport declarations, validated evidence and clean complete driver
intervals. These are native binary qualifications, not image certificates.

| Frozen candidate | Single prefill tok/s | TP2 prefill tok/s | Prefill scaling | TP2 decode tok/s |
|---|---:|---:|---:|---:|
| CUDA explicit FP16 | 944.493 | 976.298 | 1.033674x | 66.750 |
| ROCm explicit FP16 | 248.511 | 431.656 | 1.736971x | 46.221 |

The ROCm candidate exceeds the revised prefill target. CUDA FP16 improves
prefill 33.60% over the former hybrid/8192 policy, but needs further work. All
24 greedy measured iterations per backend match their single-card token
sequence. CUDA schema/threshold-zero and FP16/threshold-zero have identical
MTP work; ROCm transport candidates have different MTP work, so their decode
changes are not matched-work kernel improvements.

Fresh paired CUDA graph observations preserve tokens, MTP ledgers and prefill
chunks. The FP16 512/64-row parents take approximately 427/112 ms under
instrumentation. Removing collective interval weights from the recorded DAG
models 246/70 ms before observed readiness gaps. This is not measured
communication-disabled throughput: event intervals include scheduling,
instrumentation and communication interference, while conditional bodies are
opaque. It still identifies communication as a substantial remaining cost.

The actual prefill schedules also differ: single-card execution uses
384 + 64 + 64 physical rows, while TP2 uses a 512-row capture with 448 live rows
followed by 64 rows. TP2 therefore computes 576 physical rows for the same
512-token prompt. A raw compute aggregate must use these schedules, rather
than compare identical M values and assume that proves model scaling.

The isolated FP16 channel sweep passes all 17 byte/payload/driver cases. Eight
installed channels average 1249.952 us at capacity 512/live 448; automatic
selection takes 1392.640 us. Forced 1/2/4/8/16/32 and adaptive capped-8 controls
do not improve on the installed setting. Full/partial/empty/single-row cases
also pass, and the FP32 reference takes 2459.072 us. No new channel default is
justified by these results.

Evidence remains local and ignored under `/tmp/qwen38-27b-tp2-tuning/`:
`cuda-frozen-transport-live-rank-fold-integrated-r1/`,
`rocm-frozen-transport-live-rank-fold-integrated-r1/`,
`http-cuda-fp16-live-rank-fold-integrated-r1/`,
`http-rocm-fp16-threshold0-live-rank-fold-integrated-r1/`,
`cuda-prefill-transport-dag-live-rank-fold-integrated-r1-r2/`, and
`cuda-fp16-native-channel-sweep-live-rank-fold-integrated-r1/`.

## Qualified live-prefix evidence before default promotion

The following retained investigation record predates the completed transport
cohorts and the newly approved global default. Pending statuses below are
historical; the current state is recorded above.

The newest live-prefix slice completes both full builds, including all CUDA
SM80/86/89/90 and HIP gfx906 spill checks. Its Release core is
`71beb5d7d6f6e18b3e0fdae930d9e26c49db3fe75c87a10b922f248f1e88535c`
and Integration core is
`b2aa01eb337c8a2f9bf561dfcdc46f482a82eb9c6189dd9081e782d2a1451f5c`.
All nine focused checks pass: head defaults, both live FP16 conversion cases,
three new inactive canonical-fold cases and three original overlap cases.
An added standalone MI50 benchmark changed test registration during the first
gate, so that transaction was canceled and retains its own source-drift and
terminal receipts. Production binaries remain unchanged. The first refresh
was interrupted during focused checks, with no complete gate receipt. The
detached `live-rank-fold-integrated-r1` refresh completes both builds and all
nine focused checks again with a clean driver interval. Its complete
**1350-prerequisite gate passes** in 2099.51 seconds: 690 Unit and 660 preflight
registrations, with no failures or skips in any XML receipt. The gate is
`/tmp/qwen38-tp2-prerequisites-live-rank-fold-integrated-r1/prerequisites.json`,
SHA256 `1d8c0be8cdda2a17cb957947f6fdd64fde471ee67de99e0e346eb7901e8a1ddd`.
All **80 conversion and 42 canonical rank-fold candidates pass** their exact
active/inactive byte oracles and complete clean driver intervals. Exact
profiling binaries, 31 native-event samples per candidate, compiler commands,
production source and static resource/ISA receipts are authenticated separately.
Each process times one kernel, with no model or communication in its graph.
At capacity 512/live 448/width 5120, median microseconds on the actual pair are:

| Backend | FP32 to FP16 | FP16 to FP32 | Degree-two canonical fold |
|---|---:|---:|---:|
| CUDA 0/1 | 18.18 / 18.18 | 18.88 / 18.88 | 35.07 / 35.01 |
| ROCm 1/2 | 31.95 / 31.94 | 31.92 / 31.89 | 44.16 / 44.02 |

ROCm degree-four folds take 68.04/67.18 microseconds at the same live prefix.
Native occupancy APIs report allocation ceilings of 100% CUDA and 80% ROCm;
these are not achieved-occupancy measurements. Logical traffic rates are not
hardware DRAM counters. Evidence is the two
`live-rank-fold-integrated-r1-{cast-native-profile,native-profile}/authenticated-native-profile-summary.json`
receipts under the investigation root. Both current-slice canonical Auto HTTP
cells **pass 45/45**: CUDA in 144.82 seconds and ROCm in 227.53 seconds. Their
full evidence validators pass, both driver intervals are clean, and native
vocabulary-gather declarations prove head sharding ran without an explicit
head override. The report is `tp2-http-live-rank-fold-integrated-r1.json`,
SHA256 `48fc2402ad2c7f060e5b73086f19bcd0a7142e424d611fb893de22598d3d1934`.
The promoted Auto head defaults therefore have fresh native binary HTTP proof.
The current CUDA frozen transport cohort completes all eight rows, three
measured iterations per row. All 24 exact greedy outputs match the single-card
baseline, and the complete driver interval passes. The mean of the two row
medians gives:

| CUDA configuration | Prefill tokens/s | Decode tokens/s | Prefill / single |
|---|---:|---:|---:|
| Single card | 944.49 | 62.06 | 1.000 |
| TP2 schema, current threshold | 730.78 | 55.00 | 0.774 |
| TP2 schema, FP16 threshold zero | 869.58 | 62.14 | 0.921 |
| TP2 explicit FP16, threshold zero | **976.30** | **66.75** | **1.034** |

Explicit FP16 improves prefill by 33.6% over the current TP transport. It remains
well below the 1.8x target. Changing transport also changes the default
candidate's MTP acceptance/work ledger; decode gains against that candidate are
whole-configuration throughput, not a comparison with identical verifier work.
The two threshold-zero candidates do retain identical MTP work. Single-card
prefill bracket drift is below 0.04%. Evidence is
`cuda-frozen-transport-live-rank-fold-integrated-r1/summary.json` under the
investigation root; native path authentication is still being completed.

The first benchmark counter observer fails because `BenchmarkRunner` deliberately
clears runtime collective records after preparing the captured graphs. Those
host declarations are not re-emitted by replay. Its failed receipt and clean
driver interval are retained; this is not evidence that the FP16 kernels failed.
Fresh paired HTTP observers collect capture declarations without that measurement
reset and compare exact tokens and the immutable terminal MTP ledger against
unobserved controls. The explicit CUDA FP16 candidate already **passes the full
canonical HTTP workload, 45/45**. Its actual native counters contain 1290 live-row
and 530 fixed-extent FP16 declaration records, plus four FP32 embedding records.
These records prove graph construction, not replay wire traffic. Full HTTP
evidence, validated driver closure and native type receipt are in
`http-cuda-fp16-live-rank-fold-integrated-r1/`; the report SHA256 is
`9e806ccf10a41f5afde6bad255717c6c299b0efb0803b77cdd015fb8cd819480`.
No transport precision default has been changed. The resumed exclusive campaign
finishes paired CUDA HTTP probes, then runs the ROCm timing/probe cohort and its
fastest candidate's full HTTP gate. A fresh CUDA DAG observation follows those
exclusive model jobs. The complete 1350-test gate and current binaries remain
unchanged.

The completed model cohorts below identify the earlier
`f986ca55...` core. Exact profiling binaries retain separate register/ISA
receipts in `live-rank-fold-profile-static-resources/receipt.json` under the
investigation root. New casts use 10 CUDA registers or 5 VGPRs/16 SGPRs; the
live rank fold uses 40 CUDA registers on SM86 or 15 VGPRs/23 SGPRs on gfx906.
All have zero spills, private scratch and shared memory. These are static
allocation records, not achieved-occupancy measurements.

The sharded-head divergence has a reproduced memory-lifetime cause and a
validated second-verifier correction. Before the fix, host-only capture
metadata showed `native_vocabulary_rank_major` and
`gdn_speculative_state_slots_grouped_mtp_verifier_layer0` starting at the same
physical address on **both** CUDA participants. The gather bank spans
15,892,480 bytes inside the 25,165,824-byte checkpoint bank. A live vocabulary
gather therefore overwrote recurrence snapshots before accepted-state restore;
the next verifier saw corrupted recurrence. The first verifier's head values
agree, and its post-publication local recurrence already differs, consistent
with this ordering. Setup enabled the recurrent verifier policy while
retaining a serial terminal output policy, so admission never saw both banks
in the same verifier lifetime. A complete verifier-policy/output declaration
and a focused real-graph regression are now implemented. Workspace-family
admission uses the same complete grouped-verifier policy as execution and
temporarily binds the actual all-position output owners. The scoped binding
restores the ordinary graph's policy and buffer selection after construction.
The regression failed before the correction and passes all six request/depth
geometries afterward in 683 ms; it is explicitly registered as
`V2_Integration_MTPWorkspaceVerifierOutputOwnership` in production preflight.
No kernel arithmetic, host readback, replay topology or communication policy
changed in this correction.
The observed runs retain identical tokens and MTP work relative to their
unobserved controls and have clean driver windows. Evidence is
`/tmp/qwen38-27b-tp2-tuning/cuda-terminal-head-fixed1-2-stage-bindings/`,
particularly `authenticated-checkpoint-gather-overlap.json` and the per-policy
interval comparisons. Previous correctness/timing receipts below identify the
previous Release core and do not certify the correction.

Both complete builds passed, including Integration's CUDA SM80/86/89/90 and
HIP gfx906 spill checks. The verifier-workspace Release core SHA256 is
`f986ca55a07174755b99ac9a4a9ec41e7f937e9d6560423814512cc3ed7084aa`;
the Integration core is
`6afab07d34d04c4c4519dcec94f5e9330d11bc178f1f3c3a452a80c24387a3c2`.
The full **1345 prerequisites passed**: 690 Unit and 655 model-free production
preflight registrations, in 2107.57 seconds. The authoritative receipt is
`/tmp/qwen38-tp2-prerequisites-verifier-workspace/prerequisites.json`, SHA256
`4f858f119cf4d1fc3f2db552e4c6ef2de0fae73d0b42b9c88fb176e2279fcec6`.
The fresh fixed-depth-one, four-token CUDA observation **passes** the formerly
failing second transaction. All fifteen retained matrix comparisons are
byte-identical, with matching geometry and finite values; both policies accept
two drafts in two verifier runs. All 96 available local GDN digests per
participant match, covering 48 convolution and 48 recurrence banks. The native
gather bank is disjoint from all 96 recurrent/convolution checkpoint banks on
each GPU, and cache storage is disjoint from those workspaces. Both observations
retain their paired unobserved tokens/MTP work and the driver window passes.
Evidence is
`cuda-terminal-head-fixed1-4-verifier-workspace/authenticated-verifier-workspace-evidence.json`
under the investigation root.

Both verifier-workspace-core canonical Release HTTP cells **pass 45/45**: CUDA in 150.22
seconds and ROCm in 239.96 seconds. The report is
`/tmp/qwen38-27b-tp2-tuning/tp2-http-verifier-workspace.json`, SHA256
`3ad6964418bf089177569c82d7fb5ae521898096bc99472a1421d68c5025585e`.
In this certified core, CUDA Auto selects the mirrored head; ROCm Auto selects vocabulary
sharding. The previously failing ROCm tool and long-generation cases pass,
including 177 ordered lines over 2048 generated tokens. Both cells validate
stochastic dynamic MTP, prefix reuse, complete captured parents, required head
ownership/collectives and terminal-only GPU D2H, with clean driver windows.
This is current-binary HTTP evidence, not a Docker image certificate.

The fresh CUDA head cohort uses the unchanged physical plans, current core and
new prerequisite receipt. Each of six bracketed rows has one warmup and three
measured iterations. Means of the two row medians are:

| Topology and head | Prefill tokens/s | Decode tokens/s | Prefill scaling | Decode scaling |
|---|---:|---:|---:|---:|
| CUDA:1, mirrored single-card brackets | 944.00 | 62.01 | 1.00x | 1.00x |
| CUDA:0/1, mirrored | 728.36 | 52.03 | 0.772x | 0.839x |
| CUDA:0/1, vocabulary sharded | 730.43 | 54.93 | 0.774x | 0.886x |

Sharding improves decode **5.59%** over mirroring. All 18 measured iterations
emit the identical 256-token single-card sequence; all 12 TP2 iterations have
the same MTP ledger (139 accepted drafts, 280 draft steps and 116 verifier
runs). Single-card bracket drift is -0.52% prefill/-0.08% decode. The driver
window passes. Evidence is
`cuda-frozen-head-verifier-workspace/authenticated-head-cohort-summary.json`
under the investigation root. The equivalent ROCm cohort also passes with
clean driver evidence:

| Topology and head | Prefill tokens/s | Decode tokens/s | Prefill scaling | Decode scaling |
|---|---:|---:|---:|---:|
| ROCm:3, mirrored single-card brackets | 248.70 | 37.36 | 1.00x | 1.00x |
| ROCm:1/2, mirrored | 382.60 | 40.80 | 1.538x | 1.092x |
| ROCm:1/2, vocabulary sharded | 383.33 | 42.40 | 1.541x | 1.135x |

ROCm sharding improves decode **3.93%** over mirroring. All 18 measured
iterations match the single-card token sequence; all 12 TP2 iterations share
the same MTP ledger (139 accepted drafts, 282 draft steps and 117 verifier
runs). Single-card bracket drift is +0.05% prefill/-0.21% decode. Evidence is
`rocm-frozen-head-verifier-workspace/authenticated-head-cohort-summary.json`.
This supports the installed narrow ROCm sharding default, alongside its
canonical Auto HTTP pass. The explicit CUDA sharded-head canonical HTTP cell
also **passes 45/45**, in 145.60 seconds, with complete evidence validation and
a clean driver window. Its report is
`http-cuda-sharded-verifier-workspace-canonical-lane/report.json`, SHA256
`12550f278cb5c3d3bc1a5bf9d0eb1e13e21a50415a574988cef6329834446ca2`.
The first local wrapper attempt passed response checks but incorrectly tagged
GPU memory telemetry as CPU; the retained failed attempt is separate from
this corrected complete lifecycle. The canonical `harness_backend` helper now
selects the GPU `tp` evidence lane. The measured CUDA sharding default is ready
for the next production change and its fresh prerequisite/Auto HTTP gates.

Matched depth-two controls expose an expensive depth-three choice in the
current dynamic policy. These are explicit tuning configurations, not new
depth defaults. One warmup and three measured iterations per bracketed row,
with identical physical placement and the same single-card/TP2 depth limit,
give the following means of row medians:

| Matched dynamic max-2 configuration | Prefill tokens/s | Decode tokens/s | Prefill scaling | Decode scaling |
|---|---:|---:|---:|---:|
| Single RTX 3090 | 935.07 | 91.66 | 1.00x | 1.00x |
| Two RTX 3090, sharded head | 727.67 | 115.31 | 0.778x | 1.258x |
| Single MI50 | 248.20 | 55.30 | 1.00x | 1.00x |
| Two MI50, sharded head | 383.41 | 74.71 | 1.545x | 1.351x |

All measured token sequences match, all matched dynamic-depth MTP ledgers
agree, and both driver windows pass. Relative to the unchanged single-card
depth policy these TP2 results are 1.859x CUDA and 1.996x ROCm decode, but that
comparison changes speculative work and is not TP scaling. The 1.8x target
remains open on both phases/topologies. Evidence is the two
`{cuda,rocm}-frozen-mtp-matched-verifier-workspace/authenticated-matched-depth-summary.json`
receipts under the investigation root.

Independent mirror/shard/shard/mirror depth-two cohorts also agree on every
token and MTP ledger. CUDA decode rises from 110.56 to 114.99 tokens/s
(**4.00%**); ROCm rises from 71.29 to 74.65 (**4.71%**). Their prefill rates
are 725.67/724.22 CUDA and 382.69/382.86 ROCm. Both explicit depth-two sharded
HTTP cells pass **45/45**, including stochastic sampling, prefix reuse,
long-context/tool requests and complete capture evidence, with clean drivers:
CUDA 114.25 seconds and ROCm 209.26 seconds. Report SHA256s are
`0d6d34f772b088cba310a1e0560cb145bef65ad42574a5df51b2f29938f02d87` and
`d2b89295f31574919db91a47237fb68dde35afb839a5b9c714182fdba9d1f816`.
These qualify automatic head sharding for the measured two-/fifteen-draft
dynamic families on local RTX 3090 and MI50 pairs. That production change is
now installed alongside the conversion correction below. Both complete builds
pass, with new Release core
`a6d4cef5125714e7a9b1009fe7406f51d88628c64cc5546cb10afaa9fbcef92a`
and Integration core
`74c1e92a871b149ff6b40789041f60413e4055c3c278a6726fdf3bc76fe0a36b`.
The focused defaults and both FP16 inactive-row regressions pass. The complete
1347-prerequisite transaction passes all 690 Unit and 282 host cases, then
fails the strengthened CUDA canonical rank-order overlap case. Its failure
stops the other GPU lane and all queued profiles/models. Fresh Auto HTTP
certification remains pending. Wider
depth-policy performance evidence is still required before changing the
automatic depth limit.

New model-free communication controls reproduce **inactive FP32 row
modification under FP16 allreduce conversion on both backends**. The same
captured sideband fixture passes FP32, fails FP16, and retains clean driver
windows. This defect must be corrected with a focused preflight registration
before promoting a lower transport threshold. A focused before-fix sweep
fails on both backends with clean drivers. Its ordinary and sideband FP32/FP16
cases cover every count 0–17 plus full/empty/partial replays. The installed
correction makes both precision conversions consume the same device-owned
live prefix as native allreduce; inactive rows retain their original bytes.
The focused CUDA and ROCm registrations are explicit production preflight
entries. The focused before/after proof now passes on both backends, alongside
the topology-default Unit test, with a clean driver window. CUDA SM80/86/89/90
and HIP gfx906 builds pass. Both live conversions use 10 CUDA registers, or
5 ROCm VGPRs/16 SGPRs, with no spills, shared memory or private scratch. Exact
profiling copies have the same allocation as production. Native-event profiles
are queued after the full prerequisite gate; static allocation is not achieved
occupancy. Evidence is
`communication-safety-verifier-workspace/` under the investigation root.

The broader inactive-row guard exposes a separate **canonical rank-fold
extent defect**. Native allgather communicates only the live prefix, but its
following fold still visits the capacity bank, reading stale scratch and
republishing it into inactive in-place output. Focused before-fix cases
reproduce this on CUDA TP2 and ROCm TP2/TP4 with clean driver windows. Evidence
is `live-rank-fold-before/summary.json`, against Release core `a6d4cef...`.
The symmetric correction is installed: a typed `NativeCollectiveRows`
overload retains the capacity stride between rank banks and clips the fold to
the canonical device-owned live count. Every live value keeps the same explicit
ascending-rank FP32 addition order; empty replays touch no output values. No
new storage, host count readback, communication padding or synchronization is
introduced. Three focused regressions explicitly enter production preflight,
covering counts 0–17 and full/empty/partial replays. Both full architecture
builds, all nine focused cases and the complete 1350-prerequisite gate pass.
Reduced-precision model tuning has not started on this source slice.

SHM receiver-copy mode passes isolated, mixed one-/four-row and large CUDA
prefill reductions through both ordinary and device-counted NCCL entrypoints.
It is slower than direct SHM: median 3137.66 versus 2434.30 microseconds at
capacity 512/live 448, and 3603.07 versus 2706.37 at 512/512. All byte checks,
counted payload receipts and driver windows pass. These bounded controls do
not reproduce the earlier stall and establish no winner for receiver-copy.
The full sample arrays and exact commands are in
`shm-sequence-verifier-workspace/`. Fixed-depth-one/two/three CUDA capacity
controls now measure the unchanged FP32 path before transport precision work.

## Previous-core measurements and investigation

The installed dispatch, communicator and serial-row correction passed **1344**
complete prerequisites: 690 Unit and 654 model-free production preflight
registrations, in 2095.05 seconds. This includes 1008 CUDA and 756 ROCm installed-choice
format/geometry cases. The receipt is
`/tmp/qwen38-tp2-prerequisites-serial-projection-rows-final/prerequisites.json`,
SHA256 `d6c08fd375973b9aa6df8479555b948235a1b5102a59d54791207c43a9774c4d`.
The full Integration architecture set is CUDA SM80/86/89/90 and HIP gfx906;
the native Release build is SM86/gfx906. Both passed the spill guard.

The previous serial-row-corrected Release core is
`3a9146b0300fb3aec25bdf77e135aa0de5563662c3ac92e05a101ddf287139f7`.
Its canonical HTTP results were **CUDA 45/45 passed** in 150.07 seconds and
**ROCm 42/45 failed** in 229.44 seconds. CUDA exercised stochastic dynamic
MTP, long recall, JSON/tools/SSE, prefix reuse, retained complete capture and
terminal-only GPU D2H. Both servers retired cleanly with clean driver windows.
ROCm required/automatic tool calling and two long-generation requests failed;
its native RCCL trap no longer occurs. The aggregate report remains red:
`/tmp/qwen38-27b-tp2-tuning/tp2-http-serial-row-fix.json`, artifacts
`e2e-1791089957630966666/{1,2}`. That core did not certify ROCm or vocabulary
sharding; the current workspace correction and evidence supersede this result.

The previous RCCL trap exposed serial projections borrowing prompt/verifier
lengths instead of publishing one committed row. The correction retains a
device count only for grouped verifier graphs. Its focused before-fix failure,
native GPU core/backtrace and new serial/grouped preflight proofs remain under
`/tmp/qwen38-27b-tp2-tuning/`.

The previous-core CUDA medians use one frozen physical placement and an exclusive
single/mirror/shard/shard/mirror/single cohort. Every row contains three
measured iterations after one warmup; single-card brackets bound drift.

| Topology and head | Prefill tokens/s | Decode tokens/s | Matches single greedy tokens |
|---|---:|---:|---|
| CUDA:1, Auto mirrored | 946.76 | 62.10 | Yes |
| CUDA:0/1, mirrored | 723.22 | 52.25 | Yes |
| CUDA:0/1, sharded | 726.76 | 40.76 | No, first difference at index 124 |
| CUDA:0/1, sharded repeat | 726.65 | 40.67 | No, same difference |
| CUDA:0/1, mirrored repeat | 727.83 | 51.98 | Yes |
| CUDA:1, Auto mirrored bracket | 943.92 | 62.02 | Yes |

Mirrored TP2 therefore reaches approximately **0.77x prefill and 0.84x decode**.
Sharding gives negligible prefill improvement, reduces decode and changes the
MTP acceptance ledger: 88 accepted drafts/168 verifier runs versus mirrored
139/116. Stable raw head equivalence does not certify that full-model
publication/consumption. Evidence is
`/tmp/qwen38-27b-tp2-tuning/cuda-frozen-head-serial-row-fix/`.
The equivalent frozen-pair ROCm cohort completed with a clean driver window:

| Topology and head | Prefill tokens/s | Decode tokens/s | Matches single greedy tokens |
|---|---:|---:|---|
| ROCm:3, Auto mirrored | 248.54 | 37.51 | Yes |
| ROCm:1/2, mirrored | 383.01 | 40.71 | Yes |
| ROCm:1/2, sharded | 383.39 | 41.01 | No, first difference at index 33 |
| ROCm:1/2, sharded repeat | 383.19 | 40.91 | No, same difference |
| ROCm:1/2, mirrored repeat | 382.61 | 40.70 | Yes |
| ROCm:3, Auto mirrored bracket | 248.72 | 37.35 | Yes |

ROCm mirrored TP2 reaches **1.54x prefill/1.09x decode**. The sub-1% sharded
timing difference does not establish a promotable winner when its output and
acceptance ledger differ. Earlier unconstrained head controls used different
physical pairs; this cohort removes that confound. Evidence is
`/tmp/qwen38-27b-tp2-tuning/rocm-frozen-head-serial-row-fix/`.

A fresh CUDA communication control holds this same frozen placement, mirrored
head, kernel policy and workload fixed. The bracketed single-card means are
945.44/62.08 prefill/decode tokens/s; the installed eight-channel policy reaches
725.99/52.12 versus native automatic channels at 661.82/51.22. This establishes
**9.70% prefill and 1.77% decode improvement from the channel policy**, with
identical output tokens and identical TP2 MTP ledgers in every measured
iteration. The native control changes only the exact communicator policy ABI;
one main communicator binds both participants, while the replicated sidecar
and mirrored head need no communicator. All raw samples and clean driver
receipts are retained in
`/tmp/qwen38-27b-tp2-tuning/cuda-frozen-channel-control-eligible-groups/`.

A terminal-only CUDA observer retained identical outputs and MTP work relative
to each unobserved control. The engine's captured greedy proposal record gives
identical first-transaction activation and scored-logit hashes for every
recorded slot on both participants. The complete stochastic outcome record is
unavailable in greedy mode; this diagnostic cannot certify later verifier
rows or continuation. No hot-path host readback or production policy changed
for these diagnostics.

Terminal matrix observations now cover paired eight-, 32- and 64-token CUDA
requests, each with its own unobserved production control. Every observed
request retained identical output tokens and MTP work relative to that control,
and every driver window passed. Immutable captured head geometry owns the
matrix strides; a later serial arena view is not a grouped-row contract.
At eight and 32 tokens, head inputs, all live full-vocabulary logits, terminal
hidden state and MTP acceptance ledgers match byte-for-byte between policies.
At 64 tokens, output tokens still match, but mirrored execution accepts 43
drafts in 21 verifier runs, versus 41 in 22 runs for sharding. Both finish at
main position 575 and sidecar position 574. The first GDN layer's retained
local convolution digest matches while its recurrence digest differs; terminal
hidden values differ too. The gathered logits still exactly reproduce the
assembled local head outputs on both participants. This terminal observation
does not identify the first faulty transaction or prove a cause, but directs
the next investigation toward MTP state advancement/rollback and fixed-depth
controls before further head-policy promotion. Evidence is
`/tmp/qwen38-27b-tp2-tuning/cuda-terminal-head-short{8,32,64}-capture-geometry/`,
including live matrix byte comparisons and authenticated observer controls.

Fixed-depth controls now exclude the dynamic depth controller as a necessary
trigger. At depth one, a two-token request agrees through the first verifier:
all live normalized inputs, full-vocabulary logits, terminal hidden and MTP
hidden rows are byte-identical. A four-token request still emits the same
tokens, accepts the same two drafts, and verifies the same token/position rows,
but the second verifier's normalized hidden rows and logits differ. At twenty
tokens the mirror accepts 10 drafts in 10 verifier runs, versus 4 in 15 for
sharding; at 64 tokens the ledgers are 32/32 versus 9/54. The longer runs have
different retained main/sidecar positions, so their terminal matrices are not
matched-transaction arithmetic comparisons. Local GDN recurrence hashes
already differ after the first transaction; retained full GDN banks are
entirely zero in these controls and cannot certify live-state equivalence.
This bounds the first matched-input verifier difference to the second
transaction, without yet assigning its cause. Evidence is
`/tmp/qwen38-27b-tp2-tuning/cuda-terminal-head-fixed1-{2,4,20,64}/`.

An authenticated, host-only inspection of the sealed CUDA capture fragments
finds the same non-head kernel names, geometry and compiler resources between
head policies. The actual Q6_K head uses K-parallel width 128, three K
partitions and a fixed-order reduction on both paths. Mirror/shard vocabulary
grids differ as expected; their grouped heads reuse scratch in row batches of
3/6. Additional probes feed the retained real model normalized rows from both
64-token CUDA runs into the actual full and independently prepared half-vocab
heads, replay every live count 0–16, and compare **every** grouped row to its
serial M=1 oracle. CUDA and ROCm both pass all bytes and inactive guards with
clean driver windows. ROCm uses those same saved CUDA activation inputs with
its own serial oracle. This proves the exercised head arithmetic and batching,
not full-model continuation or all possible numerical inputs. Evidence is
`cuda-terminal-head-capture-owner8/` and
`{cuda,rocm}-real-activation-head-dynamic64/` under the same investigation root.

Communication-free fixed-input head comparisons passed on **both** backends
at M=1, 2, 4, 8 and 16. Fresh probes additionally passed the production fused
verifier API with synthetic and actual model Q6_K weights: one retained graph
replayed every live count from zero through sixteen, including shrinking and
growing sequences. Full projections under the partition-equivalence scope were
byte-identical to assembled half-vocabulary projections and to serial first-row
publication; inactive rows and guards stayed poisoned. Both driver windows
were clean. These are fixed-input kernel proofs; full-model token and MTP
ledger comparisons remain separate. Evidence is under
`/tmp/qwen38-27b-tp2-tuning/{cuda,rocm}-fused-head-partition-serial-row-fix/`.

The expanded ordinary-prefill corpus measures these installed projection
ratios without communication. Each cell retains five warmups, 31 native-event
samples, its exact physical producer and a byte certificate. Full and local
geometries may use different cards of the same model, so these are isolated
kernel ratios rather than full-request critical-path measurements.

| Projection | CUDA M384 | CUDA M448 | ROCm M384 | ROCm M448 |
|---|---:|---:|---:|---:|
| IQ4_XS FFN gate/up | 1.873 | 1.973 | 1.967 | 1.997 |
| IQ4_XS FFN down | 1.955 | 2.048 | 2.052 | 2.036 |
| Q5_K GDN QKV | 1.746 | 1.803 | 1.850 | 1.761 |
| IQ4_XS GDN gate | 1.807 | 1.632 | 1.973 | 1.933 |
| Q5_K GDN output | 2.047 | 1.958 | 2.071 | 2.092 |
| IQ4_XS full-attention K | 1.144 | 1.235 | 1.442 | 1.437 |
| Q5_K full-attention V | 1.194 | 1.196 | 1.526 | 1.517 |
| IQ4_XS full-attention output | 1.888 | 2.015 | 1.909 | 1.862 |

The complete timing/source records and standalone PNG/SVG figure are
`/tmp/qwen38-27b-tp2-tuning/exercised-prefill-installed-kernel-scaling.*`.
Skinny K/V projections scale less than the large FFN and output projections;
the GDN recurrence retains its fixed arithmetic and its separate measured
scaling limitation.

The **1.8x end-to-end target remains unmet**. Localize full-model head
divergence before promoting defaults, then continue kernel and communication tuning.
Historical measurements below describe their own
source and workload; they are not interchangeable with this latest cohort.

## Hardware and workload

- CUDA: two RTX 3090 cards, PCIe PXB, without NVLink. Native NCCL selects SHM;
  the installed driver reports peer reads unsupported in both directions.
- ROCm: two of four MI50 cards, gfx906, using the selected ROCm 10 SDK and
  repository-authenticated HIP/RCCL runtime repairs.
- Model: `Qwen3.8-27B-IQ4_XS.gguf`, 64 main layers (48 GDN, 16 full attention),
  hidden width 5120, FFN width 17408, vocabulary 248320. The terminal head is
  Q6_K; dominant FFN/gate matrices are IQ4_XS and GDN QKV/output matrices Q5_K.
- Timing: Release, 512 exact prompt tokens, 256 generated tokens, dynamic MTP,
  FP32 activations, FP16 KV, one warmup and three measured iterations.
  Compare consistent medians. The head-placement diagnostic uses context 8192
  throughout; canonical ROCm single-card certification has a different context.
- Isolated probes execute real production kernels with valid fixed inputs and
  no communication. Timings use native graph events, retain samples, and require
  a complete output-byte certificate. They cannot be summed into an inference
  critical path because cache and overlap conditions differ.

## Starting evidence

Fresh canonical CUDA medians before this branch's production changes:

| Topology | Prefill tokens/s | Decode tokens/s | TP2/single |
|---|---:|---:|---:|
| One CUDA card | 918.15 | 62.07 | — |
| Two CUDA cards | 637.38 | 50.81 | 0.694 / 0.819 |

Single-card prefill executes physical rows 384 + 64 + 64. TP2 executes a padded
512-row bucket for 448 live rows, followed by 64 live rows. Equal-M raw scaling
is useful kernel evidence but does not explain this bucket difference away.

Selected CUDA raw TP1/TP2 times, in microseconds, excluding communication:

| Operation | M | TP1 | TP2 | Ratio |
|---|---:|---:|---:|---:|
| IQ4_XS FFN gate/up projection | 512 | 1226.75 | 707.14 | 1.735 |
| IQ4_XS FFN down projection | 512 | 1202.88 | 615.23 | 1.955 |
| Q5_K GDN QKV projection | 512 | 992.45 | 544.19 | 1.824 |
| IQ4_XS GDN gate projection | 512 | 483.07 | 358.02 | 1.349 |
| Q5_K GDN output projection | 512 | 597.44 | 329.98 | 1.811 |
| GDN recurrence, 48/24 heads | 512 | 556.86 | 304.13 | 1.831 |
| IQ4_XS FFN down projection | 64 | 237.82 | 212.22 | 1.121 |
| Q5_K GDN QKV projection | 64 | 171.97 | 185.66 | 0.926 |

The actual mirrored TP2 head's output-partition equivalence scope was retained
in its probe. Its Q6_K four-row projection measured 2707.58 us, versus
780.16 us for the half-vocabulary projection. At M=1 the corresponding times
were 1284.48 and 648.00 us. Gather cost and full-model acceptance behavior are
still required before choosing the default head policy.

## Implemented candidates and proofs

The previous vocabulary-sharded GPU MTP mode could not construct its
participant-local verifier outcome. The candidate now retains sharded weights
and adds an explicit participant-local NCCL/RCCL vocabulary allgather before
sampling and verification. Projection ownership and completed-logit publication
are separate typed facts. A verifier authenticates the actual gather producer
and full output tensor; a policy flag alone cannot authorize verification.

The gather transmits only the device-owned live row prefix. A persistent
rank-major bank retains capacity strides, followed by a bit-preserving live-row
transpose; a one-row bank needs no transpose. Both backend integration tests
passed 20 large/partial/empty/growing replays, reversed physical ordinals,
NaN/signed-zero bytes, poisoned inactive tails and exact native payload counts.
Focused device-free regressions cover missing/foreign publication producers,
retained-capacity policy and one/four/sixteen-request sidecar construction.

The assembly launch grid uses one occupancy-sized resident wave instead of
launching a block for every admitted row element. All copy-byte checks passed.
At capacity 16, ROCm full-live time improved from 74.07 to 68.98 us and a
one-live-row replay from 51.68 to 12.26 us. Isolated rocprof traces report
16 allocated VGPRs, 48 SGPRs, zero scratch and zero LDS. CUDA full-live time
improved from 47.55 to 42.30 us and the one-live-row case from 13.44 to 3.71 us.
CUDA compiler resource evidence covers SM80/86/89/90 with zero local storage.
CUDA hardware counter profiling is unavailable on this host; prior Nsight attachment
with this driver caused assertions and Xid 31/43. The recorded occupancy is a
compiler/runtime ceiling, not achieved occupancy.

ROCm isolated counter launches succeeded against the exact persistent copy
kernel. The 480-block launch reports 1920 wavefronts. Counter collection is
separate from the timing record, and setup kernels are excluded from the
candidate's resource evidence.

The common dense-prefill tournament completed all 210 planned CUDA cells:
five physical shapes, M=64/512, all 21 source formats. It produced 160 pooled
execution-codebook entries, retaining 25168 unrelated installed entries and
the existing selector. Every candidate and raw timing sample remains in the
local corpus; only complete, byte-certified cells entered generation. For
IQ4_XS/IQ4_NL aliases, measured speedups over Auto include 1.74x for the
64-row GDN gate, 1.73x for the 64-row FFN down projection, and 1.31x for the
512-row GDN gate. A sweep defect that treated compiler-excluded spilling tiles
as unexpected query failures is covered by the all-codebook resource gate.

The ROCm ordinary-prefill tournament completed 252 cells: those five shapes
plus the 512x5120 TP-local attention K/V projection, M=64/512, all 21 formats.
The common generator emitted 192 execution-codebook entries. A typed runtime
consumer now validates physical producer ownership and uses those exact keys
before generic dispatch. Native low-bit and Q8 blockwise INT8 retain separate
ABIs. Diagnostic tournaments suspend the table on their construction thread
so an installed choice cannot certify itself against its own Auto baseline.
The captured installed-choice proof passed all 252 format/geometry cases, including physical-producer ownership, zero-spill resource admission, poison guards and exact serial-row bytes. It then passed in the complete 1319-test prerequisite transaction.

Selected ROCm raw TP1/TP2 times, in microseconds, excluding communication:

| Operation | M | TP1 | TP2 | Ratio |
|---|---:|---:|---:|---:|
| IQ4_XS FFN gate/up projection | 512 | 5446.27 | 2816.23 | 1.934 |
| IQ4_XS FFN down projection | 512 | 5570.52 | 2789.09 | 1.998 |
| Q5_K GDN QKV projection | 512 | 4160.73 | 2508.42 | 1.659 |
| IQ4_XS GDN gate projection | 512 | 1964.29 | 1061.21 | 1.851 |
| Q5_K GDN output projection | 512 | 3004.59 | 1271.08 | 2.364 |
| GDN recurrence, 48/24 heads | 512 | 1027.15 | 973.92 | 1.055 |
| Full attention | 512 | 2677.91 | 1530.96 | 1.749 |
| IQ4_XS attention K projection | 64 | 192.43 | 192.15 | 1.001 |

ROCm GDN remains a known scaling limitation: halving the heads reduces its
column-split grid from 96 to 48 blocks on a 60-CU device. Its fixed key
partition/reduction order has not been changed.

## Communication and rejected kernel candidates

The native CUDA communication matrix uses 16 captured operations per sample,
31 samples and five warmups, with actual outgoing-byte receipts, poisoned
inactive rows and driver-health windows. Peer access is absent; SHM/direct is
the authenticated transport. Eight channels reduced the 448-live-row,
512-capacity hidden-state all-reduce from 2938.11 to 2478.27 us. The one-row
reduction remained approximately 21 us; four-row reduction approximately
50 us. One/four-row vocabulary gathers improved from 197.25/769.34 to
182.08/656.38 us. Every large/partial/empty case passed its byte/extent proof.

A matched full-model A/B compared eight channels with an immediate default
repeat on the same Release binary. CUDA TP2 prefill was 725.60 versus 657.47
tokens/s (1.104x); decode was 52.28 versus 51.07 (1.024x). Generated token IDs
were identical. These early environment-controlled diagnostics preceded the
installed typed eight-channel communicator policy; the fresh frozen-placement
controls in Current status measure that installed policy against native
automatic channels. Protocol,
thread-count and native buffer-size candidates completed their byte/extent proofs without a better full-workload candidate. Forced Simple or LL, 512 KiB/1 MiB/8 MiB buffers and 512-thread limits did not justify another default.

The ROCm raw decode probe completed 60 full/half/quarter-width cases, including
actual mirrored-head partition equivalence, with serial-row byte certificates.
At M=1, FFN gate/down measured 85.88/83.81 us full versus 53.59/50.35 us half;
Q5_K GDN QKV was 73.07 versus 43.45 us. The Q6_K terminal projection was
1605.98/1876.25 us at M=1/4 full vocabulary versus 810.88/938.45 us half.
Gather latency and model-level work remain separate costs.

A ROCm GDN experiment changed only independent value-column ownership to use
128- or 64-thread blocks, retaining the fixed four key partitions, reduction
order and normalization additions. All nonzero-state, aliased-state,
normalization, head-count and large/partial/empty serial-row byte cases passed.
An interleaved 256/128/64/128/256 timing sequence rejected both candidates:
M=512, 24-head time was approximately 973/1254/3866 us respectively. The
original 256-thread production implementation remains the fastest measured
choice. This experiment does not alter GDN production arithmetic or dispatch.

## Validation and remaining work

The original source/runtime baseline passed all 1306 Unit/preflight tests.
The first complete candidate transaction then passed 1312 tests: 688 Unit and
624 model-free production preflight tests, in 1849.02 seconds. This includes
both native vocabulary gathers and actual sharded MTP graph construction.
Its receipt is `qwen38-tp2-prerequisites-native-head-prefill-final` under `/tmp`.

Fresh matched Release medians admitted by that transaction:

| Topology and head | Prefill tokens/s | Decode tokens/s | TP2/single |
|---|---:|---:|---:|
| CUDA one card, mirrored | 923.21 | 62.42 | — |
| CUDA two cards, mirrored | 660.13 | 51.10 | 0.715 / 0.819 |
| ROCm one card, mirrored | 248.42 | 37.42 | — |
| ROCm two cards, mirrored | 357.86 | 40.77 | 1.441 / 1.090 |

All four runs emitted identical greedy generated token IDs. Dynamic MTP work
counts differ between topologies and are retained alongside the throughput.
These are matched diagnostics, not canonical HTTP certification.

The sharded CUDA model then exposed a physical-memory admission defect before
capture: `prefix_device_tier` requested 238712832 bytes against 238216192
admitted bytes. The 496640-byte discrepancy is precisely one half-vocabulary
FP32 row. Its completed native publication owns the full vocabulary, while
prefix planning still priced the physical projection shard. The typed BOM now
records the explicit native publication independently of weight width; both
prefix slot sizing and transpose scratch use that fact. A focused regression
prices exact archive slots on both GPU backends and excludes CPU/distributed
shards and non-terminal pipeline owners. It is explicitly in preflight.

The prefix publication fix and ROCm installed dispatch passed the complete
1319-test transaction (689 Unit and 630 production preflight tests) in 1985.23
seconds. The receipt is
`qwen38-tp2-prerequisites-prefix-published-rocm-overlay-final` under `/tmp`.

Fresh Release controls admitted by that transaction:

| Topology and head | Prefill tokens/s | Decode tokens/s | TP2/single |
|---|---:|---:|---:|
| CUDA one card, mirrored | 903.33 | 61.89 | — |
| CUDA two cards, mirrored | 656.43 | 50.97 | 0.727 / 0.824 |
| ROCm one card, mirrored | 244.11 | 37.30 | — |
| ROCm two cards, mirrored | 383.56 | 40.91 | 1.571 / 1.097 |

All four runs retained the same greedy token sequence. Head sharding passed
prefix admission but exposed a separate serial-family setup omission on both
backends: the 16-row verifier needed 15892480 bytes for
`native_vocabulary_rank_major`, while serial setup had published zero. The
native stage ignored the family's requested operation M, and the workspace
scanner treated `lm_head_allgather` as a terminal projection by name. The fix
previews the compact family capacity while preserving the frozen replay
geometry and published address; runtime growth remains forbidden. Focused
CPU and native CUDA/ROCm regressions passed, followed by all 1323 prerequisites
(689 Unit and 634 preflight) in 2053.74 seconds. The receipt is
`qwen38-tp2-prerequisites-serial-vocabulary-workspace-final` under `/tmp`.

The repeated CUDA raw curves completed 81 exact-byte cases and 199 isolated
kernel timing records after dispatch installation. Selected equal-M results:

| Operation | M | TP1 us | TP2 us | Ratio |
|---|---:|---:|---:|---:|
| IQ4_XS FFN gate/up | 512 | 1207.74 | 658.43 | 1.834 |
| IQ4_XS FFN down | 512 | 1194.24 | 606.66 | 1.969 |
| Q5_K GDN QKV | 512 | 977.92 | 495.81 | 1.972 |
| IQ4_XS GDN gate | 512 | 477.57 | 282.69 | 1.689 |
| Q5_K GDN output | 512 | 593.66 | 307.84 | 1.928 |
| IQ4_XS FFN gate/up | 64 | 190.59 | 109.25 | 1.745 |
| IQ4_XS FFN down | 64 | 233.09 | 130.88 | 1.781 |
| Q5_K GDN QKV | 64 | 171.20 | 127.87 | 1.339 |
| IQ4_XS GDN gate | 64 | 155.78 | 66.75 | 2.334 |
| Q5_K GDN output | 64 | 192.96 | 75.14 | 2.568 |

The selected ROCm GDN 256-thread dispatch also has an isolated native trace:
124 VGPRs, 64 SGPRs, 512 bytes LDS and zero scratch for the 24-head,
512-row geometry. This is resource evidence, not achieved occupancy.

Canonical HTTP E2E and validation of newly installed automatic
terminal-head/communication defaults remain outstanding; the 1.8x target is
not reached.

The first refreshed Unit run exposed a missing CPU owner tier for the new TP
aliases and historical split expansion that rewrote a signed v12 identity.
The aliases now retain the full 27B owner's measurement tier. Historical
v10-v12 ownership resolves its original frozen inventory; current certification
includes the new shapes. Both focused regressions are explicit preflight
entries, and the 1312-test transaction passed. The signed v12 identity remains
`sha256:f66ec74e78387098e0f77a2a1e4a2898d8d72be88393de4dff51663f991626fe`.

Local evidence is under `/tmp/qwen38-27b-tp2-tuning/`: canonical baseline JSON,
head-policy commands/logs, CUDA raw-kernel JSONL, the 210-cell
`cuda-prefill-corpus-v2`, immutable installed base, generated candidate,
retention receipt, winner summary, ROCm raw curves and isolated copy profiles.
These generated results are not source-repository payloads. Published corpora
belong in `Llaminar/corpora` if publication is requested.

## October 4: complete head cohorts and installed candidates

After the 1323-test transaction, both head placements completed matched Release
cohorts. These are diagnostics, not HTTP certificates:

| Topology and head | Prefill tokens/s | Decode tokens/s | TP2/single |
|---|---:|---:|---:|
| CUDA one card, mirrored | 919.62 | 62.43 | — |
| CUDA two cards, mirrored | 657.02 | 51.11 | 0.714 / 0.819 |
| CUDA two cards, sharded | 652.06 | 39.65 | 0.709 / 0.635 |
| ROCm one card, mirrored | 246.23 | 36.83 | — |
| ROCm two cards, mirrored | 382.51 | 40.90 | 1.553 / 1.110 |
| ROCm two cards, sharded | 383.69 | 41.94 | 1.558 / 1.139 |

ROCm sharding retained the exact greedy token sequence and complete MTP work
ledger of its mirrored TP2 control. CUDA sharding first changed tokens at
index 124, reduced acceptance from 0.597 to 0.384 and selected less economical
draft depths. CUDA therefore retains mirroring while that discrepancy is
investigated; synthetic gather correctness alone does not establish model
projection equivalence.

A per-communicator eight-channel diagnostic using the canonical NCCL 22809
ABI passed the native live-byte matrix and reproduced the CUDA mirrored token
sequence and complete MTP ledger. Release prefill was 725.19 tokens/s and decode
51.73, versus 657.02/51.11 for native automatic channels. The source now selects
this profile only for a same-process pair of distinct SM86, 82-SM, 24-GiB RTX
3090 cards with authenticated absent peer access in both directions. Selection
is immutable before capture and recorded per device; other communicators keep
native selection. The changed production binary still requires validation.

Native-event observations covered the complete executed prefill chunk families
without CUPTI. Single-card physical rows were 384+64+64; TP2 rows were 512+64,
with 130 collective nodes per chunk. A model setting observed collective costs
to zero gave 561.78 ms single-card compute intervals and 300.54/308.26 ms on
the TP2 participants, or about 1.82x. This is a DAG diagnostic, **not measured
communication-disabled scaling**: observed compute intervals include scheduling
interference, conditional attention nodes remain opaque, and cross-device
readiness is not reconstructed. Observer overhead is excluded from canonical
timing comparisons.

Copy-engine sender transport was slower in the native matrix. Receiver mode
failed to make progress at the four-row case and was terminated at the
diagnostic deadline. Forced Tree modestly improved bulk all-reduce but slowed
serial reductions and could not implement AllGather under that forced policy.
Neither option becomes a default. Sender locality alone gave no useful gain.
The ROCm four-channel native default remains preferable to eight/sixteen
channels for its bulk workload.

The CUDA K/V extension collected 42 exact-byte cells at M=64/512 and generated
32 physical-codebook keys. Additional collection then completed **504 cells
per backend**: all 21 source formats, 12 full/TP-local geometries, M=384/448,
five warmups and 31 retained timing samples. Common generation preserved all
unmeasured installed keys and refreshed 384 keys per backend. CUDA now has
25632 installed keys, SHA256
`c6334d99a4559264cb25fa1457282bd960b01dc015e4000247b4b5f6fe4e6fe8`;
ROCm has 576, SHA256
`563b2c7a4c6c9ca29dee1aab3fcb1be1b96c33dd8a36c0ce6bbed2b0700255ec`.
The new CUDA installed-choice gate covers 1008 source-format/geometry/row cells
against independent generic Auto, with poisoned outputs and compiled resource
admission. The existing ROCm gate covers every expanded installed geometry.
Both must pass on rebuilt production source.

The complete SM80/86/89/90 and gfx906 Integration build and native Release
build passed the spill guard. The first expanded ROCm installed-choice group
hit its unchanged 120-second test budget without a numeric/resource failure.
Its failed receipt is retained. The fixture now constructs source weights once
per N/K across row buckets and partitions all 21 formats into twelve disjoint
registered groups, preserving all 756 cases and independent captured oracles.
The complete prerequisites have not yet been rerun for that final fixture.

The bounded focused run passed 25 of 26 registrations, including all ROCm
groups, both native serial-family captures, provenance, head defaults and the
cold CUDA context regression. CUDA's all-format producer check found that the
generic Q4_0 BK256 heuristic masked an exact BK64 choice at M=64, N=5120,
K=17408. Bytes matched, but the measured producer did not run. The selector now
consults that generic route only when no exact producer owns the cell, with a
focused captured producer/resource/full-byte proof explicitly in preflight.
Full rebuilding and prerequisites remain required for this final correction.

That correction passed the affected CUDA all-format group and focused producer
regression in 64.66 seconds, with no byte mismatch or driver anomaly. The next
full prerequisite transaction stopped at Unit: 689/690 registrations passed,
but the ROCm sparse-row test still hard-coded only M=64/512 and rejected the
newly measured M=384/448 cells. Its failed receipt is retained at
`qwen38-tp2-prerequisites-installed-defaults-final`. The test now derives the
measured row set from the canonical generated inventory and checks all M from
1 through 2048 against it, retaining exact lookup and foreign-geometry checks.

The measured dense-Qwen native Q6_K head on local MI50 TP2 now selects sharding
for the retained dynamic 16-row verifier family. Selection authenticates model
geometry, head provenance, complete physical membership, batch and sidecar
policy; explicit user choices remain authoritative. CUDA keeps mirroring.
HTTP evidence now requires declared head ownership and serial/verifier native
gather captures on each participant when native publication is selected.

The prefill cap remains 512. Cap 448 produced no consistent CUDA/ROCm winner;
CUDA's cap-384 trial instead exposed a cold context-footprint mismatch. Twenty
independent plan probes did not reproduce it. The source retains strict memory
authority equality, adds the exact requested/observed footprint to its fatal
diagnostic and registers a cold 5120-wide, 384-row context-retirement regression.
There is no retry, reserve adjustment or accounting relaxation.

The current verifier correction has completed both architecture-bound builds.
Next, complete its prerequisites, prove checkpoint/gather disjointness and live
state parity at the second verifier, run both canonical Release HTTP cells,
and measure new frozen-placement head cohorts. The 1.8x end-to-end target
remains unmet.

## Prepared controls after verifier validation

The canonical CUDA HTTP PerfStats declarations confirm that schema requests
for FP16 still execute native FP32 reductions at logical row width 5120. The
row-invariant minimum is 8192, independent of prefill/grouped matrix size.
The next exclusive transport cohort compares the current policy with schema
FP16 at minimum zero, retaining its forced-FP32 layers, and with explicit FP16
at minimum zero. Separate counter-only observations will authenticate actual
native dtypes and retain their own unobserved token/MTP controls. None of these
experiments is an installed transport change. The source audit is
`/tmp/qwen38-27b-tp2-tuning/fp32-transport-source-audit.json`.

The counted-verifier FFN probe also exposes uneven raw CUDA scaling at the
live counts used in decode. These IQ4_XS points use capacity 16 and exclude
communication; each value is the mean of two retained operation medians.
The operation timing wrapper did not retain its sample distributions, so this
table is exploratory and cannot promote a dispatch choice.

| Live rows | FFN gate/up full/local ratio | FFN down full/local ratio |
|---|---:|---:|
| 1 | 1.833 | 1.744 |
| 2 | 1.744 | 1.506 |
| 3 | 1.727 | 1.430 |
| 4 | 1.651 | 1.350 |
| 8 | 1.548 | 1.229 |
| 16 | 1.644 | 2.244 |

The row-reuse-two candidate lowers local down-projection latency at live four
from 62.788 to 44.632 us, but takes 176.400 us at live sixteen versus installed
99.588 us. Any dispatch change needs the common all-format certification and
an economical complete live-count domain. Fixed depths one through three,
with retained capacity held fixed and then changed explicitly, can test the
effect of launch geometry before controller defaults change. These controls
will follow the fresh head validation. Evidence is
`cuda-counted-ffn-installed-scaling-exploratory.json` and
`cuda-counted-ffn-candidates-current/exploratory-economy-comparison.json` under
the investigation root.
