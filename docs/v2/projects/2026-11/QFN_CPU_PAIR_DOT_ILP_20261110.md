# CPU ordered-pair integer ILP: qualified component and model results

The AVX-512 GPUAligned nibble pair now accumulates the original low/high16
integer products independently, then merges them before the original FP work.
Focused bytes and generated code pass. The QFN-shaped authored FFN improves
**1.93–2.54%** against its fresh final control. Five affected source formats
have closed role economics; some phase timings regress, as recorded below.
**The matched model passes token/MTP equivalence at 339.306 prefill and
40.566 after-prefill decode tokens/s. Full preflight remains the next gate.**

## Source and numerical contract

Only the existing ordered two-row nibble body in
`CPUNativeQuantisedGemv.h` changes. Four eight-update VNNI chains become eight
four-update chains plus four integer merges per32-column/K32 half. Original
prepared bytes, nibble/activation lane addresses, compensation storage, row
policy, Auto/exact18 choices, job ownership, output extents and K partitions
remain. No packing, public mode, point dispatch or new framework is installed.

For the signed decoder program, each16-value biased-A half is bounded by
522,240 in magnitude; full32-value sums by1,044,480. Even the conservative
full-sum/compensation combination fits INT32. Identity B0..15 with signed
A−128..127 has full dot−61,440..60,960. The original non-saturating INT32 VNNI
and the final integer add therefore produce identical integers. Compensation,
INT→FP conversion, scale/minimum products, original FP32 contribution, increasing
K-block additions and fixed ordered partition fold retain their original order.
No FMA or NaN-aware branch is introduced; original serial/grouped byte tests
remain the numerical authority, including raw special scales and signed zeros.

The initial ILP build spilled one identity/symmetric integer temporary. V2
reuses the **existing** decoder-specialized M1 scale helper: canonical linear
Q4_0 and identity Q4_1/Q4_K plain FP16 scales widen exactly as the general
loader's first operation; nonlinear IQ4_NL/IQ4_XS retain its original source-factor
equation. This removes impossible factor-branch register lifetimes without
changing the dot split or FP expression. It does not infer representation from
asymmetry. The original V1 source/ISA remains retained.

The role-independent pair affects all eligible nibble projections, including
nibble down. Current QFN Q5_1/other non-nibble down, M1, AVX2 and three-row
bodies are unchanged controls. These statements do not claim unrelated16source
formats became faster.

## Actual ISA and focused correctness

Root's frozen original FFN object inspection passes: no vector frame operands
in the complete V2 dispatcher, including the formerly spilling identity/symmetric
path; all eleven emitted static dot blocks retain32VNNI operations and eight
four-update chains. Both source banks and decoder/asymmetry branches are covered.
The separate LUT/asymmetric helper is instruction-text identical to V1, with
only original64-bit XMM pointer/metadata stack bookkeeping. No hot outlined
arithmetic call or FMA was found. Actual fused caller+0x279f/PLT32 reaches the
dispatcher; section headings are excluded from caller evidence.

Dispatcher size grows13,719→16,536bytes (V1:14,316). Additional static blocks
reflect bank/first-partition cloning, not extra per-executed-block dots. This
code growth is an economy risk, not a predicted throughput gain.
[Precise ranges and retained comparison](
/tmp/qfn-cpu-pair-dot-ilp-next/revisions/v2-scale-lifetime/isa-review/NOTE.md)
record the original object generation and limitations.

Four original native CTest registrations pass in **13.734s**: complete
`FusedCompletion` and new `NibbleDotILP`, each AVX2/AVX512. The full existing
owner retains all21source-format bytes; the new boundary case covers five
actual nibble factories, both banks, signed-Q8 extrema, nonzero source slices,
N65/K768 tails/empty partitions, workers1/3/7, changing live rows and Prefill/MTP
serial-byte equivalence. The new aliases explicitly enter production preflight.
[Registration/results/XML](
/tmp/qfn-cpu-eight-row-rocm101-root/pair-dot-ilp-v2-byte-gates/result.json)
are focused evidence, not complete preflight or allpositive-worker enumeration.

## Unprofiled authored FFN comparisons

Release, original `ProductionGeometryByteExactAndEconomical` owner under
original session/process/resource/driver ownership; NUMA1 physical cores28–55,
28workers, GNU waiting variables unset, aligned numerical policy, Auto row and
job ownership. Each phase retains128warmups,40samples,batch1; serial-oracle
warmup remains its original1. Timing suppresses PerfStats and fresh route/byte
proof runs outside clocks. Samples, exceptions and bursts are retained.

The E84/R3208 vector is **authored from marginal shapes**, not recorded joint
model routing. `A1/B1/B2/A2` below denotes original/candidate/candidate/original
fresh process lifetimes. Values are the owner's upper-middle medians for40
samples; linked raw summaries also retain middle-average, min/max, CV and
first/last-half medians, with no discarded observations.

Actual-QFN projection geometry: H2560/F320, Q4_K gate/up and Q5_1 down (µs):

| Phase | A1 | B1 | B2 | A2 |
|---|---:|---:|---:|---:|
| Gate/up |4093.0|3831.5|3835.3|4013.4|
| Unchanged Q5_1 down |2009.0|1913.0|1932.2|1920.4|
| Complete FFN |6674.7|6319.7|6293.9|6501.9|
| Persistent FFN |6663.3|6355.4|6395.6|6521.3|

A1 also has slower unchanged down/oracle work. Using A2 conservatively,
gate/up improves4.44–4.53%, persistent1.93–2.54%; unchanged down varies
−0.39/+0.62%. Do not attribute A1's whole difference to the candidate.
[All per-arm raw statistics](
/tmp/qfn-cpu-pair-dot-ilp-next/revisions/v2-scale-lifetime/repo-note/qfn-raw-summary.json)
and [original comparison](
/tmp/qfn-cpu-pair-dot-ilp-next/revisions/v2-scale-lifetime/component-comparison.json)
retain exact values and source paths.

Both affected roles use the SAME real source format in the five-format panel.
H2560/F768 keeps all five source block geometries legal and down on the changed
multi-block ordered path. F320 is not a legal standalone256-value source down
row; F512 would take the unchanged one-block shortcut. Neither is substituted
or padded to manufacture coverage. The E84/R3208 vector and runtime controls
remain unchanged.

Paired latency change B1/A1 and B2/A2 (%; negative is faster):

| Source, both roles | Gate/up | Down | Complete FFN | Persistent FFN |
|---|---:|---:|---:|---:|
| Q4_0 |+2.09/+1.57|+0.47/−0.58|+0.77/+0.13|−0.15/−0.10|
| IQ4_NL |+2.41/+1.98|+2.12/+1.98|+0.93/+1.68|+0.70/+0.07|
| IQ4_XS |+0.85/+2.07|+0.57/+2.78|−0.65/+0.62|+0.04/−0.17|
| Q4_1 |+0.13/−2.82|−3.14/−3.17|−2.85/−3.16|−2.71/−3.21|
| Q4_K |−1.18/−2.32|−2.85/−3.94|−2.31/−3.12|−2.62/−3.08|

Thus Q4_1/Q4_K complete transactions improve about3%; other persistent medians
are near flat, while linear/LUT isolated phases include repeatable slowdowns.
No universal phase gain is claimed. All4800chronological samples survive.
Examples of retained persistent ranges: IQ4_NL A2 spans18.209–27.699ms (CV8.3%);
Q4_K B2 spans15.016–15.825ms (CV0.8%).
[Every per-format/per-arm median, range and CV](
/tmp/qfn-cpu-pair-dot-ilp-next/revisions/v2-scale-lifetime/repo-note/five-role-raw-summary.json)
links the four PASS receipts under
`five-role-economy/economy/{baseline-retry1,candidate1,candidate2,baseline2}`.

The first five-format `baseline1` native passed bytes with native0/driver0,
but its private observer lost a terminal child between identity and fdinfo
reads. That receipt remains failed and excluded. A focused private fix journals
only independently proven retired/replaced PID/start lifetimes; live missing
metadata, permission and malformed data still fail. Original mock RED and fixed
GREEN are retained at `/tmp/qfn-private-resource-observer-exit-race-next/`.
Fresh `baseline-retry1` and all subsequent arms pass observer/driver/process/
session/closure proof. The failed artifact is never rewritten or relabeled.

## Isolated counters, separately from clocks

The original FFN request/output handshake profiles one aligned QFN-shaped
gate/up call, after2preconditioning calls and1fresh outside-profile proof.
Setup, serial oracle and down are excluded; it returns before canonical clocks.
Both current arms pass original lifecycle and all seven events have complete
running evidence. Single-launch worker aggregate results:

| Metric | Original | V2 | Change |
|---|---:|---:|---:|
| Cycles |234,606,429|227,796,315|−2.90%|
| Instructions |449,768,192|461,009,623|+2.50%|
| IPC |1.917|2.024|+5.56%|
| L1 load misses |29,824,087|29,687,983|−0.46%|
| LLC load misses |457,639|426,227|−6.86%|

This supports the intended tradeoff—more integer instructions with fewer
cycles—in this one diagnostic, not a cache cause or statistical performance
certificate. Existing team/control/planning and narrow PerfStats work remain
inside the interval; worker sums/spin are not removable elapsed overhead.
Profiled wall-clock does not replace unprofiled medians.
[Counter comparison and original/candidate receipts](
/tmp/qfn-cpu-pair-dot-ilp-next/revisions/v2-scale-lifetime/isolated-profile-plan/comparison.json)
retain exact request/CSV/closure paths. Older r1151 counters are historical,
not substituted as a matched current control.

## Real model and remaining gate

Release Qwen3.8-Flash-Next UD-Q4_K_XL with its Q8_0 MTP head, two MI50
continuation participants and two real CPU sockets using automatic MPI
bootstrap. CPU experts are tensor-sharded across sockets with 28 physical
workers each; GPU experts are apportioned. Both arms use nine transfer slots,
nine active cycles, four transfer streams, dynamic MTP (capacity 15), dynamic
expert balancing, 262,144 admitted context, M512, RAM ngrams with bounded local
GPU caches, and 16/32 GiB RAM/disk prefix-cache budgets. The benchmark uses
512 prompt tokens, 256 generated tokens, seed 42, one warmup and three measured
requests. Decode after prefill uses the same 255-token denominator.

| Implementation | Prefill tok/s | Decode after prefill tok/s |
|---|---:|---:|
| Qualified original nine-slot control |335.998636|40.798332|
| Ordered-pair V2 |339.306309|40.565618|
| Observed change |+0.98%|−0.57%|

All 768 generated token IDs and complete MTP records match. Both arms resolve
the same initial placement (7,940 GPU experts, 3,970 per card) and saved
configuration contents. Runtime configuration differences are confined to the
bound output, config and prefix-cache paths. Fresh admission passed in 8.291 s;
the complete model process lifetime was 250.902 s.

Dynamic movement remains active and its completed work differs: the control
has 96 promotions in 12 transactions with 941,056,000 physical bytes; V2 has
80 promotions in 10 transactions with 830,771,200 bytes. Thus the small model
rate changes are observations, not isolated causal speedup claims. V2 request
prefill rates span 333.297–350.409 and decode rates 39.931–41.034. The component
A/B/B/A clocks and isolated counters above own the narrower CPU gain.

The original native/driver owners pass with no errors and all owned processes
retired. The passive resource observer recorded 241 samples without errors;
the two inference ranks peaked at 36.47 and 77.75 GiB RSS, with zero sampled
swap. This short run does not establish a long-duration leak certificate or
PMA audit. Retired prefix cleanup removed 727,912,499 logical bytes across
18 files, reading only 28,675 metadata bytes and zero payload bytes.

[Matched model comparison](
/tmp/qfn-cpu-pair-dot-ilp-next/revisions/v2-scale-lifetime/model-arm/model-comparison.json),
[resource observations](
/tmp/qfn-cpu-pair-dot-ilp-next/revisions/v2-scale-lifetime/model-arm/model/native/resource-observation-result.json)
and [cleanup receipt](
/tmp/qfn-cpu-pair-dot-ilp-next/revisions/v2-scale-lifetime/model-arm/model/prefix-reclamation.json)
retain the original evidence. Checkpoint this bounded improvement, then run
complete Unit and ProductionTestPreflight before more tuning. The 500/40
hybrid target remains open; the prepared 17-slot experiment stays deferred.
