# QFN M448-only native-Q8 dispatch

The matched model measures **332.087 prefill /39.851 after-prefill decode
tok/s**. The two preceding continuation controls measured331.486/39.679 and
327.546/39.893. This is a small model-level difference within the recent
variation, not proof of a1.4% engine speedup. All768 output IDs, complete MTP
records, configuration apart from output paths and initial placement match.
The500/40 hybrid goal remains open.

## Narrow installed policy

Four canonical generated records select existing checked NativeCooperative
Q8 kernels at actual M448/N320/K2560 or10240:

| Actual architecture | N tile | M tile | Minimum blocks | Unroll |
|---|---:|---:|---:|---:|
| gfx906 |64|16|1|2|
| gfx1100 |64|32|1|2|

All576 preceding records and the selector remain byte-identical. No M64,
decode, MTP, CUDA, numerical policy or kernel implementation changes. The
gfx1100 entries retain its independently measured winner; they prevent a
nearby architecture's different tile from taking precedence. Missing exact
geometries continue through the existing complete general dispatcher.
This is exact M448 qualification, not an M>=128 policy interval.

The existing canonical collector, profile validator and additive emitter
authenticate both complete original source-alias corpora before selecting
this subset. Q8_0, Q8_1 and Q8_K remain separate source preparations. Original
M448 complete-projection timing shows geometric-mean1.106/1.112× on gfx906
and1.103/1.091× on gfx1100 for the two K values. Those component speedups do
not multiply the whole model. The prior full four-shape experiment, including
M64, had a decode regression and remains unpromoted.

Historical candidate profiles cover both actual cards and report no scratch
spills. This table change instantiates no new kernel. The canonical Release
build retains HIPgfx906/gfx1100 and CUDA80/86/89/90, with its mandatory spill
guard. Fresh native function receipts show gfx90671 registers/10,752B LDS
and gfx110083 registers/12,800B LDS for the selected variants, zero scratch;
these API-reported counts are distinct from rounded ISA allocation counts.
No unsupported occupancy/utilization metric is inferred.

## Focused qualification

Release build passes in48.736s. All17 focused registrations pass: three host
dispatch suites in0.123s and seven native suites each on MI50 (70.958s) and
7900 XTX (71.159s). These are original registered owners, including explicit
production-preflight aliases, not replacement launchers. The source-format
groups cover Q8_0/Q8_1/Q8_K; targeted probes capture Auto and explicit controls,
authenticate the selected function/tuple and replay dirty448→64→448 inputs
with guard bytes. M64 Auto remains64×32 with no new exact record on either
card. Every selected cell executes; none is skipped.

The new unit `NativeQ8MeasuredM448SelectionAndActualRowIsolation` verifies
the four measured selections. Missing-row checks use authored M448-only
spans, so future measured records are not forbidden by a frozen inventory
size. The existing `V2_Integration_ROCmNativeQ8ArchitectureExactReplay`
wildcard registers it explicitly in production preflight.

CTest's default1024-byte successful-output truncation hid the native tuple
receipts. The complete XTX log was preserved; the two MI50 row-tile owners
were rerun in3.938s with an enlarged report-output limit. This changes only
the private collection wrapper. The receipts authenticate PCI0000:8c:00.0
gfx906 and0000:49:00.0 gfx1100 against actual participant metadata.

## Real model and lifecycle

Same [continuation checkpoint](QFN_ACTIVE_PROGRESS_MIGRATION_20261110.md):
two local MI50s plus two actual CPU socket MPI ranks,28 physical workers per
socket, Qwen3.8-Flash-Next UD-Q4_K_XL plus Q8_0 MTP sidecar,262144 context,
512/256 tokens, seed42, one warmup and three measurements. Dynamic MTP and
expert balancing remain enabled; RAM ngrams, GPU row caches and16/32GiB
RAM/disk prefix budgets are unchanged. The original application performs
automatic MPI bootstrap and admission.

The run completes56 measured promotions in14waves with471,552,000 physical
movement bytes. Dynamic movement and cold calibration vary between lifetimes;
do not attribute their timing to the table alone. Peak per-process RSS is
36.010/77.155GiB and selected per-client VRAM30.308/30.076GiB. Resource observers
report no errors. Both actual MPI compute processes map GNU
`/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0`, with no LLVM OpenMP mapping and
no preload. These observations do not certify long-running leak freedom.

Native, driver and process-tree retirement pass in250.730s. After retirement,
727,912,514 bytes of prefix files are reclaimed using28,690 metadata bytes
and zero payload reads. Full preflight, broader topology model panels and
OpenCode remain deferred; this slice does not claim those certificates.

## Evidence and next work

Original immutable producer/profile corpora:
`/tmp/qfn-rocm-native-q8-exact-sdk101-r2/{gfx906,gfx1100}/corpus`.
The private additive-emitter packet and retention receipt are under
`/tmp/qfn-moe-active-progress-continuation-next/next-prefill-review/native-q8-m448/`.
Current model/config/output comparison, resource/provider records and prefix
retirement are under `/tmp/qfn-rocm-native-q8-m448-only-next/`.
Focused build/CTest records are
`/tmp/qfn-cpu-eight-row-rocm101-root/native-q8-m448-*`.
Corpus/result payloads are not committed to this source repository.

Next measure reuse of the existing exact swap scorer for isolated two-edge
tier exchanges. That optimization is separate from this dispatch change and
must preserve the complete final coherent objective and original admission.
