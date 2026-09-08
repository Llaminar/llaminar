# CUDA Qwen3.8 tiny-projection tuning — September 6, 2026

## Scope and result

Continuation of the single-3090, Qwen3.8-27B IQ4_XS, fixed-MTP3 Release tuning
goal. The preceding fused-QKV/gate-up concurrency work is checkpointed in
`5635e0538`. This slice changes only the CUDA small-N floating projection
implementation. It does not change GGUF weights, FP32 activations, KV precision,
MTP depth, graph ownership, collectives, or admitted memory.

The first production measurement improves prefill from **930.700 to 957.651
tokens/s (+2.90%)**. Decode is **63.701 tokens/s**, versus 63.635 before this
slice: effectively unchanged, not a demonstrated decode gain. Both receipts
have identical 256 output token IDs, 63.203% MTP acceptance, 330 verifier
transactions over three measured iterations, and zero transaction failures.
This is progress, **not completion of the external llama.cpp performance goal**.

Receipts (local, intentionally not committed):

- Baseline: `/tmp/qwen38-cuda1-mtp3-concurrent-fused-final.json`.
- Candidate: `/tmp/qwen38-cuda1-mtp3-tiny-projection.json` and sibling `.log`.
- Isolated unprofiled sweep: `/tmp/qwen38-tiny-final-test.log`.

The production command uses one warmup and three measured iterations:

```bash
env LLAMINAR_BENCHMARK_ITERATIONS=3 LLAMINAR_BENCHMARK_WARMUP_ITERATIONS=1 \
./build_v2_release/llaminar2 benchmark \
  -m /mnt/llaminar-production-parity/cache/models/Qwen3.8-27B-IQ4_XS.gguf \
  -d cuda:0 -c 4096 --prompt-file /tmp/qwen38-exact512-prompt-nonl.txt \
  -n 256 --mtp --mtp-draft-tokens 3 --mtp-graph-capacity-draft-tokens 3 \
  --mtp-depth-policy fixed --deterministic \
  --benchmark-json-output /tmp/qwen38-cuda1-mtp3-tiny-projection.json
```

Exactly 512 prompt tokens are evaluated. PerfStats and profilers are disabled
for that timing sample. The receipt identifies graph execution and retained
prefill replay. Prepared weights remain 17,380,802,560 bytes; reusable workspace
remains 1,590,397,956 bytes. The extra VRAM allowance was not needed here.

## Attribution and installed schedules

The saved Nsight Systems node trace
`/tmp/nsys-qwen38-cuda1-mtp3-rows4-final-node.sqlite` exposes **prefill only**;
it must not be used to attribute conditional-parent decode kernels. In that
trace, 96 small FP32 batched projection launches consume 75.346 ms across two
prompt runs: approximately 37.7 ms per prompt for GDN alpha/beta projections.
The model geometry is two projections, N=48, K=5120, M=512.

The old schedule assigns 256 threads to each dot product and uses nine block
barriers for the fixed reduction. The installed schedules preserve every
stride-256 partial sum and every addition's parenthesization:

- Short batches retain a block per output, but one warp evaluates the original
  cross-warp reduction after a single shared-memory publication barrier. The
  last five levels use warp shuffles. This is not a new reduction order.
- M>=32 assigns one warp per output. Each lane keeps eight independent partials
  corresponding to the old threads L, L+32, ..., L+224. Register reductions and
  shuffles reproduce the original tree without shared memory or barriers.
- FP16 uses native exact finite-weight conversion. FP32 and BF16 use their
  existing native encodings; all accumulation remains FP32. The unchanged
  fused SwiGLU/down kernel is outside this optimization.

Trials with two or four rows per warp were rejected. Register pressure and
reduced occupancy outweighed their weight reuse. Enforcing a four-block
occupancy target also slowed FP32. No losing candidate or runtime opt-out was
installed.

## Correctness and profiling

`V2_Integration_CUDATinyProjectionAllFloatingFormats` is added to
`ProductionParityPreflight`. It invokes the production launch bridges inside
retained graphs and compares bytes with an independent copy of the historical
arithmetic, including the old software FP16 conversion. Coverage totals 1,076
format/geometry cases, with twenty graph replays per case: M=1..65, neighboring
larger bucket sizes, M=4096, N/K tails, two distinct weight batches, and every
finite FP16/BF16 encoding. The Release sweep passes in 3.20 seconds.

The separate `V2_Perf_CUDATinyProjection` registration is **not** in preflight.
Its parameterized names allow one exact dtype/row count per profiler process.
Canonical unprofiled M=512 latencies versus the old-tree oracle are:

| Native weight format | Candidate | Historical oracle |
| --- | ---: | ---: |
| FP32 | 427.331 us | 719.954 us |
| FP16 | 383.473 us | 881.618 us |
| BF16 | 393.994 us | 523.131 us |

The oracle timings are an isolated reference, not a substitute for production
before/after timing. In particular its FP16 path deliberately retains the old
software conversion. M=16/31 FP32 block timings are slightly slower than this
independently compiled oracle; do not claim a universal per-shape speedup.

Six independent Nsight Compute launches profile M=4 and M=512 in each format.
Reports use `/tmp/qwen38-tiny-final-{FP32,FP16,BF16}-M{4,512}.ncu-rep`, with
`-details.log` exports. Do not use profiler-perturbed timing prints as economy
evidence. Every candidate reports zero local-memory loads/stores and zero spill
requests; compiler inspection also reports zero stack/local storage.

| Schedule / format | Registers/thread | Achieved occupancy |
| --- | ---: | ---: |
| M=4 / FP32 | 64 | 58.16% |
| M=4 / FP16 | 56 | 57.12% |
| M=4 / BF16 | 56 | 57.19% |
| M=512 / FP32 | 77 | 47.45% |
| M=512 / FP16 | 80 | 49.31% |
| M=512 / BF16 | 78 | 49.28% |

All use 256 threads/block; M=512 launches 6,144 blocks. FP32 prefill reaches
87.99% L2 throughput with 97.68% L2 hit rate. Its 50% theoretical occupancy is
register-limited, but the higher-occupancy trial was slower.

Full Unit is **633/633 passing**, 71.51 seconds:
`/tmp/qwen38-tiny-unit-gate.log`. The expanded preflight is **94/94 passing**,
236.16 seconds: `/tmp/qwen38-tiny-preflight-gate.log`. The new floating
regression takes 3.96 seconds through its registered launcher. No performance
test entered the functional gate.

The final-source confirmation is
`/tmp/qwen38-cuda1-mtp3-tiny-projection-final.json`: **948.547 tok/s prefill,
63.507 tok/s decode**, again with all 256 token IDs identical and no workspace
increase. Thus two independent three-iteration trials give 948.5–957.7 prefill
(1.9–2.9% above the saved baseline), not an invariant 957.7 result. Decode remains
effectively unchanged. Both production timing runs have PerfStats disabled.

## Refreshed external comparison

The pinned upstream checkout is unchanged at
`73a43d1f69345aee8bb186ef4b3172cef892f2e5`. A fresh unprofiled Release
`llama-server` run uses the same GGUF, GPU0, context 4096, batch/ubatch 512,
FP16 KV, flash attention and `draft-mtp` maximum depth three. It binds only
`127.0.0.1:19082`. One warmup precedes three measured requests, each containing
the exact prompt bytes, 256 requested tokens, temperature zero, no prompt-cache
reuse, and no EOS truncation. Every receipt confirms 512 evaluated tokens,
256 generated tokens and `cache_n=0`.

Receipts: `/tmp/qwen38-tiny-llama-baseline-{0,1,2,3}.json` (zero is warmup),
with `/tmp/qwen38-tiny-llama-baseline-server.log`. The mean is **981.901 tok/s
prefill and 70.600 tok/s decode**. The server was terminated cleanly after
collecting the receipts. Its command is:

```bash
env CUDA_VISIBLE_DEVICES=0 \
/tmp/llama.cpp-mainline-73a43d1/build-cuda/bin/llama-server \
  -m /mnt/llaminar-production-parity/cache/models/Qwen3.8-27B-IQ4_XS.gguf \
  --host 127.0.0.1 --port 19082 -ngl 999 -c 4096 -np 1 -b 512 -ub 512 \
  -fa on -ctk f16 -ctv f16 --spec-type draft-mtp --spec-draft-n-max 3 --no-ui
```

This comparison exposes an attribution caveat, not a newly established
correctness defect: the generated outputs share their first 77 tokens and then
diverge. Llama.cpp accepts 157 drafted tokens per request; Llaminar accepts 146
and runs 110 verifier transactions. Therefore the two engines do not perform
the same number of full-model verifier evaluations, even at the same depth and
output length. Their published acceptance percentages also use different
denominators; compare raw counts, not those percentages. The different
completion trajectory could explain part of the acceptance difference. Do not
attribute the whole roughly 10% decode gap to slower kernels, nor declare an
MTP correctness bug from this comparison alone.

## Common-output-prefix diagnostic

A second pair of unprofiled runs limits generation to 64 tokens, before the
observed cross-engine divergence. The prompt, model, precision, context, GPU
and fixed depth remain unchanged. Both engines retain the same 64-token output
prefix as their respective 256-token receipts; those prefixes are identical.
Each measurement still has one warmup and three measured requests.

| Engine | Prefill tok/s | Decode tok/s | Verifier transactions/request | Drafted / accepted per request |
| --- | ---: | ---: | ---: | ---: |
| Llaminar | 960.224 | 64.145 | 27 | 81 / 37 |
| llama.cpp | 995.036 | 68.889 | 24 | 72 / 38 |

Receipts are `/tmp/qwen38-cuda1-mtp3-common-prefix64.{json,log}` and
`/tmp/qwen38-llama-prefix64-{0,1,2,3}.{json,metrics}`. The external server's
`spec_decode_num_drafts_total` is 96 after four requests, establishing 24
verifier transactions per request independently of its draft-token count.
Both servers are stopped before another timing or device gate starts.

This bounds a real work-count difference even on the same output trajectory.
It does **not** yet isolate a predictor defect or prove that our verifier kernel
is slower. Whole-request time divided by verifier count is not a kernel timing:
initial tokens, final response clipping, serial tails and shifted-cache work
have different accounting boundaries. The published acceptance-rate fields
also use different denominators; raw drafted/accepted/transaction counts are
the comparable evidence.

Cross-engine token divergence alone does not establish a batch-invariance
failure in either engine. That requires a within-engine serial/grouped test on
the same inputs and state. The next correctness check is the canonical Qwen3.8
CUDA matrix with its CPU/FP32 Hugging Face checkpoints, exact serial-token
witnesses, prefix restore and graph-path evidence. Its short reference prompt
must not be presented as a complete proof of the 512-token benchmark trajectory.

## Next

### MTP correctness follow-up

The canonical Qwen3.8 CUDA production campaign now passes all six generated
cells: off, fixed 1/2/3/15, and dynamic depth. The driver validates all 53 CSV
artifacts. Fresh prerequisites pass Unit 633/633 and preflight 94/94. Total wall
time is 464.928 seconds, of which the six-cell campaign uses 154.007 seconds
including first-time CPU reference generation and additive depth-15 sidecars.
The GGUF is a persistent tmpfs cache hit; no model copying occurs.

Report: `/tmp/qwen38-cuda-mtp-math-proof.json`; evidence root:
`/tmp/production-campaign-artifacts/20260906T213827Z-718392-1788730707268303977`.
Each cell includes captured-path and fresh/full/partial prefix-restore proof.
The five enabled cells retain their exact declared execution depth, publish
serial-exact output witnesses and accept real drafts. The separate dynamic
witness accepts fifteen drafts and proves a serial-exact adaptive window.

| Numerical checkpoint | Cosine against CPU/FP32 HF |
| --- | ---: |
| Main prefill LM head | 0.999967 |
| Main decode LM head, steps 0/1/2 | 0.999963 / 0.999951 / 0.999928 |
| MTP3 first / terminal recursive LM head | 0.999910 / 0.999922 |
| MTP15 terminal recursive LM head | 0.999725 |

The exact benchmark prompt also has a fresh Release non-MTP receipt,
`/tmp/qwen38-cuda1-no-mtp-serial-proof.{json,log}`. All 256 token IDs match
the MTP3 final-source receipt, and every measured repetition within each run
is identical. Non-MTP counters confirm zero drafts/verifiers, with captured
generation and prefill replay. Serial decode measures 42.587 tok/s versus
63.507 for MTP3: a 49.12% gain without token drift on this trajectory.

These are complementary proofs, not interchangeable ones: the canonical short
prompt supplies HF stage comparisons; the 512-token prompt supplies a longer
within-engine token-equivalence check. Neither token equality nor a cosine
score claims byte-identical hidden/logit tensors for every benchmark row.
No mathematical defect was exposed and no arithmetic, tolerance, sampling,
format, depth or precision was changed to chase upstream acceptance.

### Non-speculative external baseline

A fresh upstream run changes only `--spec-type draft-mtp` to `--spec-type
none` (and omits the irrelevant draft-depth setting), retaining the same model,
device, prompt, FP16 KV, context, batch/ubatch and request sampling fields.
One warmup precedes three measured 512-prompt / 256-output requests. Every
response reports `cache_n=0`; outputs are identical across repetitions.
Receipts are `/tmp/qwen38-llama-no-mtp-{0,1,2,3}.{json,metrics}` and
`/tmp/qwen38-llama-no-mtp-server.log`. No profiler or concurrent device workload
is active. The server is terminated after the measurements and tokenization.

| MTP off | Llaminar | llama.cpp | Llaminar relative difference |
| --- | ---: | ---: | ---: |
| Prefill tok/s | 950.651 | 1175.520 | -19.13% |
| Decode after first token, tok/s | 42.421 | 45.664 | -7.10% |

Use the common 255-token after-prefill numerator for decode here. The native
Llaminar headline counter includes all 256 output IDs and reports 42.587; the
upstream formula excludes the prompt-produced first token even though its
`predicted_n` field is 256. The complete decode windows average 6011.208 ms
and 5584.221 ms respectively. This is a real non-speculative execution gap,
not an acceptance-rate effect; the larger prefill deficit deserves attention.

The saved upstream serial/MTP tokenizations diverge at zero-based index 103
(serial token 10877, MTP token 318). This reproduces within-engine MTP-on/off
token non-equivalence on this workload, but does not by itself identify a
particular grouped kernel as its cause. Llaminar serial/MTP remains exact over
all 256 IDs. Both upstream modes first differ from Llaminar at index 77.

### Captured verifier resource evidence

`/tmp/qwen38-production-mtp3-m4-iq4.ncu-rep` profiles one real captured
four-row IQ4_NL execution-codebook projection from the IQ4_XS GGUF during the
benchmark warmup. NCU application replay runs six independent passes; none of
their timing receipts are canonical performance samples. The exact symbol is
`nativeVnniGemv_kpar_small_m<4,128,1,4,false>`, grid `(136,6,1)`, block 128.
It reports 37 registers/thread, zero local-memory spill requests, 81.34%
achieved occupancy, 74.49% DRAM throughput (678.40 GB/s), and 48.03% compute
throughput. This measured node is not suffering from low occupancy or spills;
it does not establish the complete graph's bottleneck. The first unsuccessful
selector used C++ template spelling without NCU's `(int)` annotations and
produced only a kernel inventory; it is not a profile certificate.

Keep the external baseline tied to an exact unprofiled 512-prompt /
256-decode receipt: older 425-token llama.cpp samples and 64-token profiled
samples are not comparable. The remaining decode gap needs captured
conditional-body attribution and acceptance/transaction analysis, particularly
over the common output prefix. Increasing draft depth or changing precision is
not the next lever. This slice's Unit, preflight, isolated arithmetic, profiling
and final-source production gates are complete; the larger throughput goal is
still open.
