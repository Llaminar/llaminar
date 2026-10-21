# MI50 FP16 and Q16 causal QK benchmark

`v2_perf_mi50_attention_qk_precision` compares native gfx906 FP16
`v_dot2_f32_f16` with signed INT16 `v_dot2_i32_i16`. It is an attention **QK
scoring experiment**. It does not install a production Q16 ROCm attention mode.
Softmax, multiplication by V, model execution and token throughput are outside
its measurement boundary.

Both candidates use the same inputs, 16-query by 64-key LDS tiles, 256 threads,
four accumulator chains, GQA head mapping and causal mask. LDS rows use the
same odd dword stride, preventing the different native storage strides from
biasing shared-memory bank conflicts. Each complete score
is reduced by one thread. Q16 keys use the canonical `Q16_1Block_64` or
`Q16_1Block_128` structure and its full signed range; query codes use
`cpu::q16AttentionQueryLimit(head_dim)` to prove that the complete INT32 dot
cannot overflow. Q16 scales are applied before FP32 score publication. FP16
uses FP32 accumulation.

Unlike fused FlashAttention, this experiment publishes a dense FP32 score
matrix, including masked entries. That common output traffic and LDS behavior
can limit both modes. A scoring speedup here warrants a subsequent fused
attention experiment; it is not evidence of a full prefill speedup.

## Build and run

Configure a Release Ninja build using the repository's canonical ROCm SDK and
runtime dependencies. The target has an intentional gfx906 architecture and
uses the same compiler spill guard as other HIP targets.

```bash
cmake --build build_v2_rocm10_release --target v2_perf_mi50_attention_qk_precision --parallel
ctest --test-dir build_v2_rocm10_release --output-on-failure --no-tests=error \
  -R '^V2_Perf_MI50AttentionQKPrecision$'

./build_v2_rocm10_release/tests/v2/v2_perf_mi50_attention_qk_precision \
  --device=0 > /tmp/mi50-qk.csv

./build_v2_rocm10_release/tests/v2/v2_perf_mi50_attention_qk_precision \
  --queries=512 --keys=512 --heads=14 --kv-heads=2 --head-dim=64 \
  --samples=51 --warmups=8 --batch=8 > /tmp/mi50-qk-512.csv
```

Without a shape selector, the binary sweeps 12 geometries: head dimensions 64
and 128, query counts 17/128/425/512/2048, GQA ratios seven and eight, partial
tiles, and 128 new queries attending an 8192-key history. Shape selectors use
the remaining defaults declared by `Options` in the source. `--help` lists the
available selectors. Queries occupy the last positions in
the live history, so `keys - queries` is the existing prefix.

The binary requires a gfx906 GPU and fails on an unavailable device, invalid
geometry, failed native operation or numerical mismatch. It does not skip a
failed candidate or replace its requested arithmetic.

## Timing and accuracy

CSV contains three independently captured scopes for each precision:

| Scope | Complete measured transaction |
| --- | --- |
| `prepared_qk` | QK scoring from persistent prepared operands |
| `prepare` | FP32 conversion of all queries and only newly appended keys |
| `prepare_qk` | That conversion followed by QK scoring |

Preparation is a GPU operation in both modes. Prefix keys are already prepared
and retained. Storage, copies, capture and numerical observations are outside
timing. Native timing events measure retained graph replays with eight embedded
transactions by default, four warmups and 25 samples. Paired samples alternate
FP16/Q16 order. Medians and p10/p90 durations are microseconds per transaction;
`useful_qk_tops` counts two operations per dimension for each causally visible
score. Preparation-only throughput is zero because it performs no useful QK
operations. Do not add independently measured scopes: `prepare_qk` directly
measures their combined cost.

For the same shape and scope, `FP16 median_us / Q16 median_us` is Q16 speedup;
values above one favor Q16. Compare `prepare_qk` as well as `prepared_qk` to
include conversion economics.

Outside timing, the harness validates all prepared rows against independent
CPU conversion, including scales, code bounds and `sum_qs`. It checks leading
and trailing output guards and representative scores at causal, tile, GQA and
partial-row boundaries against FP64 equations. Integer reference reductions
use INT64 and independently check the INT32 bound. Original-input relative L2
and maximum absolute score errors are reported separately from the error
against the decoded operand equation. These are sampled score errors, not
model-quality or softmax-output certification. Inputs include constant extreme
code rows, signed rows, ordinary random values and a large within-row outlier.

## Isolated profiler evidence

`--profile` requires one explicit precision and launches one captured
transaction. Prepared operands are initialized on the host before upload, so a
`prepared_qk` graph contains exactly one scoring dispatch and no preparation
or numerical-oracle kernels. Use the selected SDK's profiler and keep profiler
output outside the repository.

```bash
"${ROCM_PATH}/bin/rocprofv3" --kernel-trace --hip-runtime-trace --hip-graph-trace \
  --stats --output-format json csv --kernel-include-regex 'mi50_qk_score' \
  --output-directory /tmp/mi50-qk-fp16-profile -- \
  ./build_v2_rocm10_release/tests/v2/v2_perf_mi50_attention_qk_precision \
  --mode=fp16 --profile=prepared_qk \
  --queries=512 --keys=512 --heads=14 --kv-heads=2 --head-dim=64
```

Run Q16 and each additional shape in separate invocations. `prepare` graphs
contain one conversion dispatch; `prepare_qk` contains its two ordered nodes.
Join dispatches to the JSON graph record; setup can include native copy/memset
kernels outside that graph. For conversion counters, select
`--kernel-include-regex 'mi50_qk_prepare'` instead.
Authenticate native instruction use and per-specialization VGPR, SGPR, LDS and
zero scratch/spills from the compiled code object. Gather hardware counters
in separate supported groups using the ROCm tuning workflow. Profiler durations
are diagnostic evidence; use the unprofiled CSV for comparative throughput.
