# Fixed-K Explicit-Operand Prefill Projections (issue #74)

Status: in progress (PR #75). CUDA half-RNE and strict-FP32 paths complete
and verified on GB10. ROCm (gfx906 scalar, gfx1100 WMMA) compiled spill-free
and preflight-green on gfx906; gfx1100 has no hardware and is reported
runtime-unqualified. gfx906 wide-group economy is being tuned natively.

## Problem

With prefix caching on (default) a cold prompt is prefilled in different
chunks than with it off, because recurrent checkpoints split it (778 tokens:
512+256+10 versus 512+266). The default `FP16MoEAlphaBeta` GDN policy computed
ordinary multi-row alpha/beta prefill with `cublasGemmEx` (half A/B, FP32
accumulate). cuBLAS selects kernels and split-K by M, so prompt rows 768..777
produced different alpha/beta bits at physical M=64 and M=384 and the greedy
token at completion index 7 flipped. hipBLAS has the same structure on ROCm;
explicit full-FP16 (wide QKV/Z) and strict FP32 GDN modes too.

Kernel-level reproduction on GB10 (K=2048, N=32): cuBLAS rows differ from its
own M=512 rows for M in {1, 2, 10, 16, 33, 48, 64, 128}.

## Contract

`src/v2/kernels/common/FixedKProjectionContract.h`. K splits into eight
partitions fixed by K alone; each partition is one ascending chain from +0
(m16n8k16 MMA units over 64-element blocks for half, IEEE FMA per element for
scalar); partitions fold left to right. The result for output (m, n) depends
only on input row m, weight row n and K. Serial decode and MTP verifier rows
keep their fixed 256-lane tree. The scalar contract is reproducible on the
host bit for bit (checked by the tuning harness).

## CUDA half schedules

`src/v2/kernels/cuda/gemm/CUDAFixedKHalfProjection.cuh`, policy in
`CUDAFixedKProjectionLaunch.h`:

- partitioned (narrow, at most sixteen 64x64 tiles): one CTA per
  (row tile, compact column tile, partition) publishes chains to a bounded
  named bank (`gemm_fixed_k_partials`, at most 2 MiB), then a fold kernel;
- tiled: cp.async stages of one 64-element block, register partition flush,
  banded rasterization (16 row tiles) for L2 reuse at large M.

All schedules byte-identical: 0 differing rows over 1,900 (schedule, M,
offset) cases, M in 1..1024, offsets 0..511, ragged widths, K tails.

### Economy (GB10 sm_121, graph replay, weights rotated through 160 MiB so they come from DRAM; µs, min of 3)

| group (K) | M=16 | 64 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|
| alpha/beta 32+32 (2048), cuBLAS | 24.1 | 8.7 | 9.3 | 10.5 | 13.0 | 24.6 | 54.3 |
| fixed-K | 5.3 | 5.5 | 5.6 | 6.6 | 11.0 | 16.4 | 32.9 |
| QFN alpha/beta 48+48 (2560), cuBLAS | 29.9 | 10.3 | 12.0 | 11.8 | 17.9 | 26.9 | 89.4 |
| fixed-K | 6.0 | 6.1 | 7.2 | 11.8 | 14.0 | 26.1 | 71.1 |
| full-FP16 8192+4096+32+32 (2048), cuBLAS | 256.8 | 290.5 | 331.0 | 473.3 | 915.8 | 1819 | 3832 |
| fixed-K | 237.2 | 249.8 | 301.0 | 441.3 | 877.6 | 1757 | 3720 |
| QFN full-FP16 10240+6144+48+48 (2560), cuBLAS | 397.4 | 458.2 | 523.1 | 791.0 | 1547 | 2875 | 5933 |
| fixed-K | 380.0 | 399.0 | 480.4 | 692.2 | 1377 | 2769 | 5716 |

Known cells above parity (deterministic arithmetic retained, documented):
memory-bound M=16 for 5120+3072 K=2560 (+2%), 4096-only M=448 (+3-4%),
1536/3072-column groups at M=384 (up to +6%; not production groups).

## Strict FP32 (explicit `fp32` GDN mode)

Scalar contract: eight partitions of K elements, one IEEE FMA chain each,
left fold; bit-identical to a host `std::fma` evaluation (checked in the
preflight regression). Narrow groups use the shared CUDA/HIP partitioned body
(`kernels/common/FixedKScalarProjectionBody.cuh`); wide groups use the
SGEMM-class kernel (`kernels/cuda/gemm/CUDAFixedKScalarProjection.cuh`):
128x128 tiles, 8x8 outputs per thread, three cp.async stages, register
partition flush, plus a wave-balanced tail (the final partial wave is cut into
2 or 4 row slices when every slice fits one round). cuBLAS pedantic
`GemmEx`/`GemmBatchedEx` explicit-operand paths are retired; serial
explicit-FP32 decode now uses the verifier's fixed tree like half.

GB10 (.79), cold weights, µs, policy vs pedantic cuBLAS:

| group (K=2048) | M=256 | 384 | 448 | 512 | 1024 |
|---|---|---|---|---|---|
| TP2 4096+2048+16+16, cuBLAS | 590.0 | 919.4 | 1070.1 | 937.8 | 1884.9 |
| fixed-K | 510.2 | 724.6 | 932.2 | 946.2 | 1823.5 |
| full 8192+4096+32+32, cuBLAS | 1255.9 | 1750.8 | 2095.6 | 2019.1 | 4030.0 |
| fixed-K | 1004.6 | 1472.1 | 1934.2 | 1945.3 | 3803.9 |

QFN TP2 5120+3072+24+24 (K=2560) is 0.76-0.85x throughout; small M (16-128)
0.65-0.97x. Above parity: TP2 M=512 +1% (noise level) and the Z-only 4096
group at M=512 (+9%, not a production group: Z always runs fused with QKV).

## Kernel resource evidence (ncu, GB10, base clocks, one kernel per launch)

No local-memory loads or stores in any kernel (`local_ld`/`local_st` = 0);
ptxas reports zero spills for every instantiation.

| kernel (case) | grid x block | regs | smem | occ theo/ach % | SM % | mem % |
|---|---|---|---|---|---|---|
| half partitioned 64x64 (32+32, M=512) | 8,1,8 x 128 | 123 | 49 KB | 16.7/11.2 | 6.3 | 13.3 |
| half fold (32+32, M=512) | 128 x 256 | 22 | 0 | 100/32.2 | 1.8 | 4.5 |
| half partitioned 64x64 (48+48, M=512) | 8,2,8 x 128 | 123 | 49 KB | 16.7/15.8 | 12.7 | 21.8 |
| half tiled 64x32 w4 (48+48, M=1024) | 48 x 128 | 111 | 49 KB | 16.7/8.3 | 11.4 | 29.1 |
| half tiled 32x64 w4 (12352 cols, M=32) | 193 x 128 | 118 | 49 KB | 16.7/16.6 | 6.1 | 15.5 |
| half tiled 64x32 w4 (12352 cols, M=128) | 772 x 128 | 111 | 49 KB | 16.7/16.4 | 18.6 | 47.5 |
| half tiled 64x64 (12352 cols, M=256) | 772 x 128 | 153 | 49 KB | 16.7/16.5 | 32.0 | 54.6 |
| half tiled 128x64 w8 (3072 cols, M=384) | 144 x 256 | 147 | 74 KB | 16.7/16.7 | 29.9 | 38.5 |
| half tiled 128x64 (12352 cols, M=512) | 772 x 128 | 233 | 49 KB | 16.7/16.4 | 40.8 | 52.5 |
| half tiled 64x128 w8 (12352 cols, M=512) | 776 x 256 | 147 | 74 KB | 16.7/16.6 | 35.8 | 46.1 |
| half tiled 128x128 w8 (12352 cols, M=512) | 388 x 256 | 217 | 66 KB | 16.7/16.7 | 44.9 | 39.2 |
| FP32 partitioned 64x32 (32+32, M=128) | 2,2,8 x 256 | 64 | 0 | 66.7/16.7 | 11.3 | 16.8 |
| FP32 partitioned 64x64 (32+32, M=512) | 8,1,8 x 256 | 62 | 0 | 66.7/22.3 | 18.1 | 26.5 |
| FP32 sgemm 64x64 (12352 cols, M=64) | 193 x 256 | 122 | 41 KB | 33.3/32.6 | 22.2 | 28.1 |
| FP32 sgemm 128x128 (6176 cols, M=512) | 192 x 256 | 254 | 61 KB | 16.7/16.7 | 54.4 | 37.7 |
| FP32 sgemm 32x128 tail slice (6176 cols, M=512) | 16 x 256 | 124 | 38 KB | 33.3/16.7 | 7.5 | 12.8 |

Low theoretical occupancy is deliberate: multi-stage shared pipelines and
large register tiles hide latency per warp, and the measured end-to-end kernel
times above are at or below the vendor BLAS they replace. Narrow partitioned
launches are latency bound by construction (few output tiles); their SM/memory
utilization is low but their wall time is 0.2-0.9x of cuBLAS.

## End-to-end (GB10, Qwen3.6-35B-A3B UD-IQ3_S, 778-token prompt, 160 greedy tokens)

develop + local admission cherry-picks (037ccc892, 9349a8210; not part of the PR):

| topology | prefix ON (512+256+10) vs OFF (512+266) |
|---|---|
| CUDA only | identical 160/160 (develop: first diff at index 7) |
| static CUDA+CPU split | identical, and equal to CUDA only |
| dynamic CUDA+CPU split | identical, and equal to CUDA only |

PerfStats show `cuda_fixed_k_projection_calls` (`partitioned_64x64`) for every
prefill bucket and no `cublas_gemm_ex` route.

Prefix fingerprint v19 (tiered storage, one disk directory): a v18 binary
writes the archive and its own restart restores it from disk; a v19 binary on
the same archive misses precisely (hit=false, 0 matched tokens, no warnings),
its second request hits, and a v19 restart restores from disk. All v19
completions (cold, RAM, disk) equal prefix-off; v18 differs at index 7.

## ROCm

`kernels/rocm/gemm/ROCmFixedKProjection.hip`: gfx1100 half uses
`v_wmma_f32_16x16x16_f16` (wave64) over 64-element blocks, four units in
natural K order; other targets (gfx906) and strict FP32 use the shared scalar
body. Bytes differ between architectures (different hardware arithmetic) but
each is chunk-, bucket- and offset-invariant. hipBLAS half `GemmEx` is
retired; strict FP32 wide groups no longer use hipBLAS.

### gfx906 economy (graph replay, rotated weights, µs; ratio = selected / retired hipBLAS)

Measured at a2f4b3e8b (`tune_rocm`, single MI50-class gfx906). Byte
invariance across every schedule, M in {1..512} and offsets {0,7,256}:
0 differing rows over 72 cases.

| Group (K) | M=16 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| half 32+32 (2048) | 0.29 | 0.30 | 0.28 | 0.43 | 0.63 |
| half 12+12 (2560) | 0.31 | 0.44 | 0.30 | 0.87 | 0.64 |
| half 48+48 (2560) | 0.33 | 0.29 | 0.50 | 0.75 | 1.20* |
| fp32 32+32 (2048) | 0.14 | 0.10 | 0.11 | 0.22 | 0.60 |
| half 2560+1536+12+12 (2560) | 1.29 | 1.22 | 1.20 | 1.43* | 1.51* |
| fp32 2560+1536+12+12 (2560) | 0.74 | 1.01 | 1.18 | 1.13* | 1.42* |

`*` = the measured time of the schedule the fixed selection now picks, to
be re-measured with the slabbed launch (partitioned 217 µs instead of tiled 128x64
531 µs for 48+48 at M=1024; tiled 64x64 instead of 128x64 for wide groups,
which was 2.0-2.9x hipBLAS). Policy: every scalar group of at most 128
columns partitions; groups taller than one bounded slab
(`fixed_k_projection::partitionedSlabRows`, ≥512 rows) run as consecutive
row slabs reusing the unchanged admitted bank. Wide groups use tiled 64x64;
scalar tiled 128x64 was never faster and is retired.

## Remaining

- gfx906 wide full-FP16/FP32 groups at M ≥ 128 are 1.2-1.5x hipBLAS; native
  tuning on the ROCm host (larger register tiles, conflict-free LDS reads,
  `v_dot2_f32_f16` pair chains for half) is in progress.
