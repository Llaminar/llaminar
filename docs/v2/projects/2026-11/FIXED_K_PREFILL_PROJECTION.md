# Fixed-K Explicit-Operand Prefill Projections (issue #74)

Status: in progress (draft PR). CUDA half-RNE path complete and verified on
GB10; strict-FP32 mode and ROCm (gfx906 scalar, gfx1100 WMMA) in progress.

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

## Remaining

- strict-FP32 mode: SGEMM-class scalar kernel at least as economical as
  pedantic cuBLAS for wide QKV/Z at M >= 256 (ncu evidence).
- ROCm: gfx906 scalar half/FP32 (shared body), gfx1100 WMMA half
  (runtime-unqualified; spill-guard compiled), hipBLAS route retired.
- ncu evidence for every new kernel; ROCm qualification on josh.
