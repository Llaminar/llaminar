# IQ Format GEMV Deep Dive

## Question

Why do most IQ formats underperform badly in the CUDA native-payload GEMV path, while the XXS variants are materially better, and what kernel structure should replace the current generic path?

## Matrix Summary

Using the full GEMV live-dispatch matrix from `/tmp/llaminar_gemv_matrix_full.csv`:

| Format | Avg % Peak HBM | LM Head Avg % Peak | Avg TP Eff |
|--------|-----------------|--------------------|------------|
| IQ1_M | 1.23 | 1.50 | 0.885 |
| IQ1_S | 1.11 | 1.35 | 0.887 |
| IQ2_XS | 4.17 | 5.83 | 0.801 |
| IQ4_NL | 8.21 | 11.00 | 0.795 |
| IQ3_S | 9.73 | 14.40 | 0.752 |
| IQ2_S | 11.93 | 20.27 | 0.702 |
| IQ2_XXS | 22.62 | 51.67 | 0.599 |
| IQ3_XXS | 29.70 | 66.70 | 0.593 |

Reference non-IQ baselines:

| Format | Avg % Peak HBM | LM Head Avg % Peak |
|--------|-----------------|--------------------|
| Q4_0 | 39.58 | 83.52 |
| Q4_1 | 42.85 | 86.66 |
| Q5_0 | 43.92 | 86.50 |
| Q5_1 | 43.84 | 87.38 |
| Q6_K | 42.82 | 82.93 |

Key observations:

- All IQ formats are substantially behind the Q4/Q5/Q6 family.
- IQ4_NL is not uniquely bad overall, but it is uniformly bad. It stays under 15% peak on every matrix row.
- The XXS variants are the only IQ formats that escape the worst floor. They are still slower than Q4/Q5/Q6, but much less catastrophic.
- IQ4_NL has healthy LM-head TP efficiency (~0.95), so its main issue is local decode cost, not TP scaling.

## Decode Taxonomy

The CUDA native-payload path currently runs all IQ formats through the same generic decode-to-packed-groups plus `dp4a` GEMV structure in `src/v2/kernels/cuda/CUDANativePayloadGemvTuned.cu`.

The formats do not share one decode mechanism:

### 1. IQ4_NL: 16-byte nibble LUT decode

`decode_groups<4>()` in `src/v2/kernels/cuda/CUDANativePayloadDecodeCommon.cuh` performs 32 nibble lookups per 32-element payload using `d_iq4nl_values[16]`.

This is structurally very different from Q4_0, which only does nibble extraction plus centered subtraction.

This makes IQ4_NL a pure LUT format.

### 2. IQ3/IQ2 grid formats: lookup + sign application + sideband unpack

`IQ3_S`, `IQ3_XXS`, `IQ2_S`, `IQ2_XS`, and `IQ2_XXS` use grid tables via `iq3_grid_lookup()` / `iq2_grid_lookup()` and then apply sign masks with `iq_apply_signs_4()`.

The slower S/XS variants pay extra sideband work:

- `IQ3_S` reconstructs 9-bit indices from `qs + qh`, reads separate sign bytes, and unpacks per-subblock scales.
- `IQ2_S` reconstructs indices from `qs + qh`, reads separate sign bytes from the tail of `qs`, and unpacks two sub-scales per 32-element subblock.
- `IQ2_XS` packs sign and grid index together, but still performs per-group grid lookup and scale unpack.

### 3. IQ1 formats: large grid + delta add + packed scale extraction

`IQ1_S` and `IQ1_M` use the largest grid (`iq1s_grid[2048]`, 16 KB) and also add a per-group `delta` term on top of the looked-up grid values.

`IQ1_M` additionally reconstructs global and per-subblock scales from packed metadata.

## Why XXS Performs Better

The XXS variants still use lookup-based decode, but they carry less sideband baggage.

Compared to the corresponding S/XS variants:

- `IQ3_XXS` uses packed `qs` plus a compact `scales_and_signs` tail instead of separate `qh + signs + scales` arrays.
- `IQ2_XXS` uses a very compact encoding where one byte directly selects an 8-byte grid entry and a packed control word supplies sign and sub-scale information.

That reduces both metadata traffic and bitfield manipulation.

The ROCm ISA analysis in `src/v2/kernels/rocm/gemm/README.native-quantised-isa-analysis.md` quantifies the effect:

| Format | Global Loads | Decode ALU |
|--------|--------------|------------|
| IQ3_S | 23 | 204 |
| IQ3_XXS | 20 | 166 |
| IQ2_S | 16 | 180 |
| IQ2_XS | 16 | 180 |
| IQ2_XXS | 12 | 159 |

The XXS formats are still decode-heavy, but the decode burden is meaningfully lower.

## Root Cause

The current CUDA kernel family is generic over codebooks but not over decode cost model.

That is the mismatch.

For the Q-family, decode is mostly arithmetic and can live inside the generic `dp4a` path without dominating the kernel.

For IQ formats, decode is often the dominant cost:

- IQ4_NL is dominated by LUT fetch and byte packing.
- IQ3/IQ2 S/XS are dominated by table lookup plus sideband unpack plus sign application.
- IQ1 is dominated by large-grid lookup and packed scale/delta reconstruction.

The XXS variants do better only because their metadata path is simpler, not because the generic kernel is fundamentally well matched to IQ decode.

## Proposed Kernel Solution

Do not introduce one monolithic “IQ kernel family”. Introduce two new IQ families plus one optional special case:

### Family A: `IQ4_NL_LUT_GEMV`

Target: `IQ4_NL` only.

Design:

- Keep the existing wide / kpar / direct shape split.
- Replace `d_iq4nl_values` memory lookup with a register-resident or shared-memory LUT.
- On CUDA, the 16-entry LUT can be held in registers and indexed with byte-permute style extraction, or staged once per CTA into shared memory.
- Add software-pipelined payload staging so block `b+1` loads are issued before block `b` decode begins.

Expected effect:

- Removes the current nibble-LUT bottleneck.
- Preserves the healthy TP efficiency already seen for IQ4_NL.

### Family B: `IQ_GRID_GEMV`

Target: `IQ3_S`, `IQ3_XXS`, `IQ2_S`, `IQ2_XS`, `IQ2_XXS`.

Design:

- Preload the grid table into shared memory once per CTA at kernel start.
  - IQ3 grids: 2 KB
  - IQ2 grids: 4 KB
- Use a decode microkernel that reads grid entries from shared memory, not global or divergent constant memory.
- Split the inner decode microkernel into two structural variants:
  - `IQ_GRID_XXS`: compact metadata path (`IQ3_XXS`, `IQ2_XXS`)
  - `IQ_GRID_SXS`: sideband-heavy path (`IQ3_S`, `IQ2_S`, `IQ2_XS`)
- Add software-pipelined payload staging so decode ALU overlaps the next block’s payload loads.

Expected effect:

- Removes repeated high-latency grid fetches from the hot loop.
- Better matches the fact that XXS and non-XXS grid formats do not have the same metadata complexity.

### Family C: `IQ1_GRID_GEMV` (optional but likely warranted)

Target: `IQ1_S`, `IQ1_M`.

Reason:

- `IQ1` uses the largest lookup table (16 KB) and delta-add semantics not shared by the other IQ formats.
- It is by far the weakest family in the current matrix.

Design:

- Shared-memory preload of the 16 KB grid table.
- Dedicated decode path for packed scale extraction and delta handling.
- Reuse the same shape families, but keep a dedicated inner decode loop.

## Practical Recommendation

The minimal useful split for CUDA is:

1. `IQ4_NL_LUT_GEMV`
2. `IQ_GRID_GEMV`

and inside `IQ_GRID_GEMV`, keep separate microkernels for `XXS` vs `S/XS`.

That is enough to address the measured divide:

- IQ4_NL has a unique nibble-LUT problem.
- IQ3/IQ2 share a grid-lookup problem but not identical metadata overhead.
- XXS should reuse the same family but not the same decode microkernel as S/XS.

If IQ1 becomes a priority after that, promote it to its own third family.

## Recommended Order of Work

1. Implement `IQ4_NL_LUT_GEMV` first for wide LM-head shapes.
2. Implement `IQ_GRID_GEMV` with shared-memory grid preload for `IQ3_XXS` and `IQ2_XXS` first.
3. Extend `IQ_GRID_GEMV` to the S/XS formats with specialized sideband unpack.
4. Re-evaluate whether `IQ1` deserves a dedicated family or is simply too low value.

## Implementation Update

The first `IQ4_NL` step is now landed as a low-surface-area helper optimization in `src/v2/kernels/cuda/CUDANativePayloadDecodeCommon.cuh`, not yet as a separate launch family.

`decode_groups<4>()` and `decode_groups_vec<4>()` now use a packed-register lookup helper instead of indexed `d_iq4nl_values[16]` loads:

- Four 32-bit immediates hold the full 16-entry `IQ4_NL` codebook.
- `iq4nl_lookup_reg()` selects the 4-entry pack by `(idx >> 2)` and extracts the signed byte by shift/mask.
- This removes the repeated nibble-LUT load from the inner decode loop while preserving the existing tuned wide / kpar kernels.

That is a narrower change than the originally proposed `IQ4_NL_LUT_GEMV` family, but it hits the same bottleneck first and keeps the tuned dispatch surface stable while we validate the decode win.

## Validation Update

After rebuilding `v2_perf_cuda_blockwise_tensorcore_gemm` in `build_v2_release`, a focused GEMV-only rerun on the LM-head `IQ4_NL` shapes produced:

| Shape | Old Time (us) | New Time (us) | Speedup | Old BW | New BW | Old % Peak | New % Peak |
|-------|---------------|---------------|---------|--------|--------|------------|------------|
| 3B_LM_Head | 1651.648 | 796.736 | 2.07x | 106.3 GB/s | 220.5 GB/s | 11.4 | 23.6 |
| 7B_LM_Head | 2878.560 | 1370.112 | 2.10x | 106.7 GB/s | 224.2 GB/s | 11.4 | 24.0 |
| 14B_LM_Head | 4108.288 | 1947.648 | 2.11x | 106.8 GB/s | 225.2 GB/s | 11.4 | 24.1 |

Interpretation:

- The bottleneck diagnosis was correct: `IQ4_NL` was paying heavily for the LUT path itself.
- A helper-only change is enough to recover roughly `2.1x` on the target LM-head shapes.
- `IQ4_NL` is still materially behind the Q4/Q5/Q6 family, so the broader family split remains justified.

A broader `IQ4_NL`-only rerun across the standard GEMV shape set also improved materially:

- Old average `% peak` across the saved matrix: `8.21%`
- New average `% peak` across the focused rerun: `16.26%`
- Old average LM-head `% peak` across the saved matrix: `11.00%`
- New average LM-head `% peak` across the focused rerun: `15.82%`

## IQ1 Pivot Update

After the `IQ4_NL` helper win, the next family ranking from `/tmp/llaminar_gemv_matrix_full.csv` showed `IQ1` as the weakest remaining target:

| Format | Avg % Peak HBM | LM Head Avg % Peak |
|--------|-----------------|--------------------|
| IQ1_S | 1.11 | 1.35 |
| IQ1_M | 1.23 | 1.50 |

The large-shape rows were all clustered around roughly `13-15 GB/s`, which pointed more strongly to divergent constant-memory serialization than to bad dispatch-family selection. The expanded sweep still overwhelmingly chose the `kpar` family for `IQ1_S` and `IQ1_M`, so the first retained experiment was to change the lookup-table storage, not the launch heuristic.

### IQ1 Implementation Update

`src/v2/kernels/cuda/CUDANativePayloadIQTables.cu` and `src/v2/kernels/cuda/CUDANativePayloadDecodeCommon.cuh` now move `d_iq1s_grid[2048]` off `__constant__` storage and onto the same regular device-memory path already used by the XXS grids.

The `IQ1` lookup helper now uses read-only global loads:

- `d_iq1s_grid` is declared as `__device__ uint64_t[2048]`
- initialization uses `copyHostArrayToDeviceSymbol(...)` instead of `cudaMemcpyToSymbol(...)`
- `iq1_grid_lookup()` uses `__ldg(&d_iq1s_grid[idx])`

This keeps the tuned kernel structure unchanged and tests the narrower hypothesis first: random `IQ1` grid indices are a poor fit for divergent constant-memory reads.

### IQ1 Validation Update

A focused rerun of the large `3B` / `7B` / `14B` GEMV shapes produced step-function improvements:

| Format | Shape | Old Time (us) | New Time (us) | Speedup | Old % Peak | New % Peak |
|--------|-------|---------------|---------------|---------|------------|------------|
| IQ1_S | 3B_LM_Head | 4894.720 | 219.136 | 22.34x | 1.3 | 29.9 |
| IQ1_S | 7B_LM_Head | 8582.144 | 375.744 | 22.84x | 1.3 | 30.4 |
| IQ1_S | 14B_LM_Head | 11772.928 | 533.504 | 22.07x | 1.4 | 30.6 |
| IQ1_M | 3B_LM_Head | 4911.104 | 230.592 | 21.30x | 1.5 | 31.8 |
| IQ1_M | 7B_LM_Head | 8608.768 | 395.264 | 21.78x | 1.5 | 32.4 |
| IQ1_M | 14B_LM_Head | 11811.840 | 563.072 | 20.98x | 1.5 | 32.4 |

A broader `IQ1`-only rerun across all standard GEMV shapes showed:

- `IQ1_S` average `% peak`: `1.11% -> 15.73%`
- `IQ1_M` average `% peak`: `1.23% -> 17.41%`
- average speedup across all `56` shapes per format: about `12.4x`
- no measured regressions worse than `0.2` percentage points of peak across the `112` `IQ1` cases checked

Interpretation:

- The main `IQ1` bottleneck was not the shape heuristic; it was the lookup-table memory path.
- Regular device-memory lookup with read-only caching is dramatically better than divergent constant-memory access for this table.
- `IQ1` is no longer the catastrophic outlier in the matrix, and any further work should now be driven by the remaining residual gap rather than the original pre-fix floor.
- Average TP efficiency remained healthy and improved from `0.568` to `0.701`

Recommended next step after this validation:

1. Re-run a broader `IQ4_NL` matrix slice to see whether the gain generalizes beyond LM-head.
2. Move on to the `IQ_GRID_GEMV` family for `IQ3`/`IQ2`, where the remaining decode tax is now the larger problem.

## Heuristic Retune Update

The helper change exposed a second bottleneck: the generated `IQ4_NL` exact dispatch entries for the large LM-head shapes were now stale.

A focused sweep over the existing `kpar` / `wide` families showed that the current kernel families could already do much better on the high-value LM-head cases without another inner-loop rewrite:

- Non-TP LM-head shapes were able to reach roughly `40%` to `42%` peak HBM bandwidth.
- TP2 LM-head shapes were able to reach roughly `33%` to `37%` peak HBM bandwidth.
- TP4 LM-head shapes were already close enough to their local optimum that changing them was not clearly beneficial.

Based on that, the generated `IQ4_NL` exact entries in `src/v2/kernels/cuda/CUDANativePayloadGemvDispatchHeuristicGenerated.inc` were refreshed only for the large non-TP and TP2 LM-head keys:

- `75968x896`, `151936x896`
- `75968x2048`, `151936x2048`
- `76032x3584`, `152064x3584`
- `76032x5120`, `152064x5120`

After rebuilding `v2_perf_cuda_blockwise_tensorcore_gemm`, the main benchmark path now delivers:

| Shape | Old Time (us) | New Time (us) | Speedup | Old % Peak | New % Peak |
|-------|---------------|---------------|---------|------------|------------|
| 0.5B_LM_Head | 730.112 | 210.944 | 3.46x | 11.3 | 39.1 |
| 0.5B_TP2_LM_Head | 382.976 | 123.904 | 3.09x | 10.8 | 33.3 |
| 0.5B_TP4_LM_Head | 197.632 | 73.664 | 2.68x | 10.4 | 28.0 |
| 3B_LM_Head | 1651.648 | 467.968 | 3.53x | 11.4 | 40.1 |
| 3B_TP2_LM_Head | 859.136 | 272.384 | 3.15x | 10.9 | 34.4 |
| 3B_TP4_LM_Head | 439.296 | 137.216 | 3.20x | 10.7 | 34.2 |
| 7B_LM_Head | 2878.560 | 806.624 | 3.57x | 11.4 | 40.7 |
| 7B_TP2_LM_Head | 1494.016 | 465.152 | 3.21x | 11.0 | 35.3 |
| 7B_TP4_LM_Head | 758.784 | 234.496 | 3.24x | 10.8 | 35.0 |
| 14B_LM_Head | 4108.288 | 1146.880 | 3.58x | 11.4 | 40.9 |
| 14B_TP2_LM_Head | 2130.944 | 664.576 | 3.21x | 11.0 | 35.3 |
| 14B_TP4_LM_Head | 1077.248 | 329.728 | 3.27x | 10.9 | 35.5 |

Two takeaways matter here:

- The helper-level decode fix and the exact-entry retune compound cleanly. Relative to the original saved matrix, the LM-head family is now roughly `3.1x` to `3.6x` faster end-to-end.
- Relative to the immediate post-helper benchmark, the non-TP LM-head shapes improved another `~1.70x` from dispatch retuning alone:
  - `3B_LM_Head`: `796.736 us` -> `467.968 us`
  - `7B_LM_Head`: `1370.112 us` -> `806.624 us`
  - `14B_LM_Head`: `1947.648 us` -> `1146.880 us`

At this point, `IQ4_NL` LM-head is no longer sitting in the low-teens bandwidth regime. The remaining gap to the Q-family is real, but the next win is likely to come from generalizing the better `IQ4_NL` exact tuning to more non-LM shapes rather than from another immediate rewrite of the decode helper.

## Non-LM Heuristic Retune Update

The same exact-entry retune method was then extended to the full non-LM `IQ4_NL` shape set:

- base: `Attn`, `FFN_Up`, `FFN_Down`
- TP2: `Attn_QKV`, `Attn_Wo`, `FFN_Up`, `FFN_Down`
- TP4: `Attn_QKV`, `Attn_Wo`, `FFN_Up`, `FFN_Down`

across `0.5B`, `3B`, `7B`, and `14B` model sizes.

A constrained sweep over the existing `kpar` / `wide` families showed that only a handful of non-LM entries were already at the best current-kernel setting. Most of the stale entries remained in the `kpar` family but wanted different exact values for `tile_n`, `target_waves`, `mkg`, and `force_two_phase`.

The generated `IQ4_NL` exact table in `src/v2/kernels/cuda/CUDANativePayloadGemvDispatchHeuristicGenerated.inc` was updated accordingly and then validated with the main `v2_perf_cuda_blockwise_tensorcore_gemm` benchmark path.

Realized benchmark summary versus the original saved matrix:

- Count: `44` non-LM shapes
- Average speedup: `1.94x`
- Average `% peak HBM`: `7.45%` -> `16.38%`

Largest `% peak` gains:

| Shape | Speedup | Old % Peak | New % Peak |
|-------|---------|------------|------------|
| 14B_FFN_Up | 3.14x | 10.5 | 33.0 |
| 7B_FFN_Up | 3.05x | 10.6 | 32.5 |
| 14B_FFN_Down | 3.03x | 10.5 | 31.8 |
| 14B_TP2_FFN_Down | 2.81x | 10.6 | 29.7 |
| 3B_FFN_Up | 2.77x | 10.2 | 28.3 |
| 14B_Attn | 2.81x | 10.0 | 28.0 |

Average by shape family:

| Family | Avg Speedup | Avg % Peak Before | Avg % Peak After |
|--------|-------------|-------------------|------------------|
| Attn | 1.63x | 5.92 | 11.44 |
| FFN_Up | 2.19x | 8.82 | 20.74 |
| FFN_Down | 2.20x | 8.64 | 20.27 |
| TP2_Attn_QKV | 1.58x | 6.28 | 11.65 |
| TP2_Attn_Wo | 1.70x | 6.35 | 12.93 |
| TP2_FFN_Up | 2.11x | 9.10 | 20.55 |
| TP2_FFN_Down | 2.14x | 8.97 | 20.43 |
| TP4_Attn_QKV | 1.37x | 4.72 | 7.25 |
| TP4_Attn_Wo | 1.41x | 4.95 | 8.50 |
| TP4_FFN_Up | 1.95x | 7.47 | 15.90 |
| TP4_FFN_Down | 2.07x | 7.50 | 16.95 |

Interpretation:

- The non-LM `IQ4_NL` path had substantial additional heuristic headroom after the helper optimization, especially in the FFN shapes.
- FFN and large attention shapes now sit in the high-20s to low-30s `% peak` range instead of clustering around `~10%`.
- The TP4 attention variants improved less than the FFN families, but they still moved in the right direction and no benchmark regressions were observed in the validated set.

## Focused Gap-Fill Update

After the broad non-LM retune, a follow-up comparison between the main benchmark and the sweep winners showed a smaller set of remaining gaps concentrated in:

- `0.5B_FFN_Up`
- `0.5B_FFN_Down`
- `0.5B_TP2_FFN_Up`
- `3B_TP4_FFN_Up`
- `7B_TP2_FFN_Up`
- a few TP2/TP4 attention-side shapes

A tighter sweep with `direct` enabled and the full default tile space found that most of the remaining attention-side headroom did **not** reproduce reliably in the main benchmark path. Those speculative entries were reverted.

The retained second-pass wins were modest but real:

| Shape | Before Gap-Fill | After Gap-Fill | Incremental Speedup | % Peak Before | % Peak After |
|-------|------------------|----------------|---------------------|---------------|--------------|
| 0.5B_FFN_Up | 28.544 us | 20.480 us | 1.39x | 9.3 | 12.9 |
| 0.5B_FFN_Down | 24.576 us | 20.480 us | 1.20x | 10.8 | 12.9 |
| 0.5B_TP2_FFN_Up | 25.600 us | 21.504 us | 1.19x | 5.2 | 6.2 |
| 3B_TP4_FFN_Up | 24.576 us | 21.504 us | 1.14x | 13.9 | 15.8 |
| 7B_TP2_FFN_Up | 74.752 us | 72.704 us | 1.03x | 27.4 | 28.1 |
| 7B_TP4_Attn_QKV | 21.504 us | 20.480 us | 1.05x | 9.1 | 9.5 |

Final interpretation:

- Yes, `IQ4_NL` could be pushed a bit higher after the broad retune, but the remaining gains are now shape-specific and much smaller than the first two passes.
- The robust residual headroom is mostly in a few small and medium FFN shapes, not across the whole `IQ4_NL` surface.
- The next large win is unlikely to come from more exact-entry tuning alone. At this point, another material jump probably requires a deeper kernel-structure change rather than more heuristic churn.