# CUDA Native-Payload Tensor-Core Prefill Project Plan

## Goal

Build a CUDA prefill GEMM kernel family that is competitive with CUTLASS while preserving the native-payload VRAM advantage of sub-8-bit weight formats.

This plan is specifically for the **MMA tensor-core prefill path** (`M > 1`).

The target architecture is:

- blockwise quantized activations
- weights kept in compact native payload format in VRAM
- no persistent expanded INT8 weight buffer
- decode-in-load or decode-to-shared inside the kernel
- tensor-core compute via `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`

The immediate priority is **Q4_0**, then **IQ4_NL**, because those two formats are the cleanest bridge from the current CUDA fused-TC scaffold to a native-payload tensor-core design.

## Non-Negotiable Constraints

These are hard constraints for the implementation and for any agent working this plan.

1. Weights must remain in native payload form in VRAM.
2. There must be no persistent `d_int8_data` or `d_int8_data_tc_blocked` requirement for the new native-payload prefill path.
3. Activation quantization remains blockwise INT8 with per-row per-32-element scales.
4. Weight decode must happen transiently inside the kernel, into registers or shared memory only.
5. The new path must not regress correctness relative to the current native-payload CUDA decode path.
6. Performance comparison must be against the existing CUTLASS path using the current benchmark harness.
7. The current expanded-INT8 tensor-core path remains available as a fallback until the native-payload path is proven.

## Why This Is Salvageable

The current fused tensor-core work is not throwaway.

What is reusable from the existing CUDA fused-TC path:

- CTA and warp tiling
- the `mma.sync` compute core
- `ldmatrix` fragment loading from shared memory
- split-K machinery
- cp.async staging and multi-stage pipeline structure
- the current shape-family benchmarking harness

What is not reusable as-is:

- the B-side global-memory contract (`[K/32][N][32]` predecoded INT8)
- the assumption that weights contribute only a single late `scales_B[N]` epilogue factor

The correct salvage strategy is therefore:

- keep the current A-path and MMA scaffold where possible
- replace the B-path with compact payload load + in-kernel decode
- move weight scaling/correction into the K-block loop

This is the same structural idea already used by ROCm native-VNNI GEMM:

- compact payload in HBM
- decode into LDS/registers during load
- compute phase consumes decoded INT8 tile

## Kernel Contract

This section defines the contract for the new kernel family.

### Public Entry Point

The long-term generic entry point should be a native-payload-specific sibling of the current fused-TC API, not a mutation of the expanded-INT8 API.

Suggested C API:

```cpp
extern "C" bool cudaFusedTCNativePayloadGemm_fp32(
    const int8_t* d_A_int8,
    float* d_C_fp32,
    const float* d_scales_A_block,
    const uint8_t* d_payload,
    const uint16_t* d_block_scales,
    const uint16_t* d_block_mins,
    const uint32_t* d_block_emins,
    uint8_t codebook_id,
    uint32_t blocks_per_row,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    const float* d_C_existing,
    const float* d_bias,
    int cuda_device_id,
    void* stream);
```

Contract notes:

- `d_payload`, `d_block_scales`, `d_block_mins`, `d_block_emins` are the only persistent weight buffers required for the native-payload path.
- `codebook_id` matches the existing native payload codebook IDs already used by CUDA decode and ROCm native-VNNI.
- `blocks_per_row == K / 32` for the first supported formats.
- `d_C_existing` and `d_bias` are optional and preserve the existing GEMM API behavior.

### At-Rest Weight Layout in VRAM

The new path must consume the existing native-payload upload contract, not introduce a second expanded form.

At rest in VRAM:

- `d_payload`: `[blocks_per_row * N * payload_bytes]`
- `d_block_scales`: `[blocks_per_row * N]`
- `d_block_mins`: `[blocks_per_row * N]` for asymmetric formats only
- `d_block_emins`: `[blocks_per_row * N]` for formats that require it

Linear index for a weight block is:

```cpp
linear = b * N + n;
```

This is the existing interleaved-by-N native payload layout used by CUDA packing via `VnniPackContext`.

### Temporary Decode Layout Inside the Kernel

The kernel is allowed to transiently expand a B tile to INT8 **inside shared memory or registers only**.

Required invariant:

- no global-memory expanded INT8 buffer
- no global-memory tc-blocked buffer

The temporary B tile should be shaped so the existing MMA path can consume it with minimal change:

```cpp
smem_B_decoded[STAGES][BN * SMEM_STRIDE]
```

Meaning:

- for each active K-block (`32` K elements), decode compact payload into the same logical INT8 tile that the current `load_B_tile` path would have copied from `d_weights_int8_tc_blocked`
- then reuse `ldmatrix` + `mma.sync` from shared memory

### Activation Contract

The activation side stays unchanged:

- `A`: row-major INT8
- `scales_A`: per-row per-32-element block FP32 scale

The new kernel must preserve the existing blockwise activation quantization contract exactly.

### Scaling Contract

This is the most important change from the current fused-TC path.

The current expanded-INT8 fused-TC kernel uses:

- INT8 MMA accumulation into INT32 per K-block
- multiply by `scale_A_block`
- late multiply by one `scales_B[n]`

That is not valid for native payload formats with block-local scales/mins.

The new native-payload tensor-core contract is:

```cpp
for each K-block b:
    decode B_block(b, n) -> transient INT8 values
    dot = s8xs8 tensor-core accumulate for this 32-element block
    acc_fp32 += float(dot) * scale_A(m, b) * scale_B(n, b)
```

For asymmetric or dual-scale formats, correction terms are applied per block, not in a single epilogue.

Implications:

- weight scale application moves into the K-block loop
- the compute path is format-aware, even if the MMA fragment consumption is shared
- the simple `smem_scales_B[N]` epilogue cache is not sufficient for the native-payload path

### Format Support by Phase

#### Phase 1 formats

- `Q4_0` (`codebook_id = 0`)
- `IQ4_NL` (`codebook_id = 4`)

Why first:

- symmetric decode
- no per-block min correction
- no emin array
- already supported by CUDA native payload decode helpers

#### Phase 2 formats

- `Q4_1`, `Q5_0`, `Q5_1`

Why second:

- still relatively small payloads
- introduces asymmetric min correction without the worst dual-scale complexity

#### Phase 3 formats

- `Q6_K`, `Q3_K`, `Q2_K`
- `IQ3_*`, `IQ2_*`, `IQ1_*`

Why last:

- dual-scale and/or embedded-min correction
- more decode-heavy formats
- may require cooperative `sum(A)` side paths like the ROCm native-VNNI kernels

### Dispatch Contract

The new path should be a sibling execution family under the CUDA quantized GEMM dispatcher.

Required high-level dispatch split:

1. INT8-expanded path
2. native-payload path

Within native payload:

1. GEMV (`M == 1`) -> existing native-payload GEMV path
2. GEMM (`M > 1`) -> new native-payload tensor-core prefill path

The new tensor-core path must not require `d_weights_int8_tc_blocked` to exist.

## Reference Kernel Shape for Phase 1

Phase 1 should not start with dynamic shape exploration. It should start with one clean kernel contract that is simple enough to make correct and fast.

### Recommended first kernel: `Q4_0`, `BM=64`, `BN=128`, `WARPS_M=2`, `WARPS_N=2`, `STAGES=2`

Reasons:

- closest to the current best-performing CUDA fused-TC shape family
- manageable register footprint compared with the 128x128 path
- enough N coverage to stay relevant for FFN-up and LM-head style shapes
- enough M coverage to be useful for prefill

### Phase 1 B-side algorithm

For each `(kb, n_tile)`:

1. cooperatively load compact Q4_0 payload bytes for the tile from global memory
2. decode each 32-element weight block into 8 packed `int32` groups or directly into INT8 bytes in shared memory
3. make the decoded shared-memory layout match the current `ldmatrix` expectations
4. run the existing `mma.sync` compute core
5. apply `scale_A(m, kb) * scale_B(n, kb)` immediately after each K-block contribution

### Phase 1 anti-goals

Do not do these in the first implementation:

- all-codebook generic kernel
- asymmetric correction in the same patch as Q4_0
- STAGES=3 tuning before the decode-in-load path is correct
- heuristic family proliferation before one path is competitive

## Deliverables

These are concrete deliverables for an autonomous overnight agent.

### Deliverable D0: Contract and dispatch skeleton

Must exist before tuning begins.

- new native-payload tensor-core API contract committed
- dispatcher route for native-payload GEMM added behind a guard or feature flag
- no requirement for `d_int8_data_tc_blocked` in the new path
- benchmark harness can selectively run the new path

### Deliverable D1: `Q4_0` native-payload tensor-core prefill kernel

Required:

- correct on representative prefill shapes
- uses native payload buffers directly from VRAM
- does not create or require a persistent expanded INT8 or tc-blocked buffer
- blockwise activation quantization preserved

### Deliverable D2: `IQ4_NL` native-payload tensor-core prefill kernel

Required:

- same contract as D1
- correctness proven on representative IQ4_NL shapes
- competitive performance relative to CUTLASS on the same harness

### Deliverable D3: shape-specialized kernel set

Required:

- at least a balanced family and a width-oriented family for the new native-payload path
- force-family override for direct A/B testing
- dispatch heuristics backed by measured same-shape comparisons

### Deliverable D4: parity evidence package

Required:

- correctness results
- benchmark CSVs
- aggregate parity report against CUTLASS
- concise summary of shape winners and losers

## Performance Targets

The benchmark source of truth is:

- `build_v2_release/tests/v2/v2_perf_cuda_native_payload_gemm`

### Primary overnight target

For `Q4_0`, on the existing representative/full prefill harness:

- `speedup_vs_cutlass >= 1.0` median on the target sweep
- `speedup_vs_cutlass >= 1.0` mean on the target sweep
- no correctness regressions

### Secondary target

For `IQ4_NL`:

- competitive with CUTLASS on the same representative shape set
- ideally parity, but Q4_0 parity has priority over broad format coverage

### Production-shape priority

If tradeoffs are required, prioritize these shape classes first:

1. FFN-up
2. LM-head
3. attention QKV / attention Wo
4. FFN-down

Rationale:

- FFN-up and LM-head dominate the currently painful width-heavy frontier
- solving those proves the decode-in-load tensor-core architecture is viable

## Overnight Agent Work Plan

This is the concrete sequence an autonomous agent should follow.

### Phase 0: establish branch-local architecture

1. Add a new CUDA native-payload tensor-core kernel file, separate from the existing expanded-INT8 fused-TC file.
2. Add a dedicated dispatcher entrypoint in `CUDAQuantisedGemmKernel.cpp` guarded behind an env flag.
3. Ensure the path is callable with:
   - `d_A_int8`
   - `d_scales_A_block`
   - `d_weights_native_payload`
   - `d_weights_native_scales`
   - optional min/emin arrays
4. Ensure the path never asks for `d_weights_int8_tc_blocked`.

### Phase 1: get `Q4_0` correctness first

1. Implement only `Q4_0`.
2. Hardcode one kernel family initially.
3. Reuse current A loading and MMA core where possible.
4. Replace B-side load with compact payload load + decode-to-shared.
5. Move weight scaling into the K-block loop.
6. Run focused correctness on representative Q4_0 shapes.

Exit gate:

- no correctness failures on the focused Q4_0 set

### Phase 2: reach first useful performance

1. Benchmark the single-family `Q4_0` kernel on the representative sweep.
2. Profile whether the bottleneck is:
   - decode cost
   - shared-memory bandwidth
   - register pressure
   - insufficient CTA coverage
3. Tune in this order:
   - shared-memory decode layout
   - per-block scale handling placement
   - tile size
   - stages
   - split-K

Exit gate:

- representative Q4_0 sweep shows clear movement toward CUTLASS parity

### Phase 3: add a second family only if measured

1. Introduce a width-specialized or K-heavy-specialized family only after same-shape evidence.
2. Keep a family-force override to benchmark identical shapes under alternate families.
3. Do not add more than one extra family without direct benchmark evidence.

Exit gate:

- direct family-force comparisons justify the heuristic split

### Phase 4: full Q4_0 parity run

1. Run the full two-GPU Q4_0 sweep.
2. Produce CSV and aggregate stats.
3. If mean or median is below parity, identify the worst 10 shapes and retune only those classes.

Exit gate:

- Q4_0 mean and median parity achieved, or a concise blocked-by report with worst-shape analysis

### Phase 5: IQ4_NL follow-through

1. Swap in IQ4_NL decode using the same kernel skeleton.
2. Re-run correctness.
3. Benchmark representative IQ4_NL shapes.
4. Reuse shape-family logic only if it transfers.

## Commands and Evidence

### Build

```bash
cmake --build build_v2_release --target v2_perf_cuda_native_payload_gemm --parallel
```

### Focused correctness

```bash
LLAMINAR_CUDA_NATIVE_GEMM_FORMATS=Q4_0 \
LLAMINAR_CUDA_NATIVE_GEMM_SHAPES='3B_FFN_Up,7B_FFN_Up,14B_Attn,14B_FFN_Up,14B_TP2_FFN_Down,14B_LM_Head' \
./build_v2_release/tests/v2/v2_perf_cuda_native_payload_gemm --gtest_filter='*Correctness*'
```

### Focused performance

```bash
LLAMINAR_CUDA_NATIVE_GEMM_FORMATS=Q4_0 \
LLAMINAR_CUDA_NATIVE_GEMM_SHAPES='3B_FFN_Up,7B_FFN_Up,14B_Attn,14B_FFN_Up,14B_TP2_FFN_Down,14B_LM_Head' \
LLAMINAR_CUDA_NATIVE_GEMM_PREFILL_M=128 \
LLAMINAR_CUDA_NATIVE_GEMM_WARMUP=1 \
LLAMINAR_CUDA_NATIVE_GEMM_BENCH=3 \
./build_v2_release/tests/v2/v2_perf_cuda_native_payload_gemm --gtest_filter='*Performance*'
```

### Full Q4_0 parity sweep

```bash
LLAMINAR_CUDA_NATIVE_GEMM_FORMATS=Q4_0 \
LLAMINAR_CUDA_NATIVE_GEMM_PREFILL_M=128 \
LLAMINAR_CUDA_NATIVE_GEMM_WARMUP=1 \
LLAMINAR_CUDA_NATIVE_GEMM_BENCH=3 \
LLAMINAR_CUDA_NATIVE_GEMM_SWEEP_OUT=/tmp/q4_0_native_tc.csv \
./build_v2_release/tests/v2/v2_perf_cuda_native_payload_gemm --gtest_filter='*Performance*'
```

### Required evidence artifacts

An overnight agent should leave behind:

1. one correctness log for focused Q4_0
2. one focused performance log for Q4_0
3. one full-sweep CSV for Q4_0
4. one aggregate summary with:
   - mean speedup vs CUTLASS
   - median speedup vs CUTLASS
   - parity count
   - worst 10 shapes
5. equivalent IQ4_NL artifacts if Phase 5 is reached

## Definition of Done

The project is done when all of the following are true for the new native-payload tensor-core prefill path:

1. Weights remain compact in VRAM for the supported formats.
2. No persistent expanded INT8 or tc-blocked weight buffer is required.
3. Correctness passes on the benchmark harness.
4. The kernel family is competitive with CUTLASS on the target sweep.
5. The dispatcher can choose the native-payload tensor-core path without breaking the fallback expanded path.

## Definition of Failure

The project should be considered blocked, not complete, if any of the following is true:

1. The only working tensor-core path still requires `d_int8_data_tc_blocked`.
2. The kernel preserves compact VRAM but falls materially short of CUTLASS without a clear next bottleneck.
3. The implementation becomes an all-format generic decoder before Q4_0 parity is demonstrated.
4. Heuristic tuning dominates the work before the decode-in-load architecture is proven correct and competitive.

## Overnight Agent Rules

Any agent iterating on this overnight should follow these rules strictly.

1. Keep the new native-payload tensor-core path behind an explicit gate until Q4_0 correctness is stable.
2. Start with Q4_0 only.
3. Reuse the current A-tile and MMA machinery before inventing new compute structures.
4. Treat native payload decode and scale handling as the primary design problem, not dispatch heuristics.
5. After every structural kernel change:
   - build
   - run focused correctness
   - run focused performance
6. Only run the full sweep when the focused representative set improves.
7. Revert dead-end kernel families quickly.
8. Optimize for measured results, not theoretical tile aesthetics.

## Recommended First Implementation Cut

If an agent has only one overnight window, the most likely winning scope is:

1. Add `cudaFusedTCNativePayloadGemmQ40_fp32`
2. Hook it up only for `Q4_0`, `M > 1`, feature-flagged
3. Implement one `BM=64`, `BN=128`, `WARPS_M=2`, `WARPS_N=2`, `STAGES=2` kernel
4. Decode Q4_0 payload into shared-memory INT8 tiles inside `load_B_tile`
5. Apply `scale_A * scale_B_block` inside the K-block loop
6. Benchmark until the representative Q4_0 set is within striking distance of CUTLASS

That is the smallest credible slice that preserves the project’s original intent.