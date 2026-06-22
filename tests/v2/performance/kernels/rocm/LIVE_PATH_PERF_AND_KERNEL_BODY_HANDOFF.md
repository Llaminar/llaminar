# Live-Path Perf + `KERNEL_BODY` Handoff

This is the single handoff reference for running **live-path** ROCm perf benchmarks and locating the exact production kernel code paths controlled by `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_KERNEL_BODY`.

## 1) Build the live-path benchmark target

```bash
cmake --build build_v2_release --target v2_perf_rocm_prefill_dispatch_comparison --parallel
```

Benchmark source: [tests/v2/performance/kernels/rocm/Perf__ROCmPrefillDispatchComparison.cpp](Perf__ROCmPrefillDispatchComparison.cpp)

Benchmark binary:

```bash
./build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison
```

## 2) Environment setup for live-path FFN override

Use this exact baseline env block before each run:

```bash
export LLAMINAR_ROCM_VNNI_PREFILL_EXPERIMENTAL=1
export LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE=1
export LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_GRID_KPAR=1
export LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_SPLITS=4
export LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_CPT=4
export LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_VARIANT=1
export LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_GRID_SWIZZLE=1
```

`KERNEL_BODY` values:
- `0` = baseline production loop body
- `1` = software-pipelined loop body (V7-style)
- `2` = LDS B-tile + software-pipelined loop body (V10-style)

## 3) Run A/B/C on live path

```bash
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_KERNEL_BODY=0 ./build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison | tee /tmp/body0.txt
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_KERNEL_BODY=1 ./build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison | tee /tmp/body1.txt
LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_KERNEL_BODY=2 ./build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison | tee /tmp/body2.txt
```

Quick summary extraction:

```bash
grep -E 'FFN_Up|FFN_Gate|Class|Speedup|Cosine' /tmp/body{0,1,2}.txt
```

Stability loop (5 runs per variant):

```bash
for b in 0 1 2; do
  for i in {1..5}; do
    LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_KERNEL_BODY=$b \
      ./build_v2_release/tests/v2/v2_perf_rocm_prefill_dispatch_comparison \
      >> /tmp/body${b}_runs.txt
  done
done
```

## 4) Where `KERNEL_BODY` is wired in production code

### A) Kernel template with `LDS_B_TILE_CACHE`

Location: [src/v2/kernels/rocm/ROCmQuantisedGemmKernel_CK.hip](src/v2/kernels/rocm/ROCmQuantisedGemmKernel_CK.hip#L1330-L1424)

```cpp
template <int BLOCK_X, int BLOCK_Y, int CPT, bool STAGE_B = false, bool ATOMIC_ACCUM = true, bool SOFTWARE_PIPELINED = false, bool LDS_B_TILE_CACHE = false>
__global__ void qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int k_groups_per_slice,
    int slice_index_base,
    int grid_swizzle_variant)
{
    // ...

    if constexpr (CPT == 4 && LDS_B_TILE_CACHE)
    {
        constexpr int KT = 8;
        __shared__ int32_t b_lds[KT * 128];

        const int tid = threadIdx.y * BLOCK_X + threadIdx.x;
        const int n_block_base = n_tile_idx * (BLOCK_X * CPT);
        const int n_local = threadIdx.x * CPT;

        const int32_t* a_row = reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K);
        const int32_t* b_global_base = reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + static_cast<int64_t>(n_block_base) * 4);

        for (int kt = k_begin; kt < k_end; kt += KT)
        {
            const int kg_limit = min(KT, k_end - kt);

            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int flat = i * 256 + tid;
                const int kg_l = flat >> 7;
                const int n_l = flat & 127;
                int32_t b_val = 0;
                if (kg_l < kg_limit)
                {
                    b_val = b_global_base[static_cast<int64_t>(kt + kg_l) * N + n_l];
                }
                b_lds[kg_l * 128 + n_l] = b_val;
            }
            __syncthreads();

            // ... SDOT4 compute from LDS ...
            __syncthreads();
        }
    }
    // ...
}
```

### B) `kernel_body_variant` selection and launch mapping

Location: [src/v2/kernels/rocm/ROCmQuantisedGemmKernel_CK.hip](src/v2/kernels/rocm/ROCmQuantisedGemmKernel_CK.hip#L3244-L3333)

```cpp
const int cpt_v = (cpt == 2 || cpt == 4) ? cpt : 1;
const bool use_software_pipelined = (kernel_body_variant == 1);
const bool request_lds_b_tile = (kernel_body_variant == 2);
const bool use_grid_swizzled = (grid_swizzle_variant == 1);

// ...
const bool can_use_lds_b_tile = request_lds_b_tile && ((M % BY) == 0) && ((N % (BX * 4)) == 0);
if (can_use_lds_b_tile)
{
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, false, true>),
                       grid,
                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
}
else if (use_software_pipelined)
{
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, true, false>),
                       grid,
                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
}
else
{
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, false, false>),
                       grid,
                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
}
```

### C) Env parsing for `KERNEL_BODY` (0..2)

Location: [src/v2/utils/DebugEnv.h](../../../../../src/v2/utils/DebugEnv.h)

Search for:
- `LLAMINAR_ROCM_VNNI_PREFILL_FFN_OVERRIDE_KERNEL_BODY`
- clamp range `[0,2]`

## 5) Notes for profiling agent handoff

- The **playground** still reproduces strong V10 gains; the above commands are for the **production live path** benchmark.
- For profiler runs, wrap the same A/B/C commands in `rocprof`/`rocprofv2` with identical env and executable to keep comparisons valid.
- Focus first on FFN classes in output table:
  - `Qwen2.5-0.5B_FFN_Up`, `Qwen2.5-0.5B_FFN_Gate`
  - `Qwen2.5-3B_FFN_Up`, `Qwen2.5-3B_FFN_Gate`
