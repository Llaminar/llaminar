# Gemini Pro 3.1 FFN Kernel Playground Kit

This kit is a **fast micro-harness** for iterating on ROCm FFN Up/Gate kernel changes without running the full production benchmark suite each time.

## Files
- Harness: [Microbench__GeminiFFNKernelPlayground.hip](Microbench__GeminiFFNKernelPlayground.hip)
- Build target: `v2_perf_rocm_gemini_ffn_kernel_playground`

## What it reproduces
It locks to the exact tuning shapes you care about:
- `Qwen2.5-0.5B_FFN_Up`  → `(M,N,K)=(128,4864,896)`
- `Qwen2.5-0.5B_FFN_Gate`→ `(128,4864,896)`
- `Qwen2.5-3B_FFN_Up`    → `(128,11008,2048)`
- `Qwen2.5-3B_FFN_Gate`  → `(128,11008,2048)`

And it uses the same split-K mode you’ve been tuning (`split_k=4` by default).

## Fast iteration loop
## 1) Build once
```bash
cmake --build build_v2_release --target v2_perf_rocm_gemini_ffn_kernel_playground --parallel
```

## 2) Run quick check
```bash
./build_v2_release/tests/v2/v2_perf_rocm_gemini_ffn_kernel_playground --warmup 1 --iters 4
```

## 3) Run stability pass
```bash
./build_v2_release/tests/v2/v2_perf_rocm_gemini_ffn_kernel_playground --warmup 2 --iters 12
```

## 4) Edit only the candidate zone
Inside the harness, modify code only between:
- `GEMINI TUNING ZONE START`
- `GEMINI TUNING ZONE END`

The baseline path is intentionally duplicated and must stay unchanged for apples-to-apples comparison.

## Output interpretation
For each shape, you get:
- `baseline ms`
- `candidate ms`
- `speedup (baseline/candidate)`
- `max_abs_diff(INT32) baseline_vs_cpu_ref`
- `max_abs_diff(INT32) candidate_vs_cpu_ref`
- `max_abs_diff(INT32) baseline_vs_candidate`

Correctness uses an OpenMP-parallelized CPU reference (`int8 x int8 -> int32`), and the target is linked with OpenMP (`-fopenmp` + `OpenMP::OpenMP_CXX`).
Set `OMP_NUM_THREADS` to control CPU-check parallelism.

### Promotion gate (recommended)
Treat a variant as viable only if:
- `baseline_vs_cpu_ref == 0` and `candidate_vs_cpu_ref == 0` on all 4 shapes
- Mean speedup improves across repeats
- 3B minima do not regress

## Agent prompt template for Gemini Pro 3.1
Use this exact structure when delegating:

```text
You are tuning only the candidate branch in
Microbench__GeminiFFNKernelPlayground.hip.

Constraints:
1) Do not change baseline branch.
2) Keep launch geometry unchanged unless explicitly requested.
3) Maintain exact INT32 parity (max_abs_diff must remain 0).
4) Optimize for 3B FFN Up/Gate first, while avoiding 0.5B regressions.

Workflow:
- Propose one micro-optimization.
- Apply change in candidate zone only.
- Rebuild target v2_perf_rocm_gemini_ffn_kernel_playground.
- Run with --warmup 1 --iters 4, then --warmup 2 --iters 12.
- Report per-shape baseline/candidate/speedup and parity.
- If regressive, revert and try next variant.
```

## Suggested first variant classes
- Address generation reduction (pointer increments / typed loads)
- Control-flow simplification in tail path
- Lightweight unroll experiments only where register pressure stays stable
- Atomic-path micro-tuning without extra launches

## Notes
- This harness uses random but deterministic int8 inputs (`--seed` available).
- It is intentionally synthetic and focused on kernel throughput + parity.
- After finding a promising variant here, validate in production path with:
  - `v2_perf_rocm_prefill_dispatch_comparison`
