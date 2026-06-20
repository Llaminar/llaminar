# Handover: CUDA GEMM / FusedGateUp Sweep for Qwen3.5 9B

Date: 2026-05-29

## Goal

Continue the current effort to tighten CUDA native-VNNI `GEMM` / `GEMM_FUSED_GATE_UP` prefill performance so Llaminar beats llama.cpp on the Qwen3.5 9B Q8 model.

The immediate user request at handoff was:

> Fix the CUDA native-VNNI GEMM tile sweep to use the modern API and run it to get a full results table.

## Benchmark context

Baseline numbers already measured in this session:

- Qwen3.5 4B Q8: Llaminar now beats llama.cpp after the earlier GDN/padded-prefill work.
  - Llaminar default padded bucket: ~5843 tok/s prefill, ~137.6 tok/s decode.
  - llama.cpp: ~5818 tok/s prefill, ~135.4 tok/s decode.
- Qwen3.5 9B Q8 is close but still slightly behind on prefill.
  - Llaminar default padded 600 bucket: ~3965 tok/s prefill, ~85.1 tok/s decode.
  - Llaminar exact 595 bucket: ~3987 tok/s prefill, ~85.1 tok/s decode.
  - llama.cpp mainline CUDA pp595: ~4023.5 tok/s prefill, ~84.8 tok/s decode.

Stage timing for 9B (`LLAMINAR_GPU_STAGE_TIMING=1`, short decode run) showed the remaining hot spots are GEMM, not GDN:

- `GEMM_FUSED_GATE_UP`: ~30.6% of prefill time
- `GEMM`: ~24.0% of prefill time
- `GDN_RECURRENCE`: ~16.2% of prefill time

Exact 9B prefill GEMM shapes captured with DEBUG logs:

- Attention/projection GEMMs: `M=600, N=4096, K=4096`
- Fused gate/up: `M=600, K=4096, N_gate=12288, N_up=12288`
- Down projection / fused SwiGLU+GEMM: `M=600, N=4096, K=12288`

Relevant capture command:

```bash
LLAMINAR_LOG_LEVEL=DEBUG LLAMINAR_GPU_STAGE_TIMING=0 \
  ./build_v2_release/llaminar2 benchmark \
  -d cuda:0 -m models/Qwen3.5-9B-Q8_0.gguf -n 1 \
  2>&1 | grep -E "FusedGateUpGEMMStage\] Execute|GEMMStage\] Execute GEMM" | head -n 120
```

## Current git state at handoff

Dirty files at handoff:

```text
 M .githooks/run_benchmark_check.sh
 M src/v2/execution/compute_stages/stages/GDNRecurrenceStage.cpp
 M src/v2/execution/compute_stages/stages/GDNRecurrenceStage.h
 M src/v2/kernels/cuda/gdn/CUDAGatedDeltaNet.h
 M src/v2/kernels/cuda/gdn/CUDAGatedDeltaNetKernels.cu
 M src/v2/kernels/cuda/gdn/CUDAShortConvolution.h
 M src/v2/kernels/cuda/gemm/CUDANativeVNNIPrefillKernels.cu
 M src/v2/utils/DebugEnv.h
 M tests/v2/integration/kernels/cuda/Test__CUDAGDNPaddedRealLength.cpp
 M tests/v2/performance/kernels/cuda/gemm/CUDANativeVNNIGemmPerfCommon.h
 M tests/v2/unit/stages/Test__PrefillGraphCapturability.cpp
```

Most dirty files are from the previous successful GDN graph-capture optimization. Do **not** revert them unless explicitly asked.

GEMM-related current edits:

1. `src/v2/kernels/cuda/gemm/CUDANativeVNNIPrefillKernels.cu`
   - Added env-controlled force tile knobs for full-model benchmarking:
     - `LLAMINAR_FORCE_PREFILL_TILE`
     - `LLAMINAR_FORCE_PREFILL_SPLIT_K`
   - Build succeeded after this change:
     ```bash
     ninja -C build_v2_release llaminar2
     ```

2. `tests/v2/performance/kernels/cuda/gemm/CUDANativeVNNIGemmPerfCommon.h`
   - Added Qwen3.5 9B shapes to `kQwenShapes`:
     - `9B_Attn`: `N=4096, K=4096`
     - `9B_FFN_Up`: `N=12288, K=4096`
     - `9B_FFN_Down`: `N=4096, K=12288`
     - `9B_LM_Head`: `N=248320, K=4096`
   - Attempted to modernize `runKernel()` by manually creating a `CUDAQuantisedGemmKernel`, wrapping it in a `PreparedGemmHandle`, and registering it in `PreparedWeightStore`.
   - This removed the original legacy `KernelFactory::createGemm()` exception, but the approach is not correct/reliable yet: a smoke run produced an invalid zero-time row.

## Current failing symptom

The old sweep failure was:

```text
[KernelFactory] Legacy createGemm() GPU path removed. Use GPU pipeline (DeviceLoadPipeline/WeightVRAMPool) for CUDA weight preparation.
```

After the partial `runKernel()` patch, the smoke CSV at `/tmp/gemm_9b_q8_m600_smoke.csv` now contains:

```csv
shape,m,n,k,tile,tile_id,strategy,split_k,tiles,min_us,mean_us,tops,pct_peak,gpu
9B_Attn,600,4096,4096,AUTO,-1,AUTO,0,0,0.000,0.000,0.0000,0.00,0
```

This indicates the manual direct-kernel path is still wrong. Treat the current `runKernel()` patch as a checkpoint, not a finished fix.

## Recommended next fix

Use the same modern API pattern already used by CUDA GEMM integration tests, not manual construction of `CUDAQuantisedGemmKernel`.

Reference file:

- `tests/v2/integration/kernels/cuda/Test__CUDAGemmParity.cpp`

Reference helper in that file:

```cpp
ITensorGemm *getPreparedKernel(const TensorBase *tensor, DeviceId device_id)
{
    static std::vector<std::shared_ptr<llaminar::v2::kernels::KernelFactory::PreparedGemmHandle>> handles;
    auto prepared = llaminar::v2::kernels::KernelFactory::prepareGemmHandleLocal(tensor, device_id);
    if (!prepared)
    {
        return nullptr;
    }
    handles.push_back(std::move(prepared));
    return llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(handles.back().get());
}
```

Important nearby comment in `Test__CUDAGemmParity.cpp`:

```cpp
// Note: ensureOnDevice is NOT needed for CUDA quantized GEMM path -
// prepareGpuGemmOnDemand handles the upload via WeightVRAMPool internally
```

Suggested change in `tests/v2/performance/kernels/cuda/gemm/CUDANativeVNNIGemmPerfCommon.h::runKernel()`:

1. Remove the current manual prepared-store/direct `CUDAQuantisedGemmKernel` construction.
2. Do **not** call `weights->ensureOnDevice(device)` before preparation.
3. Use `KernelFactory::prepareGemmHandleLocal(weights, device)`.
4. Keep the returned `PreparedGemmHandle` alive for the duration of `runKernel()` (local `std::shared_ptr` is enough for a single run; a static vector is only needed if returning raw kernels outside the helper).
5. Get the `ITensorGemm*` via `KernelFactory::getOrCreateGemmEngine(handle.get())`.
6. Bind workspace exactly as the existing harness does.
7. Before trusting timing, explicitly check CUDA errors after warmup/bench calls and after `cudaEventSynchronize(stop)` so illegal-access failures cannot serialize as `0.000 us` rows.

Pseudo-patch shape:

```cpp
DeviceId device = DeviceId::cuda(cuda_device_id);

auto prepared = KernelFactory::prepareGemmHandleLocal(weights, device);
if (!prepared)
    throw std::runtime_error("prepareGemmHandleLocal returned null");

ITensorGemm *kernel = KernelFactory::getOrCreateGemmEngine(prepared.get());
if (!kernel)
    throw std::runtime_error("getOrCreateGemmEngine returned null");

auto *workspace_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
// Existing workspace allocation/binding code follows.
```

If `prepareGemmHandleLocal()` is insufficient for the sweep binary due to test linkage or lifetime issues, fall back to wiring a tiny `WeightVRAMPool`/`DeviceLoadPipeline` path by copying from:

- `tests/v2/unit/Test__DeviceLoadPipeline.cpp`
- `tests/v2/unit/Test__FPWeightPipeline.cpp`

However, try `prepareGemmHandleLocal()` first because `Test__CUDAGemmParity.cpp` already validates it for CUDA GEMM and explicitly documents that it handles the upload via `WeightVRAMPool`.

## Commands to run after fixing the sweep harness

Rebuild only the sweep target:

```bash
ninja -C build_v2_release tests/v2/v2_perf_cuda_native_vnni_gemm
```

Run a one-case smoke first:

```bash
LLAMINAR_TILE_SWEEP_FORMAT=Q8_0 \
LLAMINAR_TILE_SWEEP_SHAPES=9B_Attn \
LLAMINAR_TILE_SWEEP_PREFILL_M=600 \
LLAMINAR_TILE_SWEEP_STRATEGIES=auto \
LLAMINAR_TILE_SWEEP_WARMUP=1 \
LLAMINAR_TILE_SWEEP_BENCH=1 \
LLAMINAR_TILE_SWEEP_CSV=/tmp/gemm_9b_q8_m600_smoke.csv \
./build_v2_release/tests/v2/v2_perf_cuda_native_vnni_gemm \
  --gtest_filter=CUDANativeVNNIGemmPerf.TileSweep_AllStrategies \
  --gtest_brief=1

cat /tmp/gemm_9b_q8_m600_smoke.csv
```

Expected smoke result after the fix: non-zero `min_us` and no CUDA illegal-access/sticky-error messages.

Then run the full 9B Q8 table:

```bash
LLAMINAR_TILE_SWEEP_FORMAT=Q8_0 \
LLAMINAR_TILE_SWEEP_SHAPES=9B_Attn,9B_FFN_Up,9B_FFN_Down \
LLAMINAR_TILE_SWEEP_PREFILL_M=600 \
LLAMINAR_TILE_SWEEP_STRATEGIES=auto,std,sk1,sk2,bk256,bk256_sk \
LLAMINAR_TILE_SWEEP_SPLIT_K=1,2,4,8 \
LLAMINAR_TILE_SWEEP_BK256_SPLIT_K=1,2,4,8 \
LLAMINAR_TILE_SWEEP_WARMUP=2 \
LLAMINAR_TILE_SWEEP_BENCH=5 \
LLAMINAR_TILE_SWEEP_CSV=/tmp/gemm_9b_q8_m600_sweep.csv \
./build_v2_release/tests/v2/v2_perf_cuda_native_vnni_gemm \
  --gtest_filter=CUDANativeVNNIGemmPerf.TileSweep_AllStrategies \
  --gtest_brief=1
```

Parse winners:

```bash
python3 - <<'PY'
import csv, collections
rows=list(csv.DictReader(open('/tmp/gemm_9b_q8_m600_sweep.csv')))
by=collections.defaultdict(list)
for r in rows:
    r['min_us']=float(r['min_us']); r['mean_us']=float(r['mean_us'])
    r['split_k']=int(r['split_k']); r['tile_id']=int(r['tile_id']); r['tiles']=int(r['tiles'])
    by[(r['shape'], int(r['m']))].append(r)

for key, rs in sorted(by.items()):
    auto=[r for r in rs if r['strategy']=='AUTO']
    auto_us=auto[0]['min_us'] if auto else None
    print('\n', key)
    if auto:
        r=auto[0]
        print(f" AUTO {r['tile']} tid={r['tile_id']} {r['strategy']} sk={r['split_k']} tiles={r['tiles']} min={r['min_us']:.3f} mean={r['mean_us']:.3f}")
    for r in sorted(rs, key=lambda r:r['min_us'])[:10]:
        speedup=(auto_us/r['min_us']) if auto_us and r['min_us'] > 0 else 0.0
        print(f" BEST {r['tile']} tid={r['tile_id']} {r['strategy']} sk={r['split_k']} tiles={r['tiles']} min={r['min_us']:.3f} mean={r['mean_us']:.3f} speedup={speedup:.3f}")
PY
```

## After getting the table

Likely next steps once the full table is valid:

1. Add shape-specific Q8_0 entries for the 9B shapes to the generated dispatch table path or exception mechanism.
   - Dispatch table file: `src/v2/kernels/cuda/gemm/CUDANativeVNNIPrefillDispatchGenerated.inc`
   - Manual exception file: `src/v2/kernels/cuda/gemm/CUDANativeVNNIPrefillDispatchExceptions.json`
   - Kernel dispatch/heuristics: `src/v2/kernels/cuda/gemm/CUDANativeVNNIPrefillKernels.cu`
2. If a global env choice appears best, validate via full-model benchmark first using the already-added env controls:
   ```bash
   LLAMINAR_FORCE_PREFILL_TILE=<tile_id> \
   LLAMINAR_FORCE_PREFILL_SPLIT_K=<split_k> \
   ./build_v2_release/llaminar2 benchmark \
     -d cuda:0 -m models/Qwen3.5-9B-Q8_0.gguf -n 128
   ```
3. Once heuristic/table changes are made, run correctness and focused benchmark checks:
   ```bash
   ctest --test-dir build_v2_integration -R "V2_Integration_CUDAGemmParity|V2_Integration_CUDAGemmNonDeterminism|V2_Unit_FusedGateUpGEMMStage" --output-on-failure
   ```
4. Re-run the 9B benchmark against llama.cpp baseline:
   ```bash
   LLAMINAR_GPU_STAGE_TIMING=0 LLAMINAR_LOG_LEVEL=INFO \
     ./build_v2_release/llaminar2 benchmark \
     -d cuda:0 -m models/Qwen3.5-9B-Q8_0.gguf -n 128
   ```

## Notes / cautions

- `v2_perf_cuda_native_vnni_gemm` currently failed a full attempted sweep because the old path was removed. Do not spend time on `KernelFactory::createGemm()`; it is intentionally gone for CUDA.
- Current direct `CUDAQuantisedGemmKernel` construction in the perf harness is suspect and produced `0.000 us`; replace it rather than trying to paper over the CSV parser.
- Keep the GDN changes intact. They are unrelated to the sweep harness but are part of the already-successful 4B performance work.
- Builds may emit unrelated CUDA/ROCm warnings about ignored HIP return values or unused variables; the key criterion for the sweep is valid non-zero timings and no sticky CUDA errors.