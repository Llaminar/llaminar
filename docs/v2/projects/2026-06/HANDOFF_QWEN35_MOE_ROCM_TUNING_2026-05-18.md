# Handoff: Qwen3.5 35B MoE ROCm Tuning Loop

Date: 2026-05-18

Scope: continue closing the ROCm single-device Qwen3.5-35B-A3B UD-Q4_K_XL throughput gap against llama.cpp while preserving PyTorch parity.

Model path used in this tuning session:

```bash
/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

Primary target device: `rocm:0` on gfx906 / MI60-class hardware.

## Current Baseline

Fresh same-host llama.cpp baseline was measured with `/tmp/llama.cpp/build-rocm/bin/llama-bench`:

| Runtime | Prefill | Decode |
|---------|---------|--------|
| llama.cpp ROCm | ~694 tok/s | ~69.3 tok/s |
| Llaminar current default, after build/runtime repairs | ~268 tok/s | ~57-58 tok/s |
| Llaminar with `LLAMINAR_ROCM_GDN_CONCURRENT_DECODE=1` | ~268 tok/s | ~59 tok/s |

Interpretation:

- Decode is close but still short of llama.cpp by roughly 10-12 tok/s.
- Prefill remains much farther from llama.cpp and should be treated as a separate second phase once decode parity is reached.
- Use no-env Release benchmark numbers for keep/reject decisions. `LLAMINAR_PROFILING=1` changes timing materially and should be used for attribution, not absolute throughput.

## Tuning Cycle

Use this loop for every candidate:

1. Start from a clean, buildable tree.
2. Run a no-env Llaminar benchmark and record prefill/decode/overall.
3. Run `LLAMINAR_PROFILING=1` for stage attribution.
4. Run rocprof for kernel attribution when stage labels are too broad.
5. Form one narrow hypothesis.
6. Dispatch one implementation subagent with explicit files, guard env, tests, and benchmark acceptance criteria.
7. Audit the returned diff before accepting.
8. Run focused tests, then Release build, then benchmarks.
9. Keep a win only if it is repeatable and correctness-safe.
10. Leave neutral/risky experiments default-off behind env flags, or revert them if they create maintenance/build risk.

Practical thresholds used so far:

- Default-enable only clear wins, not run noise. A single <1% improvement is not enough.
- Default-off is acceptable for useful experiments if tests pass and code is isolated.
- `LLAMINAR_DETERMINISTIC=1` should disable atomic or reduction-order-changing fast paths unless exact parity is proven.

## Benchmark Commands

Build Release:

```bash
cmake --build build_v2_release --target llaminar2 --parallel
```

Default Llaminar benchmark:

```bash
LLAMINAR_LOG_LEVEL=WARN \
LLAMINAR_GPU_GRAPHS=1 \
./build_v2_release/llaminar2 benchmark \
  -d rocm:0 \
  -m /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

Profiling attribution benchmark:

```bash
LLAMINAR_LOG_LEVEL=WARN \
LLAMINAR_PROFILING=1 \
LLAMINAR_GPU_GRAPHS=1 \
LLAMINAR_VALIDATE_BUFFERS=0 \
LLAMINAR_VALIDATE_INPUTS=0 \
./build_v2_release/llaminar2 benchmark \
  -d rocm:0 \
  -m /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  > /tmp/llaminar_perf/profile_qwen35_rocm.log 2>&1

rg -n "PREFILL|DECODE|Throughput|GPU STAGE TIMELINE: DECODE|GDN_|MOE_|ATTENTION|GEMM|Kernel Efficiency|tok/s" \
  /tmp/llaminar_perf/profile_qwen35_rocm.log | head -220
```

Fresh llama.cpp comparison:

```bash
HIP_VISIBLE_DEVICES=0 \
/tmp/llama.cpp/build-rocm/bin/llama-bench \
  -m /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  -ngl 99 \
  -p 595 \
  -n 128
```

If llama.cpp is missing, use the existing `/tmp/llama.cpp` checkout or rebuild it with ROCm before comparing. Do not use stale repo baselines as the final parity decision.

## rocprof Strategy

### Important Profiling Notes

- `rocprofv3` crashed on the full `llaminar2` app path in this environment when attached across MPI/bootstrap/HIP graph behavior.
- Legacy `/opt/rocm/bin/rocprof --stats` worked for app-level kernel stats.
- Use `--no-mpi-bootstrap` for profiling direct app execution when possible. This is for profiling/debug only, not final benchmark numbers.
- Turn GPU graphs off for kernel-level traces if graph replay hides individual kernels.
- For app-level traces, use short decode-heavy prompts where possible to reduce prefill noise.
- Standalone integration binaries are safer for focused kernel profiling and do not hit the app MPI/bootstrap path.

### App-Level rocprof

```bash
mkdir -p /tmp/llaminar_perf/rocprof_qwen35

LLAMINAR_LOG_LEVEL=WARN \
LLAMINAR_GPU_GRAPHS=0 \
LLAMINAR_VALIDATE_BUFFERS=0 \
LLAMINAR_VALIDATE_INPUTS=0 \
/opt/rocm/bin/rocprof --stats \
  -o /tmp/llaminar_perf/rocprof_qwen35/app_stats.csv \
  -- \
  ./build_v2_release/llaminar2 benchmark \
    --no-mpi-bootstrap \
    -d rocm:0 \
    -m /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
```

If app profiling fails, profile a focused test binary:

```bash
mkdir -p /tmp/llaminar_perf/rocprof_moe

/opt/rocm/bin/rocprof --stats \
  -o /tmp/llaminar_perf/rocprof_moe/moe_stats.csv \
  -- \
  ./build_v2_integration/tests/v2/v2_integration_rocm_moe_kernel \
    --gtest_filter='*DecodeRouteSelect*:*Grouped*:*SharedExpert*'
```

### Compact Kernel Aggregation

rocprof CSV column names vary by version. This parser tries to find kernel-name and duration columns heuristically:

```bash
python3 - <<'PY'
import csv, glob, collections, os

paths = glob.glob('/tmp/llaminar_perf/rocprof*/**/*.csv', recursive=True)
rows = collections.defaultdict(lambda: [0, 0.0])

def pick(fieldnames, needles):
    lower = {f.lower(): f for f in fieldnames or []}
    for needle in needles:
        for low, orig in lower.items():
            if needle in low:
                return orig
    return None

for path in paths:
    try:
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            name_col = pick(reader.fieldnames, ['kernel', 'name'])
            dur_col = pick(reader.fieldnames, ['duration', 'time'])
            if not name_col or not dur_col:
                continue
            for row in reader:
                name = row.get(name_col, '').strip()
                if not name:
                    continue
                raw = row.get(dur_col, '0').replace(',', '').strip()
                try:
                    dur = float(raw)
                except ValueError:
                    continue
                rows[name][0] += 1
                rows[name][1] += dur
    except Exception:
        pass

for name, (count, total) in sorted(rows.items(), key=lambda kv: kv[1][1], reverse=True)[:40]:
    print(f'{total:14.3f} {count:8d} {total / max(count, 1):12.3f}  {name}')
PY
```

## Hard Findings So Far

### Router

The live Qwen3.5 MoE model uses 256 router experts and top-k 8. Older notes mentioning 64 experts/top-k 6 are stale.

Initial profiling correction:

| Kernel | Finding |
|--------|---------|
| `rocm_moe_gate_logits_single_token_kernel` | Not the main bottleneck; roughly 10-11 us per launch in short profiling, vectorized load path, low register pressure |
| `rocm_moe_softmax_topk_decode_runtime_kernel` | Main router bottleneck before fixes; roughly 407 us per launch in runtime-router profile |

Accepted router work:

- Shared-memory wave top-k runtime path.
- Block-parallel top-k regression for 256 experts/top-k 8.
- Runtime histogram bridge so rebalance histograms no longer force host-routed decode.

Result: decode moved from the low 40 tok/s range to roughly 58 tok/s.

Default-off router experiments:

| Experiment | Env | Result |
|------------|-----|--------|
| Grouped LDS router logits | `LLAMINAR_ROCM_MOE_GROUPED_DECODE_ROUTER=1` | Correct, flat/slower |
| FP16 router gate cache | `LLAMINAR_ROCM_MOE_ROUTER_FP16=1` | Top-k stable, not faster |
| Q8 router gate cache | `LLAMINAR_ROCM_MOE_ROUTER_Q8=1` | Stable in focused cases, not faster due hidden quant/reconstruction overhead |
| K-part router logits | `LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE=1` | Correct, flat/noise |

### MoE Expert FFN

Accepted wins:

- `LLAMINAR_ROCM_MOE_PARALLEL_DOWN_DECODE` default-on unless deterministic: parallelizes down projection across active experts.
- `LLAMINAR_ROCM_MOE_GATEUP_KPART_DECODE` default-on unless deterministic: K-partitions grouped gate/up decode; default `LLAMINAR_ROCM_MOE_GATEUP_KPARTS=8`.
- Descriptor validation is cached at descriptor-table upload time.
- Runtime pointer arrays are cached.
- Runtime top-k ids/weights are consumed directly.

Default-off / neutral:

- Shared expert grouped decode was essentially neutral: opt-in around 58.38 tok/s versus default around 58 tok/s.
- Native-VNNI direct/tuning knobs were added during a pass but did not produce a default win. Keep them guarded unless a future profile proves otherwise.

### GDN

`LLAMINAR_ROCM_GDN_CONCURRENT_DECODE=1` routes GDN decode projections (`qkv`, `z`, `alpha`, `beta`) through the existing ROCm multi-stream fused GEMV path.

Observed result:

- Default decode around 58.30 tok/s.
- Flagged decode around 59.01 tok/s.
- Keep default-off for now; it is a small win and profiling mode can make it look worse.

Next likely target from the last tuning context: `rocm_gdn_recurrent_step_kernel` and GDN recurrence/projection interaction.

### Native-VNNI Decode GEMV

Decode-focused rocprof showed native-VNNI decode GEMV is still a major bucket, especially separate `gemv_native_vnni_reduce_kernel_t` cost after codebook-19 GEMVs.

Ideas that still look worth exploring:

- Fuse partial/reduce for decode shapes where split-K is overkill.
- Add a single-pass no-reduce path for small decode GEMV shapes.
- Specialize the hot codebook-19 decode path if rocprof confirms it dominates current traces.
- Compare direct path versus split-K under no-env benchmarks, not only profiling mode.

## Current Environment Flags

Useful toggles:

```bash
# Expert decode defaults; disable to A/B.
LLAMINAR_ROCM_MOE_PARALLEL_DOWN_DECODE=0
LLAMINAR_ROCM_MOE_GATEUP_KPART_DECODE=0
LLAMINAR_ROCM_MOE_GATEUP_KPARTS=2   # or 4, 8

# Router experiments.
LLAMINAR_ROCM_MOE_ROUTER_WAVE_TOPK=0
LLAMINAR_ROCM_MOE_ROUTER_FP16=1
LLAMINAR_ROCM_MOE_ROUTER_Q8=1
LLAMINAR_ROCM_MOE_ROUTER_KPART_DECODE=1
LLAMINAR_ROCM_MOE_ROUTER_KPARTS=4   # or 2, 8, 16

# GDN experiment.
LLAMINAR_ROCM_GDN_CONCURRENT_DECODE=1

# Deterministic mode should disable non-deterministic/approximate fast paths.
LLAMINAR_DETERMINISTIC=1
```

## Known Build/Runtime Pitfalls

- If decode suddenly drops back near 20 tok/s, check that `Qwen35MoEGraph` still wires `MoERuntimeTable` into both `MoERoutingStage::Params::moe_runtime_table` and `MoEExpertComputeStage::Params::moe_runtime_table`. Losing that wiring disables the fast runtime-table route/expert path.
- If benchmark fails during memory planning with apparently idle GPUs, check device inventory matching. A previous dirty change matched GPU inventory by ordinal only and selected the wrong memory entry; the device type must match too.
- Remove incomplete sidecar/overlay hooks if they include non-existent headers such as overlay CPU fallback runner headers. Do not leave Release unbuildable for a speculative tuning hook.
- Clear stuck processes before large-model runs:

```bash
ps -eo pid,cmd | rg 'llaminar2|rocprof|v2_integration_rocm' || true
rocm-smi --showmeminfo vram --showuse
```

## Recommended Next Implementation Targets

Work in this order unless new profiling contradicts it:

1. **GDN recurrence decode**
   - Target `rocm_gdn_recurrent_step_kernel`.
   - Look for scalar memory traffic, redundant per-head state loads, and launch shape under-occupancy.
   - Benchmark with and without `LLAMINAR_ROCM_GDN_CONCURRENT_DECODE=1`.

2. **Native-VNNI decode GEMV reduce overhead**
   - Target `gemv_native_vnni_reduce_kernel_t` and associated codebook-19 GEMV path.
   - Try single-pass or fused reduce variants for decode shapes.
   - Keep direct-path knobs default-off unless no-env benchmark clearly wins.

3. **Flash decode attention**
   - rocprof still shows flash decode attention in the residual bucket.
   - Confirm whether it is kernel compute, KV cache layout, or launch overhead before changing it.

4. **Prefill parity phase**
   - llama.cpp prefill is around 694 tok/s versus Llaminar around 268 tok/s.
   - Treat this as a separate project after decode reaches parity.
   - Likely targets: native-VNNI prefill kernels, packed layout, and prefill graph/launch overhead.

## Subagent Dispatch Template

Use this pattern for each implementation subagent:

```text
You are working in /workspaces/llaminar. IMPLEMENTATION TASK.

Target: <one kernel family only>.

Hard data:
- Current default benchmark: <prefill/decode>.
- llama.cpp target: ~694 prefill, ~69.3 decode.
- LLAMINAR_PROFILING stage evidence: <stage and ms/token>.
- rocprof kernel evidence: <kernel names, launches, total/avg time>.

Requirements:
1. Preserve existing default behavior unless benchmark proves a win.
2. Add an env guard for experiments.
3. Add focused tests for correctness/parity.
4. Run focused tests, Release build, default benchmark, opt-in benchmark.
5. Return files changed, benchmarks, tests, and default-on/default-off decision.
```

## PyTorch Parity Tests For 35B MoE Correctness

Build integration first:

```bash
cmake --build build_v2_integration --parallel
```

Core 35B MoE single-device PyTorch parity:

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen35MoE_SingleDevice_.*(PrefillParity|DecodeParity)_Qwen35MoE_35B_(CPU|ROCm)_KV_FP16$' \
  --output-on-failure --parallel
```

ROCm snapshot infrastructure for stage-level debugging:

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen35MoE_SingleDevice_.*SnapshotInfrastructure_Qwen35MoE_35B_ROCm_KV_FP16$' \
  --output-on-failure --parallel
```

Local TP RCCL 2xROCm 35B MoE parity, when two ROCm GPUs are available:

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen35MoE_LocalTP_.*(PrefillParity|DecodeParity)_LocalTP_RCCL_2xROCm_35B_MoE$' \
  --output-on-failure --parallel
```

Graph-native hot/cold production-path parity:

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen35MoE_GraphNative_RocmHotCpuCold_.*(PrefillParity|DecodeParity)$' \
  --output-on-failure --parallel
```

Expert overlay parity, if tuning touches overlay placement, GPU hot cache, or CPU cold fallback paths:

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen35MoEExpertOverlay_.*(PrefillParity|DecodeParity)_' \
  --output-on-failure --parallel
```

Hybrid PP/TP named-domain parity, if graph placement or runtime-table wiring changes:

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen35MoE_HybridPPTP_.*(PrefillParityWithGpuExpertCache|DecodeParityWithGpuExpertCache)_NamedDomainPP_LocalTP_ROCm_NodeLocalTP_CPU_35B_MoE$' \
  --output-on-failure --parallel
```

Focused non-PyTorch smoke tests that should be run after kernel changes:

```bash
ctest --test-dir build_v2_integration \
  -R '^(V2_Integration_ROCmMoEKernel|V2_Unit_DecodeExpertHistogram|V2_Unit_MoERuntimeTable|V2_Unit_MoERoutingStage)$' \
  --output-on-failure --parallel
```

Parity CSV/log output is written under:

```text
tests/v2/integration/parity/results/<git-hash>/<test-suite>/
```

Use `prefill_stages.csv`, `decode_stages.csv`, and `test_log.txt` to inspect stage-level cosine, routing overlap, and drop-stage diagnostics.

## Completion Criteria

Decode parity target:

- Llaminar no-env Release decode is within run-to-run noise of llama.cpp `~69.3 tok/s` on the same host/GPU/model.
- PyTorch parity tests above pass for the touched execution path.
- `git diff --check` passes.
- `cmake --build build_v2_release --target llaminar2 --parallel` passes.

Prefill parity target:

- Llaminar prefill approaches llama.cpp `~694 tok/s` on the same host/GPU/model.
- This likely requires a separate prefill-kernel campaign and should not be conflated with decode work.