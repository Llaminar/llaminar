# Handover: Qwen35 MoE ROCm Decode Parity RoPE Lead

Date: 2026-05-20
Workspace: `/workspaces/llaminar`
Branch/HEAD: `feat/qwen35-moe` at `ed903cf2`
Primary target: `DecodeParity_Qwen35MoE_35B_ROCm_KV_FP16`
CTest target: `V2_Integration_Parity_Qwen35MoE_SingleDevice_Qwen35MoE_Qwen35MoESingleDeviceParityTest_DecodeParity_Qwen35MoE_35B_ROCm_KV_FP16`
Model: `/opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`

This continuation supersedes older status in:

```text
docs/v2/HANDOVER_QWEN35_MOE_ROCM_DECODE_PARITY_2026-05-20.md
```

The earlier doc contains useful background on mixed native-VNNI codebooks and snapshot-table decode guards, but the current debugging lead has moved to Qwen3.5 full-attention RoPE.

## Current Status

The focused ROCm decode target still fails the strict early-layer gate after current diagnostic instrumentation:

```text
Expected passed_early_layers >= 5, actual: 3 vs 5
At least 5 of the first 6 layers should pass decode parity at step 3 (cosine >= 0.98000001907348633)
```

However, token-level decode parity remains good. Latest `decode_steps.csv` for ROCm:

```text
step 0 cosine=0.998363 token_match=true passed=true
step 1 cosine=0.998215 token_match=true passed=true
step 2 cosine=0.994591 token_match=true passed=true
step 3 cosine=0.995518 token_match=true passed=true
step 4 cosine=0.997980 token_match=true passed=true
```

The important finding: FP16 KV cache is not proven as the cause. A prior CPU FP32-KV A/B in this session still showed the same full-attention drop, and the current instrumented CPU CSV shows the same Q_ROPE failure pattern as ROCm.

## What Was Fixed Before This Point

### GDN short-conv in-place prefill scratch

The original layer-0 decode issue was a real in-place prefill hazard in GPU short convolution. Prefill runs over time in parallel, and when input/output alias, one timestep can clobber values another timestep still needs.

Files changed:

```text
src/v2/tensors/TensorKernels.h
src/v2/kernels/KernelFactory.cpp
src/v2/kernels/cuda/gdn/CUDAShortConvolution.h
src/v2/kernels/rocm/gdn/ROCmShortConvolution.h
```

Key behavior now:

- `ITensorShortConvolution::allocateGPUScratch(int)` was added.
- `KernelFactory::createHybridKVCache()` preallocates per-layer in-place prefill scratch using `max_seq_len * qkv_dim` during GDN kernel setup.
- CUDA/ROCm `forward()` no longer grows scratch in the hot path. It fails loudly if in-place prefill scratch was not prepared.
- Backend headers use existing backend-local `*_gpu_memcpy_async()` wrappers; do not include CUDA and HIP runtimes in these public C++ headers because mixed CUDA/ROCm builds collide on vector types.

After this fix, early GDN stages became essentially perfect:

```text
layer0 QKV_PROJECTION        cosine=0.999997
layer0 GDN_CONV1D_OUTPUT     cosine=1.000000
layer0 GDN_DELTA_RULE_OUTPUT cosine=1.000000
layer0 GDN_NORM_GATE_OUTPUT  cosine=1.000000
```

So the current remaining issue is not the short-conv/GDN path.

## Diagnostic Instrumentation Added

The next debugging pass added finer snapshot boundaries for Qwen3.5 full attention.

### C++ snapshot dimensions and names

Files changed:

```text
src/v2/snapshots/SnapshotCapture.cpp
src/v2/execution/compute_stages/stages/QGateSplitStage.cpp
src/v2/execution/compute_stages/stages/AttentionOutputGateStage.cpp
```

Important details:

- `QGateSplitStage::buildDumpInfoImpl()` now reports logical rows and `n_heads * head_dim`, not arena capacity.
- `AttentionOutputGateStage::buildDumpInfoImpl()` now reports logical rows and tensor cols, not arena capacity.
- `SnapshotCapture` maps:
  - `_q_gate_split` -> `FA_GATE`
  - `_attn_output_gate` -> `ATTENTION_CONTEXT_GATED`
  - `_attention` -> `ATTENTION_CONTEXT`

These fixes made the new diagnostic rows survive parity comparison.

### Python reference snapshots

Files changed:

```text
python/reference/pipeline_stages.py
python/reference/qwen35.py
python/reference/qwen35_moe.py
```

Added reference stages:

```text
FA_GATE
ATTENTION_CONTEXT_GATED
Q_NORM
K_NORM
Q_ROPE
K_ROPE
```

Reference hooks now capture:

- `FA_GATE` from the q projection split.
- `ATTENTION_CONTEXT_GATED` from `o_proj` pre-hook input.
- Raw `ATTENTION_CONTEXT` reconstructed as `gated_context / sigmoid(fa_gate)`.
- `Q_NORM` and `K_NORM` from HF `q_norm` / `k_norm` outputs.
- `Q_ROPE` and `K_ROPE` by wrapping HF `apply_rotary_pos_emb` while a full-attention layer is active.

Snapshot regeneration must be clean to avoid stale `.npy` files:

```bash
rm -rf pytorch_qwen35_moe_snapshots
python python/reference/generate_qwen35_moe_pipeline_snapshots.py \
  --model /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  --decode-steps 5 \
  --output pytorch_qwen35_moe_snapshots
```

The latest clean regeneration captured `733` snapshots per prefill/decode pass and `4398` snapshots total.

## Current Decisive Metrics

Latest ROCm decode CSV:

```text
tests/v2/integration/parity/results/ed903cf2/Qwen35MoE_Qwen35MoESingleDeviceParityTest_DecodeParity_Qwen35MoE_35B_ROCm_KV_FP16/decode_stages.csv
```

Decode step 3, layer 3:

```text
Q_PROJECTION              cosine=0.999989 rel_l2=0.004955 max_abs=0.0962567
K_PROJECTION              cosine=0.999839 rel_l2=0.018044 max_abs=0.0454114
V_PROJECTION              cosine=0.999785 rel_l2=0.020771 max_abs=0.049677
FA_GATE                   cosine=0.999994 rel_l2=0.003769 max_abs=0.0618658
Q_NORM                    cosine=0.999842 rel_l2=0.017798 max_abs=0.153905
K_NORM                    cosine=0.999853 rel_l2=0.017167 max_abs=0.0818316
Q_ROPE                    cosine=0.936985 rel_l2=0.355169 max_abs=7.46706
K_ROPE                    cosine=0.967511 rel_l2=0.254875 max_abs=3.62645
ATTENTION_CONTEXT          cosine=0.942799 rel_l2=0.336659 max_abs=1.73968
ATTENTION_CONTEXT_GATED    cosine=0.932859 rel_l2=0.362430 max_abs=0.0761816
ATTENTION_OUTPUT           cosine=0.919352 rel_l2=0.394082 max_abs=0.0960874
```

Decode step 3, layer 7:

```text
Q_PROJECTION              cosine=0.999051 rel_l2=0.048028 max_abs=1.1274
K_PROJECTION              cosine=0.992817 rel_l2=0.119735 max_abs=0.266044
V_PROJECTION              cosine=0.982778 rel_l2=0.187339 max_abs=0.637945
FA_GATE                   cosine=0.999534 rel_l2=0.036817 max_abs=0.572288
Q_NORM                    cosine=0.988421 rel_l2=0.152579 max_abs=1.1488
K_NORM                    cosine=0.993446 rel_l2=0.114593 max_abs=0.542843
Q_ROPE                    cosine=0.918404 rel_l2=0.404883 max_abs=7.45374
K_ROPE                    cosine=0.972329 rel_l2=0.235438 max_abs=3.77531
ATTENTION_CONTEXT          cosine=0.979706 rel_l2=0.200948 max_abs=1.69751
ATTENTION_CONTEXT_GATED    cosine=0.968802 rel_l2=0.251416 max_abs=0.157096
ATTENTION_OUTPUT           cosine=0.926555 rel_l2=0.391590 max_abs=0.192169
```

Current CPU decode CSV shows the same RoPE pattern, so this is probably not ROCm-specific:

```text
layer3 Q_ROPE cosine=0.936681, K_ROPE cosine=0.967059
layer7 Q_ROPE cosine=0.918509, K_ROPE cosine=0.971833
```

Interpretation:

- Q/K/V projections are good before RoPE.
- Q/K per-head RMSNorm is good before RoPE.
- `FA_GATE` is good, so the q/gate deinterleave is not the issue.
- The first large full-attention divergence is `Q_ROPE` / `K_ROPE`.
- Attention context and Wo projection are downstream amplifiers.

## Commands Already Run

Focused target build:

```bash
cmake --build build_v2_integration --config Integration \
  --target v2_integration_parity_qwen35moe_single_device \
  --parallel
```

Python syntax check:

```bash
python -m py_compile \
  python/reference/pipeline_stages.py \
  python/reference/qwen35.py \
  python/reference/qwen35_moe.py
```

Snapshot regeneration:

```bash
rm -rf pytorch_qwen35_moe_snapshots
python python/reference/generate_qwen35_moe_pipeline_snapshots.py \
  --model /opt/llaminar-models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  --decode-steps 5 \
  --output pytorch_qwen35_moe_snapshots
```

Focused ROCm decode parity:

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen35MoE_SingleDevice_Qwen35MoE_Qwen35MoESingleDeviceParityTest_DecodeParity_Qwen35MoE_35B_ROCm_KV_FP16$' \
  --output-on-failure
```

Focused CPU decode parity for backend-independent comparison:

```bash
ctest --test-dir build_v2_integration \
  -R '^V2_Integration_Parity_Qwen35MoE_SingleDevice_Qwen35MoE_Qwen35MoESingleDeviceParityTest_DecodeParity_Qwen35MoE_35B_CPU_KV_FP16$' \
  --output-on-failure
```

Useful CSV extraction:

```bash
csv='tests/v2/integration/parity/results/ed903cf2/Qwen35MoE_Qwen35MoESingleDeviceParityTest_DecodeParity_Qwen35MoE_35B_ROCm_KV_FP16/decode_stages.csv'
awk -F',' 'NR==1 {print "step,layer,stage,cosine,drop,rel_l2,max_abs"; next}
  $2==3 && ($3==3 || $3==7) &&
  ($4=="Q_PROJECTION" || $4=="K_PROJECTION" || $4=="V_PROJECTION" ||
   $4=="Q_NORM" || $4=="K_NORM" || $4=="Q_ROPE" || $4=="K_ROPE" ||
   $4=="FA_GATE" || $4=="ATTENTION_CONTEXT" ||
   $4=="ATTENTION_CONTEXT_GATED" || $4=="ATTENTION_OUTPUT")
  {print $2","$3","$4","$5","$6","$7","$8}' "$csv"
```

Logs saved during this pass:

```text
/tmp/qwen35moe_rocm_decode_scratch_fix.log
/tmp/qwen35moe_rocm_decode_gate_shape_fix.log
/tmp/qwen35moe_rocm_decode_raw_context.log
/tmp/qwen35moe_rocm_decode_qknorm_context.log
/tmp/qwen35moe_rocm_decode_rope_context.log
/tmp/qwen35moe_cpu_decode_rope_context.log
/tmp/qwen35moe_regen_rope_context.log
```

## Recommended Next Steps

1. Inspect C++ RoPE convention against HuggingFace Qwen3.5.

   Start with:

   ```text
   src/v2/execution/compute_stages/stages/RoPEStage.cpp
   src/v2/kernels/cpu/ops/CPURoPEKernelT.h
   src/v2/kernels/rocm/ops/ROCmRoPEKernelT.h
   src/v2/kernels/cuda/ops/CUDARoPEKernelT.h
   ```

   Current suspicion is not generic numerical precision. The max_abs values around `7.4` at `Q_ROPE` strongly suggest a convention/layout/positioning issue.

2. Verify partial RoPE semantics for Qwen3.5.

   Qwen3.5 uses partial rotary dimensions. C++ computes:

   ```text
   rotary_dim = int(head_dim * partial_rotary_factor)
   effective_head_dim = rotary_dim when partial_rotary_factor < 1
   kernel->apply_tensor(... effective_head_dim ..., pos_offset, rotary_dim)
   ```

   Compare this against HF `apply_rotary_pos_emb(query_states, key_states, cos, sin)` and the model config's `rope_parameters` / `partial_rotary_factor`. Check whether C++ is rotating the correct elements and whether its rotate-half pairing matches HF.

3. Build a tiny direct RoPE comparison.

   Useful approach:

   - Dump or load one layer's Q_NORM/K_NORM for decode step 3.
   - Apply the C++ CPU RoPE kernel in isolation with the same `pos_offset`, `theta_base`, `head_dim`, and `rotary_dim`.
   - Compare against Python `decode_step3_layer3_Q_ROPE.npy` and `K_ROPE.npy`.
   - Print first head first 64 dims before/after RoPE. A sign/pairing/offset mismatch should be obvious.

4. Check position offset for decode.

   HF derives decode `position_ids` from `past_key_values.get_seq_length()`. C++ passes `params_.pos_offset` into RoPE and also materializes `position_ids_cache_` when needed. Verify decode step 3 uses the same absolute position as HF for the prompt length plus decode history.

5. Keep attention/MoE downstream for now.

   `ATTENTION_CONTEXT`, `ATTENTION_CONTEXT_GATED`, `ATTENTION_OUTPUT`, and later MoE routing drift are downstream of the RoPE mismatch. Do not chase expert math before RoPE is resolved.

## Worktree Notes

The worktree is intentionally dirty and includes unrelated staged WIP. Do not reset, checkout, stash, or revert unrelated files unless explicitly asked.

Relevant files touched in this pass:

```text
python/reference/pipeline_stages.py
python/reference/qwen35.py
python/reference/qwen35_moe.py
src/v2/snapshots/SnapshotCapture.cpp
src/v2/execution/compute_stages/stages/QGateSplitStage.cpp
src/v2/execution/compute_stages/stages/AttentionOutputGateStage.cpp
src/v2/tensors/TensorKernels.h
src/v2/kernels/KernelFactory.cpp
src/v2/kernels/cuda/gdn/CUDAShortConvolution.h
src/v2/kernels/rocm/gdn/ROCmShortConvolution.h
```

`git diff --stat` for the most relevant current diffs before this handover showed roughly:

```text
python/reference/pipeline_stages.py                | 12 +++
python/reference/qwen35.py                         | 90 ++++++++++++++++++++++
python/reference/qwen35_moe.py                     | 90 ++++++++++++++++++++++
AttentionOutputGateStage.cpp                       | 11 ++-
QGateSplitStage.cpp                                |  9 ++-
KernelFactory.cpp                                  | 10 +++
CUDAShortConvolution.h                             | 60 ++++++++++++++-
ROCmShortConvolution.h                             | 60 ++++++++++++++-
TensorKernels.h                                    |  7 ++
```

Editor diagnostics are noisy for Python reference files because of existing type-checker complaints around HuggingFace constructors/hooks. `python -m py_compile` succeeds for the edited Python modules.

## Key Takeaway

Do not continue explaining the full-attention drop as FP16 KV cache unless new evidence appears. The current best lead is a Qwen3.5 RoPE convention/position/layout mismatch shared by CPU and ROCm: projections and Q/K norms match, then Q_ROPE/K_ROPE diverge sharply.