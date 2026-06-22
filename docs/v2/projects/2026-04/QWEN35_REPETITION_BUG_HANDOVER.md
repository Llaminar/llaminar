# Qwen3.5 Decode Repetition/Looping Bug — Handover Document

**Date**: April 7, 2026  
**Branch**: `tensor-parallel`  
**Build**: `build_v2_integration` (Integration mode, 679/679 targets)  
**Status**: Root cause NOT yet identified. FusedQKV TP sharding bug is fixed (separate issue).

---

## 1. Symptom

Qwen3.5-4B (`models/Qwen3.5-4B-Q8_0.gguf`, Q8_0) produces short coherent output then degenerates into repetitive loops during decode. This happens on **both single-socket and TP** configurations.

### Llaminar output (broken):
```
./build_v2_integration/llaminar2 -d cpu:0 \
  -m models/Qwen3.5-4B-Q8_0.gguf \
  -p "$(printf '<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n')" \
  -n 80 -t 0

<think>

</think>

Your question is a bit ambiguous.
```
Generates only ~13 tokens before hitting EOS (248046). On longer prompts it degenerates into repetitive phrases.

### llama.cpp output (correct, same model, same prompt):
```
CUDA_VISIBLE_DEVICES="" /tmp/llama.cpp/build/bin/llama-cli \
  -m models/Qwen3.5-4B-Q8_0.gguf \
  -p "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n" \
  -n 80 --temp 0 --no-display-prompt

[Start thinking]
Thinking Process:

1.  **Analyze the Request:** The user is asking a simple factual question...
2.  **Retrieve Knowledge:** Access general knowledge about geography...
3.  **Formulate Answer:** State the answer clearly and concisely.
```
llama.cpp produces a full, coherent thinking chain. **BE CAREFUL**: `llama-cli` blocks on stdin for user input after generating; use `-n 80` to limit and Ctrl+C / `/exit` to exit.

### Key observations:
- **0.8B model**: Works perfectly (both single-socket and TP) → "The capital of France is **Paris**."
- **4B model, single socket (`-d cpu:0`)**: Degenerates
- **4B model, TP (`-d cpu`)**: Also degenerates (after the TP fix)
- **Qwen3.5-4B has 32 layers**: 24 GDN + 8 FA (every 4th: layers 3,7,11,15,19,23,27,31 are FA)
- **Qwen3.5-0.8B has 28 layers**: 21 GDN + 7 FA (same pattern)
- The 0.8B works → the core GDN recurrence logic is likely correct. The bug may be in dimension handling, state management, or something scale-dependent

---

## 2. Architecture Reference

### Qwen3.5 GDN (Gated Delta Network) Layer Pipeline

```
Input (HIDDEN_STATE)
  ↓
FusedResidualNorm (residual_add + RMSNorm → NORMALIZED)
  ↓
GDNProjection (4 parallel GEMMs)
  ├── attn_qkv.weight × NORMALIZED → GDN_QKV [2*key_dim + value_dim]
  ├── attn_gate.weight × NORMALIZED → GDN_Z [value_dim]
  ├── ssm_alpha.weight × NORMALIZED → GDN_ALPHA [n_v_heads]
  └── ssm_beta.weight × NORMALIZED → GDN_BETA [n_v_heads]
  ↓
ShortConv1d + SiLU (in-place on GDN_QKV, uses persistent conv_state)
  ↓
GDNRecurrence (deinterleave+expand Q/K, delta-rule recurrence → ATTN_OUTPUT)
  ├── Q/K: 16 heads → repeat_interleave → 32 heads (×128 dim each)
  ├── V: 32 heads × 128 dim (straight copy)
  ├── Per-head: S = exp(gate)*S + k⊗(beta*(v - S^T@k))   [delta rule]
  └── output = S^T @ q
  ↓
GatedRMSNorm (per-head RMSNorm(output) × SiLU(Z), in-place on ATTN_OUTPUT)
  ↓
Wo GEMM (ssm_out.weight × ATTN_OUTPUT → ATTN_PROJ [d_model=2560])
  ↓
[AllReduce if TP]
  ↓
FFN block (identical to FA layers): gate/up → SwiGLU → down → residual
```

### 4B Model Dimensions

| Parameter | Value |
|-----------|-------|
| `d_model` (hidden_size) | 2560 |
| `n_layers` | 32 (24 GDN + 8 FA) |
| FA `n_heads` | 32 |
| FA `n_kv_heads` | 8 |
| FA `head_dim` | 256 (some GGUF fields say 80, varies by interpretation) |
| GDN `n_k_heads` (group_count) | 16 |
| GDN `n_v_heads` (time_step_rank) | 32 |
| GDN `d_k = d_v` (state_size) | 128 |
| GDN `key_dim` | 2048 (16×128) |
| GDN `value_dim` | 4096 (32×128) |
| GDN `qkv_dim` | 8192 (2×2048+4096) |
| `d_ff` | 9216 |
| `vocab_size` | 248320 |

### 0.8B Model Dimensions (works correctly)

| Parameter | Value |
|-----------|-------|
| `d_model` | 1536 |
| `n_layers` | 28 (21 GDN + 7 FA) |
| FA `n_heads` | 16 |
| FA `n_kv_heads` | 4 |
| GDN `n_k_heads` | 8 |
| GDN `n_v_heads` | 16 |
| GDN `d_k = d_v` | 128 |
| GDN `key_dim` | 1024 (8×128) |
| GDN `value_dim` | 2048 (16×128) |
| GDN `qkv_dim` | 4096 (2×1024+2048) |

---

## 3. Prior Investigation Summary

### What was checked and found correct:
- **GDN components individually** (gate, beta sigmoid, conv1d SiLU, conv state management) — all verified against PyTorch reference
- **repeat_interleave** Q/K expansion (16→32 heads) — index arithmetic correct
- **GatedRMSNorm** per-head normalization — correct (norm_dim=128, 32 groups)
- **FusedResidualNorm** residual path — correct, same path for GDN and FA
- **GDN Projection** 4-way GEMM — no aliased buffers, straightforward
- **A_log sign convention** — GGUF stores `-exp(A_log)` (negative), so decay is correct
- **FFN** — identical for GDN and FA layers, not GDN-specific

### What was tried:
1. **Per-layer decode comparison** (prior session): L0 close (max_diff=0.035), L1 already diverging (max_diff=0.836), grows to 20+ by L30
2. **Stage dumps** for layer 0 and layer 1 — confirmed outputs exist but differ from PyTorch
3. **PyTorch 4B reference**: First token matches (248068 = `<think>`), second token differs (PyTorch: 198 `\n`, ours: 271 `\n\n`) with logit gap only ~0.5. Small early-decode divergence snowballs.

### What was NOT checked:
- **Actual numerical comparison of GDN recurrence state** between llaminar and PyTorch
- **Whether `chunk_forward()` (prefill) produces correct state** — the prefill state initializes decode, any error propagates
- **Whether 0.8B and 4B differ in GDN kernel codepath** (AVX-512 specialization for d_v=128 exists)
- **Whether the `ssm_a` / `ssm_dt.bias` values are loaded correctly** from GGUF
- **Whether the softplus / sigmoid implementations are numerically accurate enough**
- **L2 normalization of Q/K** in GDN recurrence — is it enabled? Is it model-specific?

---

## 4. Key Files

### Model / Graph
| File | Purpose |
|------|---------|
| `src/v2/models/qwen35/Qwen35Graph.cpp` | Graph builder; GDN layer pipeline, buffer wiring, state management |
| `src/v2/models/qwen35/Qwen35Graph.h` | State vectors (`conv_states_`, `recurrence_states_`), `resetState()` |
| `src/v2/models/qwen35/Qwen35GraphConfigBuilder.cpp` | GDN config from GGUF metadata |
| `src/v2/models/qwen35/Qwen35Schema.h` | Weight sharding declarations, stage pipeline specs |

### Stages
| File | Purpose |
|------|---------|
| `src/v2/execution/compute_stages/stages/GDNProjectionStage.cpp` | 4-way GEMM (QKV, Z, alpha, beta) |
| `src/v2/execution/compute_stages/stages/ShortConv1dStage.cpp` | Conv1d + SiLU, conv state management |
| `src/v2/execution/compute_stages/stages/GDNRecurrenceStage.cpp` | QKV deinterleave, repeat_interleave, kernel dispatch |
| `src/v2/execution/compute_stages/stages/GatedRMSNormStage.cpp` | Per-head RMSNorm × SiLU(Z) gate |

### Kernels
| File | Purpose |
|------|---------|
| `src/v2/kernels/cpu/gdn/CPUGatedDeltaNet.cpp` | Core recurrence: `recurrent_step()` (decode), `chunk_forward()` (prefill) |
| `src/v2/kernels/cpu/gdn/CPUShortConvolution.cpp` | Conv1d forward, state shift+append |

### Tests
| File | Purpose |
|------|---------|
| `tests/v2/unit/stages/Test__GDNDynamicParamsRegression.cpp` | 6 regression tests for `updateDynamicParams` |

### PyTorch Reference
| File | Purpose |
|------|---------|
| `python/reference/models/qwen35_gdn.py` | PyTorch GDN reference implementation |
| `python/reference/tests/test_qwen35.py` | PyTorch unit tests for GDN components |

---

## 5. Debugging Approach Suggestions

### Priority 1: Compare prefill output against PyTorch
The prefill sets up the initial recurrence state and conv state for decode. If prefill output diverges, all subsequent decode tokens are wrong. Use stage dumps:

```bash
LLAMINAR_STAGE_DUMP_ENABLED=1 \
LLAMINAR_STAGE_DUMP_NAMES=gdn_recurrence,gated_norm \
LLAMINAR_STAGE_DUMP_LAYERS=0,1 \
LLAMINAR_STAGE_DUMP_ITERATION=0 \
./build_v2_integration/llaminar2 -d cpu:0 \
  -m models/Qwen3.5-4B-Q8_0.gguf \
  -p "$(printf '<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n')" \
  -n 1 -t 0
```

Then compare with PyTorch `python/reference/models/qwen35_gdn.py` on same GGUF weights.

### Priority 2: Check L2 normalization of Q/K
The GDN paper uses L2-normalized Q and K. In `CPUGatedDeltaNet.cpp`, there's L2 normalization logic gated by a config flag. Check:
- Is it enabled for Qwen3.5-4B?
- Does the 0.8B model use it too?
- Does PyTorch reference use it?

### Priority 3: Check recurrence state evolution
Dump recurrence state after each decode step for layer 0 and check:
- Is the state decaying? (exp(gate) < 1 per step)
- Are values staying bounded?
- Compare magnitude with PyTorch reference

### Priority 4: Logit comparison at divergence point
The first token matches, the second diverges slightly. Use:
```bash
LLAMINAR_STAGE_OUTPUT_PRINT=1 \
LLAMINAR_STAGE_OUTPUT_PRINT_STAGES=layer0_gdn_recurrence,layer1_gdn_recurrence \
./build_v2_integration/llaminar2 -d cpu:0 -m models/Qwen3.5-4B-Q8_0.gguf \
  -p "$(printf '<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n')" \
  -n 5 -t 0
```

### Priority 5: Check if scale-dependent
The 0.8B works, 4B doesn't. Key scale differences:
- `d_model`: 1536 vs 2560
- `n_k_heads`: 8 vs 16
- `n_v_heads`: 16 vs 32
- Hidden dim affects Q scaling (`1/√d_k`), state matrix size, and numerical range

---

## 6. Build & Run Commands

```bash
# Build (Integration mode)
cmake --build build_v2_integration --parallel

# Single-socket (no TP) — preferred for debugging
./build_v2_integration/llaminar2 -d cpu:0 \
  -m models/Qwen3.5-4B-Q8_0.gguf \
  -p "$(printf '<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n')" \
  -n 50 -t 0

# Stage dumps for debugging
LLAMINAR_STAGE_DUMP_ENABLED=1 \
LLAMINAR_STAGE_DUMP_NAMES=gdn_recurrence \
LLAMINAR_STAGE_DUMP_LAYERS=0 \
./build_v2_integration/llaminar2 -d cpu:0 -m models/Qwen3.5-4B-Q8_0.gguf \
  -p "$(printf '<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n')" \
  -n 5 -t 0

# Stage output print (lightweight)
LLAMINAR_STAGE_OUTPUT_PRINT=1 \
LLAMINAR_STAGE_OUTPUT_PRINT_STAGES=gdn_recurrence \
LLAMINAR_LOG_LEVEL=INFO \
./build_v2_integration/llaminar2 -d cpu:0 -m models/Qwen3.5-4B-Q8_0.gguf \
  -p "$(printf '<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n')" \
  -n 5 -t 0

# llama.cpp reference (CPU only, blocks on stdin, use Ctrl+C)
CUDA_VISIBLE_DEVICES="" timeout 120s /tmp/llama.cpp/build/bin/llama-cli \
  -m models/Qwen3.5-4B-Q8_0.gguf \
  -p "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n" \
  -n 80 --temp 0 --no-display-prompt -ngl 0
```

**IMPORTANT**: Do NOT use `--no-mpi-bootstrap` — it breaks NUMA binding.

---

## 7. Uncommitted Changes

Files modified on the `tensor-parallel` branch (not yet committed):

| File | Change | Related To |
|------|--------|------------|
| `src/v2/loaders/WeightManager.h` | Added `setGDNDimensions()` + 4 member vars | TP fix (done) |
| `src/v2/loaders/WeightManager.cpp` | GDN FusedQKV sub-block slicing | TP fix (done) |
| `src/v2/execution/factory/InferenceRunnerFactory.cpp` | Call `setGDNDimensions()` | TP fix (done) |
| `src/v2/execution/compute_stages/stages/FusedResidualNormStage.cpp` | TEMPORARY `LLAMINAR_LAYER_TRACE` debug code | Debug trace (harmless) |
| `src/v2/execution/compute_stages/stages/GatedRMSNormStage.cpp` | Cyclic gamma indexing | Harmless change |
| `src/v2/execution/compute_stages/stages/GDNRecurrenceStage.h` | `updateDynamicParams()` override | Regression fix (done) |
| `src/v2/execution/compute_stages/stages/ShortConv1dStage.h` | `updateDynamicParams()` override | Regression fix (done) |
| `tests/v2/unit/stages/Test__GDNDynamicParamsRegression.cpp` | NEW — 6 regression tests | Regression tests |
| `tests/v2/CMakeLists.txt` | Test registration | Regression tests |
