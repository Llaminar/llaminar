# PCIeBAR LOCAL TP Parity Test Debugging Handover

**Date**: January 26, 2026  
**Status**: Prefill PASSES, Decode FAILS  
**Test**: `V2_Integration_Parity_Qwen2_LocalTP_PCIeBAR_vs_PyTorch` (#244)

---

## Overview

This document captures the current state of debugging the heterogeneous LOCAL TP (CUDA + ROCm via PCIeBAR) parity test against PyTorch reference. The test validates that Llaminar's tensor-parallel inference produces results matching PyTorch when splitting work across different GPU vendors.

---

## Test Details

### Test File
```
tests/v2/integration/parity/qwen2/Test__Qwen2_LocalTP_PCIeBAR_vs_PyTorch.cpp
```

### Model
- **File**: `models/qwen2.5-0.5b-instruct-q4_0.gguf`
- **Architecture**: Qwen2.5-0.5B-Instruct
- **Parameters**: n_heads=14, n_kv_heads=2, head_dim=64, vocab=151936, 24 layers

### Hardware Configuration
- **Device 0**: CUDA (e.g., RTX 3090)
- **Device 1**: ROCm (e.g., MI50)
- **Backend**: PCIeBAR (direct GPU↔GPU memory mapping via BAR)
- **Parallelism**: 2-way LOCAL Tensor Parallelism

### Build Directory
```
/workspaces/llaminar/build_v2_integration
```

---

## Running the Test

### Basic Run
```bash
ctest --test-dir /workspaces/llaminar/build_v2_integration \
  -R "V2_Integration_Parity_Qwen2_LocalTP_PCIeBAR_vs_PyTorch" \
  --output-on-failure -V
```

### Run Specific Sub-Tests
```bash
# Run only prefill parity
./build_v2_integration/tests/v2/v2_integration_qwen2_localtp_pciebar_vs_pytorch \
  --gtest_filter="*PrefillParity*"

# Run only decode parity
./build_v2_integration/tests/v2/v2_integration_qwen2_localtp_pciebar_vs_pytorch \
  --gtest_filter="*DecodeParity*"
```

### CRITICAL: Tee Output to File

**Always capture output to a file** - the test produces extensive logging that's essential for debugging but scrolls past terminal buffers. Re-running the test is slow (~30-60s), so save yourself time:

```bash
# Run with tee to capture output
./build_v2_integration/tests/v2/v2_integration_qwen2_localtp_pciebar_vs_pytorch \
  --gtest_filter="*DecodeParity*" 2>&1 | tee /tmp/decode_parity.log

# Then grep for specific patterns
grep -i "cosine" /tmp/decode_parity.log
grep -i "logits" /tmp/decode_parity.log
grep -i "device\|cuda\|rocm" /tmp/decode_parity.log
grep -E "row\[|max_logit|vocab" /tmp/decode_parity.log
```

### Debug Logging
```bash
LLAMINAR_LOG_LEVEL=DEBUG ./build_v2_integration/tests/v2/v2_integration_qwen2_localtp_pciebar_vs_pytorch \
  --gtest_filter="*DecodeParity*" 2>&1 | tee /tmp/decode_debug.log
```

### Stage Output Print (Tensor Values)
```bash
LLAMINAR_STAGE_OUTPUT_PRINT=1 \
LLAMINAR_STAGE_OUTPUT_PRINT_STAGES=lm_head,gather \
./build_v2_integration/tests/v2/v2_integration_qwen2_localtp_pciebar_vs_pytorch \
  --gtest_filter="*DecodeParity*" 2>&1 | tee /tmp/decode_stages.log
```

---

## Current Status

### Prefill Phase: ✅ PASSES

All 24 transformer layers pass parity with excellent cosine similarity (> 0.96):

```
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                    CUDA vs PyTorch LAYER-BY-LAYER PARITY                                 ║
╠═══════════╦═══════════════╦═══════════════╦════════════════════════════════════════╦══════╣
║   Layer   ║   Avg Cosine  ║   Min Cosine  ║            Worst Stage                 ║Status║
╠═══════════╬═══════════════╬═══════════════╬════════════════════════════════════════╬══════╣
║ EMBEDDING ║      0.999912 ║      0.999912 ║                      -                 ║  ✓  ║
║   Layer 0 ║      0.998234 ║      0.995123 ║              FFN_RESIDUAL              ║  ✓  ║
...
║  LM_HEAD  ║      0.978381 ║      0.978381 ║                      -                 ║  ✓  ║
╚═══════════╩═══════════════╩═══════════════╩════════════════════════════════════════╩══════╝
```

### Decode Phase: ❌ FAILS

Decode produces logits that **vary across steps** (coherence is working) but **don't match PyTorch**:

- Cosine similarity: 0.055 to 0.358 (threshold: 0.80)
- Token predictions differ from reference
- Both devices show varying logit values (not stale/repeated)

---

## Recent Fixes Applied

### 1. Coherence Bug Fix: `use_mapped_memory` Propagation

**Problem**: `gatherLogits()` was reading stale host data because `use_mapped_memory` wasn't being propagated to per-device InferenceRunnerConfig.

**Symptom**: ROCm device returned **identical logits** across all decode steps.

**Fix**: Added `config.use_mapped_memory = true;` in test setup:
```cpp
// File: tests/v2/integration/parity/qwen2/Test__Qwen2_LocalTP_PCIeBAR_vs_PyTorch.cpp
// Line ~352
config.use_mapped_memory = true;  // CRITICAL: Enable mapped memory for correct logits_local coherence
```

**Result**: Both devices now show **varying** logit values across decode steps. Coherence is working.

### 2. gatherLogits() Rewrite

**Problem**: `gatherLogits()` was not correctly handling 2D logits tensors for column-parallel LM head gathering.

**Fix**: Rewrote to perform row-by-row column concatenation:
```cpp
// For each row (sequence position):
//   Gather vocab_local columns from each device
//   Concatenate into combined_logits[row, 0:vocab_total]
```

### 3. K/V Weight Slicing Bug

**Problem**: `calculateProportionalColumnSlice()` was using total Q heads instead of KV heads for K/V weight slicing.

**Fix**: Now uses `totalKVHeads()` for K/V weights.

---

## Hypotheses for Decode Failure

### Hypothesis 1: KV Cache Position/Offset Handling

During decode, the KV cache grows incrementally. If position indices are handled differently between CUDA and ROCm runners, or if the cache isn't being updated consistently, attention outputs will diverge.

**Investigation**:
- Check `kv_cache_pos` values on both devices
- Verify KV cache contents match after each decode step
- Look for off-by-one errors in position calculation

### Hypothesis 2: Numerical Divergence Accumulation

Cross-vendor (CUDA vs ROCm) compute uses different FP32 implementations. Small differences accumulate over 24 layers, and decode's autoregressive nature compounds errors.

**Investigation**:
- Run layer-by-layer comparison during decode (not just final logits)
- Check if divergence increases with layer depth
- Compare intermediate attention scores

### Hypothesis 3: PCIeBAR AllReduce Precision

The PCIeBAR allreduce uses CUDA reduction kernels. If there's any precision loss or incorrect summation during the reduction phase, it would affect decode more than prefill (which has more redundancy).

**Investigation**:
- Add checksums before/after allreduce
- Compare allreduce results to CPU reference
- Use `LLAMINAR_MPI_VERIFY_CHECKSUMS=1`

### Hypothesis 4: Weight Sharding Mismatch During Decode

Decode uses different weight access patterns than prefill. If any weights are being loaded incorrectly or cached stale values, decode would fail while prefill passes.

**Investigation**:
- Verify weight tensor shapes on each device during decode
- Check if weight caching is interfering
- Log weight accesses during decode steps

### Hypothesis 5: Attention Mask Handling

Decode attention uses causal masking with growing sequence length. If the mask isn't being updated correctly across devices, attention scores will be wrong.

**Investigation**:
- Check attention mask generation during decode
- Verify mask dimensions match sequence length
- Compare mask values between CUDA and ROCm

---

## Debugging Checklist

### Quick Diagnostics
- [ ] Run with `LLAMINAR_LOG_LEVEL=DEBUG` and tee to file
- [ ] Check both devices show varying logits (coherence working)
- [ ] Compare top-5 token predictions vs PyTorch
- [ ] Run prefill-only to confirm it still passes

### Detailed Investigation
- [ ] Enable stage output print for LM head: `LLAMINAR_STAGE_OUTPUT_PRINT_STAGES=lm_head`
- [ ] Compare `logits_local` shapes on each device
- [ ] Verify `gatherLogits()` output dimensions
- [ ] Check KV cache positions during decode

### Advanced Debugging
- [ ] Use stage dump framework to capture decode iteration tensors
- [ ] Compare layer-by-layer activations during decode (not just final logits)
- [ ] Run single-device CUDA baseline vs PyTorch (isolate TP from vendor issues)

---

## Useful Commands

### Compare Logits Between Runs
```bash
# Capture logits from test
grep "logits_local" /tmp/decode_parity.log > /tmp/logits.txt

# Check for NaN/Inf
grep -i "nan\|inf" /tmp/decode_parity.log

# Find max logit values
grep "max_logit" /tmp/decode_parity.log
```

### Stage Dump for Replay Testing
```bash
LLAMINAR_STAGE_DUMP_ENABLED=1 \
LLAMINAR_STAGE_DUMP_NAMES=lm_head \
LLAMINAR_STAGE_DUMP_ITERATION=0 \
./build_v2_integration/tests/v2/v2_integration_qwen2_localtp_pciebar_vs_pytorch \
  --gtest_filter="*DecodeParity*" 2>&1 | tee /tmp/decode_dump.log

# Dumps go to /tmp/llaminar_stage_dumps/
ls -la /tmp/llaminar_stage_dumps/
```

### Run PyTorch Reference Standalone
```bash
cd /workspaces/llaminar
python3 scripts/pytorch_parity_reference.py \
  --model models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --prompt "Hello, world!" \
  --decode-steps 5
```

---

## Files to Review

| File | Purpose |
|------|---------|
| `tests/v2/integration/parity/qwen2/Test__Qwen2_LocalTP_PCIeBAR_vs_PyTorch.cpp` | Main test file |
| `src/v2/execution/RankOrchestrator.cpp` | `gatherLogits()` implementation |
| `src/v2/execution/DeviceGraphOrchestrator.cpp` | Per-device runner, KV cache handling |
| `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` | Decode loop, position management |
| `src/v2/collective/backends/PCIeBARBackend.cpp` | AllReduce implementation |
| `src/v2/loaders/WeightManager.cpp` | Weight sharding for LOCAL TP |

---

## Next Steps

1. **Isolate the problem**: Run single-device CUDA vs PyTorch decode to see if divergence is TP-related or general decode issue

2. **Layer-by-layer decode comparison**: Add decode-phase snapshots to compare intermediate activations

3. **KV cache inspection**: Add logging to verify KV cache state is consistent across devices during decode

4. **Numerical tolerance review**: Heterogeneous TP may inherently have higher divergence than homogeneous - consider relaxing thresholds for cross-vendor configs

---

## Contact

For questions about this work, refer to:
- Project plan: `docs/v2/projects/2026-01/WEIGHTMANAGER_LOCAL_PARALLELISM_PROJECT_PLAN.md`
- Architecture: `.github/instructions/llaminar-architecture-v2.instructions.md`
- Parity test framework: `tests/v2/integration/parity/README.md`
