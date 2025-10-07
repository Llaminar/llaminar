# Next Steps: OpenBLAS vs PyTorch Parity Debugging

## Current Status 🎯

 **COSMA work**: Complete and ready to merge  
 **Enhanced PyTorch snapshots**: Now capturing 339 stages (including QKV, RoPE, scores, context)  
 **Error identified**: `ATTENTION_OUTPUT_layer0` diverges by 0.337 at dimension 842  
 **Hypothesis formed**: Output projection weight issue (0.337 ≈ 0.322 o_proj contribution)  

 **Next**: Emit llaminar's `ATTENTION_CONTEXT` to compare with PyTorch

---

## Immediate Action Plan

### Step 1: Add ATTENTION_CONTEXT Emission in Llaminar

**File**: `src/kernels/MPIAttentionKernel.cpp`  
**Location**: Around line 1680, inside `computeLocalOutputProjection()`

**What to add**:
```cpp
// After computing local_attended_output (attention context)
// but BEFORE applying output projection
if (debugEnv().parity.capture && layer_index_ == 0 && getRank() == 0) {
    // Emit ATTENTION_CONTEXT for parity comparison
    std::string stage_name = "ATTENTION_CONTEXT_layer0";
    PipelineSnapshotManager::captureStage(
        stage_name,
        local_attended_output->data(),
        {seq_len, local_heads * head_dim},
        -1  // Use -1 for internal attention stages
    );
}
```

### Step 2: Run Parity Test

```bash
cd /workspaces/llaminar
mpirun -np 2 ./build/test_parity_framework \
    --gtest_filter=ParityFramework.OpenBLASPrefillVsPyTorch 2>&1 | \
    tee parity_context_check.log
```

### Step 3: Analyze Results

**Scenario A**: ATTENTION_CONTEXT matches PyTorch
- ✅ Confirms attention computation is correct
- 🔍 Problem is in output projection (o_proj)
- **Next action**: Investigate weight loading for `blk.0.attn_output.weight`

**Scenario B**: ATTENTION_CONTEXT differs from PyTorch
- 🔍 Problem is earlier in attention pipeline
- **Next action**: Compare QKV projections, RoPE, attention scores

---

## If ATTENTION_CONTEXT Matches (Scenario A)

### Check Output Projection Weights

1. **Verify weight loading**:
   ```cpp
   // In model loader, check:
   LOG_INFO("o_proj weight shape: " << w_o->shape()[0] << "x" << w_o->shape()[1]);
   LOG_INFO("o_proj sample [0,842]: " << w_o->data()[842]);
   ```

2. **Compare with PyTorch**:
   ```python
   # In Python
   import torch
   from reference import ModelRegistry
   model = ModelRegistry.create("qwen", "models/Gemini-Distill-Qwen2.5-0.5B-ead-fp32.gguf")
   model.load_model()
   w_o = model.hf_model.model.layers[0].self_attn.o_proj.weight
   print(f"PyTorch o_proj shape: {w_o.shape}")
   print(f"PyTorch o_proj [:, 842]: {w_o[:, 842][:10]}")  # First 10 elements
   ```

3. **Check orientation**:
   - PyTorch: `[d_model, total_head_dim]` (e.g., `[896, 896]`)
   - GGUF might store transposed
   - llaminar might need transpose flag

### Check Head Concatenation Order

GQA (Grouped Query Attention) with 14 query heads, 2 KV heads:
- Each KV group serves 7 query heads
- Head concatenation order must match PyTorch
- Dimension 842 falls in head 13 (second-to-last head)

---

## If ATTENTION_CONTEXT Differs (Scenario B)

### Bisect Earlier Stages

Add emissions for:
1. `Q_PROJECTION`, `K_PROJECTION`, `V_PROJECTION`
2. `Q_ROPE`, `K_ROPE`
3. `ATTENTION_SCORES` (before softmax)
4. `ATTENTION_SOFTMAX`

Compare each with PyTorch to find first divergence.

---

## Debug Environment Variables

Useful for investigation:

```bash
# Enable attention validation
export LLAMINAR_ATTN_OUTPUT_VALIDATE=1

# Enable parity capture
export LLAMINAR_PARITY_CAPTURE=1

# Disable COSMA to isolate OpenBLAS path
export ADAPTIVE_DISABLE_COSMA=1

# Verbose logging
export LLAMINAR_LOG_LEVEL=DEBUG
```

---

## Expected Outcome

After completing these steps, we will know:

1. **Exact divergence point**: Context vs o_proj vs earlier
2. **Root cause**: Weight loading, orientation, or computation
3. **Fix strategy**: Targeted fix for identified issue

**Estimated time**: 1-2 hours for diagnosis, 1-2 hours for fix and validation

---

## Success Criteria

 EMBEDDING passes (already achieved)  
 ATTENTION_NORM passes (already achieved)  
 ATTENTION_CONTEXT passes  
 ATTENTION_OUTPUT passes  
 All 147 stages pass with rel_l2 < 0.05  

Once OpenBLAS parity is achieved:
- Verify COSMA matches OpenBLAS baseline
- Both should then match PyTorch reference
- Ready to merge all fixes

---

## Files Modified This Session

### Enhanced ✅
- `python/reference/generate_test_snapshots.py` - Added 8 attention sub-stages per layer

### Created ✅
- `debug_attention_context.py` - PyTorch value analysis tool
- `OPENBLAS_PYTORCH_PARITY_ANALYSIS.md` - Comprehensive investigation doc
- `ATTENTION_DEBUG_PROGRESS.md` - Progress summary
- `NEXT_STEPS.md` - This file

### Ready to Modify ⏭️
- `src/kernels/MPIAttentionKernel.cpp` - Add ATTENTION_CONTEXT emission

---

**Current focus**: Implement Step 1 (add ATTENTION_CONTEXT emission) and run Step 2 (parity test).
