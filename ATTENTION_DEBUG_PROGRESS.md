# PyTorch Parity Investigation - Progress Summary

## What We've Accomplished (Option A then B)

### Option A: Examined Attention Implementation ✅

**Files Analyzed**:
- `src/kernels/MPIAttentionKernel.cpp` (1823 lines)
  - Output projection at lines 700-850 (post-projection gather)
  - Output projection at lines 1650-1823 (validation and TP partitioning)
  
**Key Findings**:
- Attention uses `computeLocalOutputProjection()` for o_proj
- Multiple execution paths: single GEMM, TP partitioning, or simulation mode
- Validation mode available via `LLAMINAR_ATTN_OUTPUT_VALIDATE`
- Backend selection: prefill vs inference paths

### Option B: Modified PyTorch Reference Generator ✅

**Files Modified**:
- `python/reference/generate_test_snapshots.py`

**Changes Made**:
1. Added intermediate attention stage captures:
   - `Q_PROJECTION` - Query projection (before RoPE)
   - `K_PROJECTION` - Key projection (before RoPE)
   - `V_PROJECTION` - Value projection
   - `Q_ROPE` - Query after RoPE application
   - `K_ROPE` - Key after RoPE application
   - `ATTENTION_SCORES` - Q @ K^T before softmax
   - `ATTENTION_SOFTMAX` - Attention weights after softmax
   - `ATTENTION_CONTEXT` - **Critical**: Weighted sum (attention @ V) before o_proj

2. Fixed compatibility issues:
   - Imported `apply_rotary_pos_emb` and `repeat_kv` from transformers
   - Used config attributes for head counts
   - Properly applied RoPE using model-level rotary_emb

**Results**:
- ✅ Successfully generates 339 pipeline stages (was 147)
- ✅ All 24 layers now have 8 attention sub-stages each
- ✅ Enables precise bisection of where divergence starts

## Critical Discovery

### PyTorch Attention Context Analysis

Using the new snapshots, we analyzed layer 0, position 0, dimension 842 (where error was detected):

```
Stage                Value           Notes
─────────────────────────────────────────────────────────────
ATTENTION_CONTEXT    -0.000191      Before output projection
ATTENTION_OUTPUT      0.321948      After output projection
Delta (o_proj adds)  +0.322139      What o_proj contributes
```

### Llaminar Error Pattern

From original parity test:
```
Position 0, dimension 842:
  PyTorch expected:  0.322
  Llaminar actual:  -0.015
  Difference:        0.337
```

### Hypothesis

The error magnitude (0.337) ≈ what o_proj should add (0.322). Two scenarios:

**Scenario 1**: llaminar attention context is correct (`-0.000191`)
- llaminar o_proj incorrectly adds `-0.015`
- **Problem**: Output projection weight error
- **Action**: Check o_proj weight loading/orientation

**Scenario 2**: llaminar attention context is wrong
- Could be earlier: QKV projection, RoPE, or attention computation
- **Action**: Compare intermediate stages (QKV → RoPE → Scores → Context)

## What We Need Next

### Immediate Action Required

**Emit llaminar's ATTENTION_CONTEXT for comparison**:

1. Add snapshot emission in `MPIAttentionKernel::computeLocalOutputProjection()`
2. Capture `local_attended_output` (the context before o_proj)
3. Compare with PyTorch's `ATTENTION_CONTEXT_0.npy`

### Test Strategy

```bash
# 1. Modify MPIAttentionKernel to emit ATTENTION_CONTEXT snapshot
# 2. Run parity test
mpirun -np 2 ./build/test_parity_framework --gtest_filter=ParityFramework.OpenBLASPrefillVsPyTorch

# 3. Compare:
#    - If ATTENTION_CONTEXT matches → problem is in o_proj weights
#    - If ATTENTION_CONTEXT differs → bisect earlier stages
```

### Diagnosis Tree

```
ATTENTION_OUTPUT diverges (0.337 error)
│
├─ Compare ATTENTION_CONTEXT
│  │
│  ├─ Matches PyTorch
│  │  └─> Problem: Output projection (o_proj)
│  │      Actions:
│  │      - Check o_proj weight loading
│  │      - Check weight orientation/transpose
│  │      - Verify head concatenation order
│  │
│  └─ Differs from PyTorch
│     └─> Problem: Earlier in attention
│         │
│         ├─ Compare Q_PROJECTION, K_PROJECTION, V_PROJECTION
│         │  └─> If wrong: QKV weight loading issue
│         │
│         ├─ Compare Q_ROPE, K_ROPE
│         │  └─> If wrong: RoPE implementation issue
│         │
│         ├─ Compare ATTENTION_SCORES
│         │  └─> If wrong: Q @ K^T computation issue
│         │
│         └─ Compare ATTENTION_SOFTMAX
│            └─> If wrong: Softmax or masking issue
```

## Files Ready for Next Phase

### Python Reference (Enhanced) ✅
- `python/reference/generate_test_snapshots.py` - Now captures 339 stages
- `debug_attention_context.py` - Analysis script for PyTorch values

### C++ Kernel (Needs Modification)
- `src/kernels/MPIAttentionKernel.cpp` - Add ATTENTION_CONTEXT emission
- Specifically around line ~1680 before `computeLocalOutputProjection()`

### Test Infrastructure ✅
- `tests/test_parity_framework.cpp` - Already supports arbitrary stage names
- PyTorch snapshots in `/tmp/pytorch_snapshots_openblas/`

## Summary

**Status**: We've successfully enhanced the diagnostic capability. We can now:
1. ✅ Generate fine-grained PyTorch reference (339 stages)
2. ✅ Understand the error pattern (dimension 842, 0.337 error)
3. ✅ Have a clear hypothesis (o_proj weight issue vs earlier stage issue)
4. ⏭️ Need to emit llaminar's intermediate values for comparison

**Next immediate step**: Add `ATTENTION_CONTEXT` snapshot emission in llaminar to determine if the problem is in:
- Output projection weights (most likely based on error magnitude)
- Earlier attention computation (less likely but possible)

**Confidence level**: High - we have the tools and data to precisely bisect the issue.
