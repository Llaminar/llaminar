# OpenBLAS vs PyTorch Parity Investigation

## Test Results Summary

**Test**: `ParityFramework.OpenBLASPrefillVsPyTorch`  
**Model**: `Gemini-Distill-Qwen2.5-0.5B-ead-fp32.gguf` (FP32, no quantization)  
**Tokens**: `[1, 2, 3, 4, 5]` (5 tokens)  
**Result**: 145/147 tests failed (98.6% failure rate)

## What Works ✅

1. **EMBEDDING**: Perfect match (`max_abs=0.0`, `rel_l2=0.0`) ✅
2. **ATTENTION_NORM_layer0**: Within tolerance (`max_abs=0.037`, `rel_l2=0.019`) ✅

## What Fails ❌

**First divergence**: `ATTENTION_OUTPUT_layer0`  
- `max_abs=0.337` (tolerance: 0.1)  
- `rel_l2=1.418` (tolerance: 0.05)  
- **Status**: ❌ FAIL (rel_l2 exceeds tolerance by 28x!)

### Error Pattern

Top 5 differences all show the same pattern:
```
[0,842] diff=0.337  expected=0.322  actual=-0.015  (llaminar lower)
[4,842] diff=0.327  expected=0.309  actual=-0.018  (llaminar lower)
[1,842] diff=0.310  expected=0.290  actual=-0.020  (llaminar lower)
[2,842] diff=0.288  expected=0.271  actual=-0.016  (llaminar lower)
[3,842] diff=0.276  expected=0.260  actual=-0.016  (llaminar lower)
```

**Observation**:
- All differences at **column 842** (out of 896 total)
- All show llaminar values **slightly lower** than PyTorch
- Consistent magnitude (~0.015-0.020 delta)
- Systematic across all token positions (0-4)

## Root Cause Analysis

### Stage-by-Stage Breakdown

| Stage | Status | Notes |
|-------|--------|-------|
| EMBEDDING | ✅ PASS | Perfect match |
| ATTENTION_NORM_layer0 | ✅ PASS | RMSNorm works correctly |
| **ATTENTION_OUTPUT_layer0** | ❌ **FAIL** | **First divergence** |
| ATTENTION_RESIDUAL_layer0 | ❌ FAIL | Inherits attention error |
| FFN_NORM_layer0 | ❌ FAIL | Compounds from residual |
| ... all subsequent layers | ❌ FAIL | Error propagates through pipeline |

### Hypothesis: Attention Mechanism Issue

The divergence starts at **ATTENTION_OUTPUT**, which is the output of:
```
Attention Output = softmax(QK^T / √d) × V × W_o
```

Where `W_o` is the output projection matrix.

**Possible causes**:

1. **Output Projection Weight Mismatch**:
   - Weight matrix orientation (row-major vs column-major)
   - Weight loading from GGUF format
   - Transpose flag mismatch

2. **RoPE (Rotary Position Embedding)**:
   - Frequency calculation differences
   - Application order (before/after split)
   - sin/cos precomputation vs on-the-fly

3. **QKV Projection**:
   - Weight layout for multi-head attention
   - Head splitting logic
   - Attention bias handling

4. **Softmax / Attention Scores**:
   - Numerical precision in softmax
   - Scaling factor (√d_head)
   - Masking implementation

### Why Column 842?

The error appearing consistently at **column 842** (out of 896) suggests:
- Possible off-by-one or boundary condition error
- Weight matrix indexing issue
- Head dimension boundary (896 = 14 heads × 64 dims, column 842 = head 13, position 10)

## Comparison with COSMA Results

**COSMA vs PyTorch**: 146/147 failures (99.3%)  
**OpenBLAS vs PyTorch**: 145/147 failures (98.6%)

**Conclusion**: The PyTorch parity issue affects **both backends equally**, proving it's not a COSMA bug but a fundamental issue in llaminar's transformer implementation.

## Detailed Attention Analysis (Layer 0)

### PyTorch Reference Values

Using the enhanced snapshot generator, we now have visibility into every attention sub-stage:

```
EMBEDDING          → (1, 5, 896)   ✅ Perfect match
ATTENTION_NORM     → (1, 5, 896)   ✅ Within tolerance
Q_PROJECTION       → (1, 5, 896)   ❓ Not yet compared
K_PROJECTION       → (1, 5, 128)   ❓ Not yet compared  
V_PROJECTION       → (1, 5, 128)   ❓ Not yet compared
Q_ROPE             → (1, 5, 896)   ❓ Not yet compared
K_ROPE             → (1, 5, 128)   ❓ Not yet compared
ATTENTION_SCORES   → (1, 14, 5, 5) ❓ Not yet compared
ATTENTION_SOFTMAX  → (1, 14, 5, 5) ❓ Not yet compared
ATTENTION_CONTEXT  → (1, 5, 896)   ❓ **KEY STAGE** - before o_proj
ATTENTION_OUTPUT   → (1, 5, 896)   ❌ FIRST FAILURE
```

### Critical Discovery: Attention Context Values

PyTorch attention context (before output projection) at position 0, dimension 842:
```
context[0, 842] = -0.000191
```

After output projection:
```
output[0, 842] = 0.321948
```

**Delta from o_proj**: `+0.322139`

### Llaminar vs PyTorch Error Pattern

From parity test:
```
Position 0, dimension 842:
  PyTorch expected: 0.322
  Llaminar actual:  -0.015
  Difference:       0.337
```

### Analysis

**Key insight**: The error magnitude (0.337) is very close to what the output projection should add (0.322). This suggests:

1. **If llaminar's attention context is correct** (`-0.000191`):
   - llaminar o_proj adds: `-0.015 - (-0.000191) = -0.01481`
   - PyTorch o_proj adds: `+0.322`
   - **Problem**: Output projection weight mismatch (~0.337 error)

2. **If llaminar's attention context is wrong**:
   - Could be earlier in QKV projection, RoPE, or attention scores
   - Need to compare llaminar's intermediate values

### Next Steps

1. ✅ **Generate PyTorch reference with intermediate stages** - DONE
2. ⏭️ **Emit llaminar attention context** - Need to add snapshot emission
3. ⏭️ **Compare attention context** - Determine if problem is before or after o_proj
4. ⏭️ **If context matches**: Investigate output projection weights
5. ⏭️ **If context differs**: Bisect earlier stages (QKV → RoPE → Scores → Softmax)

### 1. Isolate Attention Components

Test each attention subcomponent independently:
- ✅ Embedding (already passes)
- ✅ RMSNorm (already passes)
- ❓ QKV projection
- ❓ RoPE application
- ❓ Attention scores (QK^T)
- ❓ Softmax
- ❓ Value weighting (scores × V)
- ❓ **Output projection (W_o)** ← Prime suspect

### 2. Compare Weight Loading

Check if PyTorch and llaminar load the same weights:
```python
# PyTorch side
print(f"W_o shape: {model.layers[0].self_attn.o_proj.weight.shape}")
print(f"W_o sample: {model.layers[0].self_attn.o_proj.weight[0, 840:845]}")
```

```cpp
// llaminar side
auto w_o = weights.layers[0].attention.o_proj.weight;
LOG_INFO("W_o shape: " << w_o->shape()[0] << "x" << w_o->shape()[1]);
LOG_INFO("W_o sample [0, 840:845]: ...");
```

### 3. Add Intermediate Snapshots

Capture PyTorch snapshots at finer granularity:
- QKV projection outputs
- RoPE-rotated Q/K
- Attention scores (before softmax)
- Attention scores (after softmax)
- Weighted values (before output projection)
- **Output projection result** ← Key comparison point

### 4. Check RoPE Implementation

RoPE is a common source of subtle bugs:
```python
# Check PyTorch RoPE params
cos, sin = model.layers[0].self_attn.rotary_emb(...)
print(f"cos shape: {cos.shape}, sin shape: {sin.shape}")
print(f"cos[0, 0, :5]: {cos[0, 0, :5]}")
```

### 5. Verify Attention Mask

Causal attention mask might differ:
```python
# PyTorch causal mask
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
```

## Current Focus

**Immediate task**: Understand why ATTENTION_OUTPUT_layer0 diverges.

**Priority**: 
1. Add QKV projection snapshots to PyTorch reference
2. Add attention scores snapshots (before/after softmax)
3. Compare weight values directly between PyTorch and llaminar
4. Check RoPE implementation differences

## Success Criteria

For OpenBLAS vs PyTorch to pass:
- ✅ EMBEDDING: Already passing
- ✅ ATTENTION_NORM: Already passing
- ❌ ATTENTION_OUTPUT: Must achieve `rel_l2 < 0.05`
- ❌ All subsequent layers: Will pass once attention is fixed

**Target**: Get ATTENTION_OUTPUT_layer0 from `rel_l2=1.42` down to `rel_l2 < 0.05` (28x improvement needed)

## Conclusion

The OpenBLAS vs PyTorch parity test provides a **clear baseline** for debugging:
- Embedding and normalization work perfectly
- Attention output projection is the divergence point
- Error is systematic and reproducible
- **This is NOT a COSMA issue** - it affects the core llaminar implementation

Once we fix the attention output issue in OpenBLAS path, COSMA will automatically inherit the fix since it uses the same attention kernels.

**Next action**: Add intermediate PyTorch snapshots to isolate which attention subcomponent diverges.
