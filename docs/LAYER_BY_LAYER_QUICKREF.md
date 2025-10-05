# Layer-by-Layer Parity Testing - Quick Reference

## One-Command Investigation

```bash
# Capture all 123 PyTorch layer outputs and save to NPZ
python3 python/reference/capture_pytorch_layers.py \
  -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --tokens 1639 266 285 17 10 17 30 \
  -o pytorch_layer_captures.npz
```

**Output**: 6.47 MB NPZ file with 123 snapshots from embedding → logits

## What Gets Captured (123 Snapshots)

### Global Stages (3)
- `embeddings` - Token embeddings (1×7×896)
- `final_norm_out` - After final RMSNorm (1×7×896)
- `logits` - Final output (1×7×151936)

### Per-Layer Stages (24 layers × 5 checkpoints = 120)
For each layer 0-23:
- `layer_N_input_norm_out` - After input RMSNorm
- `layer_N_attn_out` - After self-attention  
- `layer_N_post_attn_norm_out` - After post-attention RMSNorm
- `layer_N_ffn_out` - After FFN/MLP
- `layer_N_out` - Final layer output

## When to Use

✅ **Use this when**:
- Numerical divergence between Llaminar and PyTorch
- Need to find first diverging layer
- Debugging transformer layer implementations
- Validating fixes

❌ **Don't use when**:
- Simple single-layer test is sufficient
- No divergence (tests pass)
- Performance benchmarking (adds overhead)

## Comparison Workflow

### With Llaminar Snapshots (Future)
```bash
# 1. Capture Llaminar (when instrumented)
LLAMINAR_PARITY_CAPTURE=1 ./build/llaminar ... > llaminar_snapshots.npz

# 2. Capture PyTorch + compare
python3 python/reference/capture_pytorch_layers.py \
  --llaminar-captures llaminar_snapshots.npz \
  -o pytorch_layer_captures.npz

# 3. Output shows first diverging layer
# FIRST DIVERGING LAYER: layer_0_attn_out
```

### Manual Comparison (Current)
```bash
# 1. Capture PyTorch
python3 python/reference/capture_pytorch_layers.py -o pytorch.npz

# 2. Load in Python for comparison
import numpy as np
pt_data = np.load('pytorch.npz')
ll_data = np.load('llaminar.npz')  # when available

# Compare specific layer
pt_layer0_attn = pt_data['layer_0_attn_out']
ll_layer0_attn = ll_data['layer_0_attn_out']
diff = np.abs(pt_layer0_attn - ll_layer0_attn).mean()
print(f"Layer 0 attention diff: {diff}")
```

## Interpreting Results

### Tolerance Guidelines
| Mean Abs Diff | Verdict | Action |
|---------------|---------|--------|
| < 0.001 | ✅ Perfect match | Continue |
| 0.001 - 0.01 | ✅ Acceptable (FP precision) | Continue |
| 0.01 - 0.1 | ⚠ Warning (investigate) | Check implementation |
| ≥ 0.1 | ❌ Divergence (bug) | Fix required |

### Example Output
```
✓ embeddings:
   Max abs diff: 0.000000
   Mean abs diff: 0.000000
   Cosine similarity: 1.000000

✓ layer_0_input_norm_out:
   Max abs diff: 0.000012
   Mean abs diff: 0.000003
   Cosine similarity: 0.999999

❌ layer_0_attn_out:
   Max abs diff: 2.451234
   Mean abs diff: 0.345678  ← EXCEEDS TOLERANCE!
   Cosine similarity: 0.923456

   🔍 DIVERGENCE DETECTED!
   Worst element: idx=234
     PyTorch: 1.234567
     Llaminar: -1.216567
     Diff: 2.451234

================================================================================
FIRST DIVERGING LAYER: layer_0_attn_out
================================================================================
```

**Interpretation**: 
- Embeddings perfect ✅ → Weight loading correct
- Input norm perfect ✅ → RMSNorm implementation correct
- Attention diverges ❌ → **Problem is in attention mechanism!**

## Binary Search Strategy

When you have 24 layers and want to find the first divergence:

```
Check layer 12:
  ✓ Matches → Problem in layers 13-23
  ❌ Diverges → Problem in layers 0-12

Check layer 6 (if diverged at 12):
  ✓ Matches → Problem in layers 7-12
  ❌ Diverges → Problem in layers 0-6

Continue until you isolate the exact layer...
```

The script automatically reports the **first** diverging layer.

## Next Steps After Finding Divergence

If `layer_N_attn_out` diverges:

1. **Check sub-components**:
   - `layer_N_input_norm_out` matches → RMSNorm OK
   - `layer_N_attn_out` diverges → Attention is the problem
   - `layer_N_post_attn_norm_out` → Propagates or new issue?

2. **Deep-dive into attention**:
   - Add hooks for Q, K, V projections
   - Capture before/after RoPE
   - Capture attention scores
   - Capture after softmax
   - Find exact operation causing divergence

3. **Common Issues**:
   - RoPE frequency calculation ⭐⭐⭐⭐⭐
   - Attention mask logic ⭐⭐⭐⭐
   - Softmax numerical stability ⭐⭐⭐
   - Head dimension mismatch ⭐⭐

## Files to Check

**Python Tool**:
- `python/reference/capture_pytorch_layers.py` - Main capture script

**C++ Framework**:
- `tests/parity_test_framework.{h,cpp}` - Comparison infrastructure
- `tests/test_layer_by_layer_parity.cpp` - Test integration

**Documentation**:
- `docs/LAYER_BY_LAYER_PARITY_GUIDE.md` - Full guide
- `docs/LAYER_BY_LAYER_FRAMEWORK_SUMMARY.md` - Implementation summary
- `docs/LAYER_BY_LAYER_QUICKREF.md` - This file

## Current Status

✅ **Python capture**: FULLY FUNCTIONAL  
✅ **Python comparison**: FULLY FUNCTIONAL  
⏳ **C++ NPZ loading**: Skeleton implemented  
⏳ **Pipeline instrumentation**: Pending  

**You can use the Python tool right now!**

## Tips

1. **Save captures**: NPZ files can be reused, shared, compared offline
2. **Use verbose mode**: See all layers, not just failures
3. **Start broad**: Check embeddings → layer 12 → binary search
4. **Isolate**: Once you find diverging layer, add granular hooks
5. **Document**: Save comparison output for reference

## Emergency Debug

If the tool doesn't work:

```bash
# Check dependencies
python3 -c "import torch, transformers, numpy; print('OK')"

# Check model exists
ls -lh models/qwen2.5-0.5b-instruct-q4_0.gguf

# Run with minimal tokens
python3 python/reference/capture_pytorch_layers.py \
  -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --tokens 1639
```

## Example: Current Investigation

```bash
# We know from prior investigation:
# ✓ Embeddings: PERFECT (0.000000 diff)
# ✓ LM head weights: PERFECT (0.000000 diff)
# ❌ Final logits: DIVERGENT (3.83 mean abs diff)
#
# Hypothesis: Transformer layers diverge
# Most likely: Layer 0 attention (RoPE or masking issue)

# Run this to confirm:
python3 python/reference/capture_pytorch_layers.py \
  -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --tokens 1639 266 285 17 10 17 30

# Expected result:
# ✓ embeddings: PERFECT
# ✓ layer_0_input_norm_out: PERFECT (or very close)
# ❌ layer_0_attn_out: DIVERGENT ← ROOT CAUSE HERE!
```

## Success Criteria

After fixing the root cause, re-run and verify:

```bash
# All 123 snapshots should match within tolerance
# ✓ embeddings: mean_abs_diff < 0.001
# ✓ layer_0_input_norm_out: mean_abs_diff < 0.001
# ✓ layer_0_attn_out: mean_abs_diff < 0.001  ← FIXED!
# ✓ ... (all layers match)
# ✓ logits: mean_abs_diff < 0.001
#
# Final validation: Top-5 predictions match PyTorch
```

---

**Ready to find that bug?** 🔍

```bash
python3 python/reference/capture_pytorch_layers.py \
  -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
  --tokens 1639 266 285 17 10 17 30
```
