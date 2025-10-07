#!/usr/bin/env python3
"""
Compare raw weight values between PyTorch and what llaminar should be loading.
This will definitively show if transpose is needed.
"""

import numpy as np
import sys
sys.path.insert(0, 'python/reference')

from loaders.gguf_loader import GGUFLoader
from transformers import AutoModel
import torch

print("="*80)
print("WEIGHT LAYOUT COMPARISON: PyTorch vs GGUF")
print("="*80)

# Load PyTorch model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"\nLoading PyTorch model: {model_name}")
hf_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
pytorch_weight = hf_model.layers[0].self_attn.o_proj.weight.detach().cpu().numpy()

print(f"PyTorch o_proj.weight shape: {pytorch_weight.shape}")  # [896, 896]
print(f"PyTorch stores as [out_features, in_features]")

# Load GGUF model
gguf_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf"
print(f"\nLoading GGUF model: {gguf_path}")
loader = GGUFLoader(gguf_path, verbose=False)
_, state_dict = loader.load()

# Get the o_proj weight (mapped from blk.0.attn_output.weight)
gguf_weight = state_dict['model.layers.0.self_attn.o_proj.weight'].numpy()

print(f"GGUF o_proj weight shape: {gguf_weight.shape}")  # Should also be [896, 896]

# Compare shapes
print("\n" + "-"*80)
print("SHAPE COMPARISON")
print("-"*80)
print(f"PyTorch: {pytorch_weight.shape}")
print(f"GGUF:    {gguf_weight.shape}")

if pytorch_weight.shape != gguf_weight.shape:
    print("❌ SHAPES DON'T MATCH!")
else:
    print("✓ Shapes match")

# Compare corner values
print("\n" + "-"*80)
print("CORNER VALUES (First few elements)")
print("-"*80)

print("\nPyTorch weight[0:3, 0:5] (first 3 rows, first 5 cols):")
print(pytorch_weight[:3, :5])

print("\nGGUF weight[0:3, 0:5] (first 3 rows, first 5 cols):")
print(gguf_weight[:3, :5])

# Check if they're the same or transposed
same_error = np.abs(pytorch_weight - gguf_weight).max()
transposed_error = np.abs(pytorch_weight - gguf_weight.T).max()

print("\n" + "-"*80)
print("DIRECT COMPARISON")
print("-"*80)
print(f"Max abs error (PyTorch vs GGUF):           {same_error:.6f}")
print(f"Max abs error (PyTorch vs GGUF.T):         {transposed_error:.6f}")

# Check a specific position we care about
print("\n" + "-"*80)
print("SPECIFIC VALUE CHECK (position [842, 0])")
print("-"*80)
print(f"PyTorch weight[842, 0]:     {pytorch_weight[842, 0]:.6f}")
print(f"GGUF weight[842, 0]:        {gguf_weight[842, 0]:.6f}")
print(f"GGUF weight.T[842, 0]:      {gguf_weight.T[842, 0]:.6f}")
print(f"(which is GGUF[0, 842]:     {gguf_weight[0, 842]:.6f})")

# Determine the relationship
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if same_error < 0.01:
    print("✓ GGUF and PyTorch weights are THE SAME (no transpose needed)")
    print("  → Llaminar should use these weights DIRECTLY")
    print("  → BUT: Llaminar's matmul might need transpose_B flag")
elif transposed_error < 0.01:
    print("✓ GGUF weights are TRANSPOSED relative to PyTorch")
    print("  → GGUF stores as [in_features, out_features]")
    print("  → PyTorch expects [out_features, in_features]")
    print("  → Llaminar needs to transpose when loading OR during matmul")
else:
    print(f"⚠️  Weights differ significantly (quantization error ~{same_error:.4f})")
    print("  → This is expected for Q4_0 quantization")
    print("  → Need to check orientation separately")
    
    # For quantized weights, check if general pattern matches
    # Compare row 0 across both
    row0_same = np.corrcoef(pytorch_weight[0, :], gguf_weight[0, :])[0, 1]
    row0_trans = np.corrcoef(pytorch_weight[0, :], gguf_weight[:, 0])[0, 1]
    
    print(f"\n  Correlation check:")
    print(f"    PyTorch row 0 vs GGUF row 0:    {row0_same:.4f}")
    print(f"    PyTorch row 0 vs GGUF col 0:    {row0_trans:.4f}")
    
    if row0_same > row0_trans:
        print("  → Same orientation (no transpose)")
    else:
        print("  → Transposed orientation (transpose needed)")

print("\n" + "="*80)
print("LLAMINAR IMPLICATION")
print("="*80)

print("""
Llaminar's scalar validation code expects:
  B[k][j] accessed as B[k * d_model + j]
  This means B is stored in row-major as [K, N]

PyTorch nn.Linear uses:
  weight[out][in] accessed as weight[out * in_features + in]
  This means weight is stored as [out_features, in_features] = [N, K]

Therefore:
  Llaminar B needs to be PyTorch weight TRANSPOSED
  
Current state:
  - PyTorch loader reverses GGUF dimensions to get [N, K]
  - Llaminar does NOT reverse (keeps GGUF raw [?, ?])
  
The fix:
  If GGUF == PyTorch (both [N, K]):
    → Llaminar needs to transpose to [K, N] during load OR matmul
  If GGUF == [K, N] already:
    → Llaminar should use directly (but then WHY does Python reverse?)
""")
