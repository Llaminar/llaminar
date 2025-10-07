#!/usr/bin/env python3
"""
Check the actual tensor shape stored in the GGUF file for attn_output.weight layer 0.
Compare with PyTorch expectations.
"""

import sys
sys.path.insert(0, 'llama.cpp')

from gguf import GGUFReader
import numpy as np

gguf_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf"

print("="*80)
print("GGUF TENSOR ORIENTATION CHECK")
print("="*80)

reader = GGUFReader(gguf_path)

# Find layer 0 attention output projection
target_name = "blk.0.attn_output.weight"

print(f"\nSearching for: {target_name}")

found = False
for tensor in reader.tensors:
    if tensor.name == target_name:
        found = True
        print(f"\n✓ Found: {tensor.name}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Data type: {tensor.tensor_type}")
        print(f"  N elements: {tensor.n_elements}")
        
        # In GGUF, tensors are stored with dimensions in a specific order
        # Let's check if it matches PyTorch (896, 896) or transposed
        if len(tensor.shape) == 2:
            rows, cols = tensor.shape
            print(f"\n  Interpretation: [{rows}, {cols}]")
            if rows == 896 and cols == 896:
                print("  ✓ Matches PyTorch shape [d_model, d_model] = [896, 896]")
            else:
                print(f"  ⚠ Different from PyTorch [896, 896]")
                
        # Check what the expected usage is in llaminar
        print(f"\n  Llaminar expects:")
        print(f"    local_attended_output: [seq_len, local_head_dim]")
        print(f"    local_wo: [local_head_dim, d_model]  ← This tensor")
        print(f"    Result: [seq_len, d_model]")
        print(f"\n  For layer 0 with 14 heads, 64 dim each:")
        print(f"    local_head_dim = 14 * 64 = 896")
        print(f"    d_model = 896")
        print(f"    So local_wo should be [896, 896]")
        
        if rows == 896 and cols == 896:
            print(f"\n  ✓ Shape is compatible")
        else:
            print(f"\n  ❌ Shape mismatch!")
        break

if not found:
    print(f"\n❌ Tensor '{target_name}' not found in GGUF file")
    print("\nAvailable attention output tensors:")
    for tensor in reader.tensors:
        if 'attn_output' in tensor.name or 'attn_o' in tensor.name:
            print(f"  - {tensor.name}: {tensor.shape}")

print("\n" + "="*80)
print("PYTORCH REFERENCE")
print("="*80)

from transformers import AutoModel
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"\nLoading: {model_name}")
hf_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)

layer0_o_proj = hf_model.layers[0].self_attn.o_proj.weight
print(f"PyTorch o_proj.weight shape: {layer0_o_proj.shape}")  # [896, 896]
print(f"PyTorch stores as: [out_features, in_features] = [d_model, d_model]")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

print("""
In PyTorch nn.Linear:
  - weight shape: [out_features, in_features]  
  - Forward: y = x @ weight.T + bias
  - For o_proj: out=896, in=896 → weight is [896, 896]
  - Input x: [batch, seq_len, 896]
  - Output y: [batch, seq_len, 896]
  - Operation: y = x @ weight.T

In llaminar's adaptive_matmul:
  - C = A @ B (row-major)
  - A: [seq_len, 896] (local_attended_output)
  - B: [896, 896] (local_wo)
  - C: [seq_len, 896] (result)

CRITICAL QUESTION:
  Does GGUF store the weight as-is [896, 896] or transposed?
  If GGUF stores PyTorch's weight.T (already transposed), then it's [896, 896]
  and should work directly with adaptive_matmul.
  
  If GGUF stores the original weight [896, 896], then llaminar needs to
  transpose it OR adjust the matmul operation.
""")
