#!/usr/bin/env python3
"""
Compare output projection weights between PyTorch and GGUF to identify the mismatch.
Now that we know ATTENTION_CONTEXT matches, the problem must be in o_proj weights.
"""

import numpy as np
import torch
from transformers import AutoModel
import struct
import sys

print("="*80)
print("OUTPUT PROJECTION WEIGHT ANALYSIS")
print("="*80)

# Load PyTorch model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"\nLoading PyTorch model: {model_name}")
hf_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)

# Get layer 0 output projection weight
layer0_o_proj = hf_model.layers[0].self_attn.o_proj.weight.detach().cpu().numpy()
print(f"✓ PyTorch o_proj weight shape: {layer0_o_proj.shape}")  # Should be [896, 896]

# Load GGUF file and find o_proj tensor
gguf_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf"
print(f"\n  Searching for o_proj in GGUF: {gguf_path}")

def read_gguf_tensors(path):
    """Simple GGUF parser to extract tensor metadata"""
    tensors = {}
    with open(path, 'rb') as f:
        # GGUF header
        magic = f.read(4)
        if magic != b'GGUF':
            print("❌ Not a valid GGUF file")
            return tensors
        
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
        
        print(f"   GGUF version: {version}, tensors: {tensor_count}, metadata: {metadata_kv_count}")
        
        # Skip metadata section (simplified)
        # This is a rough parser - we just want tensor names
        
        # For now, just search for the tensor name string in the file
        f.seek(0)
        data = f.read()
        
        # Search for "blk.0.attn_output.weight" pattern
        search_names = [
            b'blk.0.attn_output.weight',
            b'blk.0.attn_o.weight',
            b'model.layers.0.self_attn.o_proj.weight'
        ]
        
        for name in search_names:
            if name in data:
                pos = data.find(name)
                print(f"   ✓ Found tensor name at offset {pos}: {name.decode('utf-8', errors='ignore')}")
                
    return tensors

# Try to find the tensor
read_gguf_tensors(gguf_path)

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print(f"""
We've confirmed:
✅ ATTENTION_CONTEXT matches PyTorch perfectly (rel_l2=2.6e-06)
❌ ATTENTION_OUTPUT diverges significantly (rel_l2=1.418)

This means the problem is in o_proj weight loading or application.

PyTorch o_proj weight shape: {layer0_o_proj.shape}

Possible issues:
1. Weight orientation/transpose mismatch
2. Quantization/dequantization error  
3. Head concatenation order difference
4. Weight not loaded correctly from GGUF

Next steps:
1. Examine how o_proj is loaded in ModelLoader
2. Check tensor orientation in MPIAttentionKernel::computeLocalOutputProjection
3. Verify head ordering matches PyTorch's expectation
""")

# Sample some values from PyTorch o_proj for reference
print("\nPyTorch o_proj weight samples:")
print(f"  [0, 0:5]:   {layer0_o_proj[0, :5]}")
print(f"  [842, 0:5]: {layer0_o_proj[842, :5]}")
print(f"  min/max:    {layer0_o_proj.min():.6f} / {layer0_o_proj.max():.6f}")
print(f"  mean/std:   {layer0_o_proj.mean():.6f} / {layer0_o_proj.std():.6f}")
