#!/usr/bin/env python3
"""Quick script to check Q/K/V/O projection dimensions in GGUF and PyTorch."""

import struct

# Quick GGUF dimension reader
model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf"

with open(model_path, 'rb') as f:
    # Skip magic and version
    f.read(8)
    
    # Read metadata
    n_meta = struct.unpack('<Q', f.read(8))[0]
    for _ in range(n_meta):
        # Read key length and key
        key_len = struct.unpack('<Q', f.read(8))[0]
        key = f.read(key_len).decode('utf-8')
        # Read type and skip value
        val_type = struct.unpack('<I', f.read(4))[0]
        if val_type == 4:  # String
            str_len = struct.unpack('<Q', f.read(8))[0]
            f.read(str_len)
        elif val_type == 5:  # Array
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            if arr_type == 4:  # String array
                for _ in range(arr_len):
                    str_len = struct.unpack('<Q', f.read(8))[0]
                    f.read(str_len)
            elif arr_type in [6, 7, 8, 9]:  # Integer types
                f.read(arr_len * (4 if arr_type <= 7 else 8))
        elif val_type in [6, 7]:  # Int32/UInt32
            f.read(4)
        elif val_type in [8, 9]:  # Int64/UInt64
            f.read(8)
        elif val_type == 10:  # Float32
            f.read(4)
        elif val_type == 11:  # Bool
            f.read(1)
    
    # Read tensors
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    print("RAW GGUF dimensions (BEFORE reversal):")
    print("=" * 60)
    
    for _ in range(n_tensors):
        # Read tensor name
        name_len = struct.unpack('<Q', f.read(8))[0]
        name = f.read(name_len).decode('utf-8')
        
        # Read dimensions
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        
        # Read type and offset
        f.read(4)  # type
        f.read(8)  # offset
        
        # Only print attention weights from first layer
        if 'blk.0.attn_' in name and '.weight' in name:
            print(f"{name:30s}: raw={dims} → reversed={list(reversed(dims))}")

print()
print("PyTorch expects (from weight loading code):")
print("=" * 60)
print("wq: [d_model, total_head_dim] = [896, 896]")
print("wk: [d_model, n_head_kv * head_dim] = [896, 128]")
print("wv: [d_model, n_head_kv * head_dim] = [896, 128]")
print("wo: [total_head_dim, d_model] = [896, 896]")
print()
print("Llaminar expects (from MPIAttentionKernel validation):")
print("=" * 60)
print("wq: shape[0]=d_model=896, shape[1]=total_head_dim=896")
print("wk: shape[0]=d_model=896, shape[1]=n_head_kv*head_dim=128")
print("wv: shape[0]=d_model=896, shape[1]=n_head_kv*head_dim=128")
print("wo: shape[0]=total_head_dim=896, shape[1]=d_model=896")
