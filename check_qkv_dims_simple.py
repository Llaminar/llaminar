#!/usr/bin/env python3
"""Check Q/K/V/O projection dimensions."""
import sys
sys.path.insert(0, 'python/reference')

from loaders.gguf_parser import GGUFParser

parser = GGUFParser("models/qwen2.5-0.5b-instruct-q4_0.gguf")
parser._parse_metadata()
parser._parse_tensor_info()

print("Dimensions AFTER Python reversal (what PyTorch loader produces):")
print("=" * 70)
for tensor in parser.tensors:
    if 'blk.0.attn_' in tensor.name and '.weight' in tensor.name:
        print(f"{tensor.name:30s}: {tensor.dimensions}")

print()
print("Llaminar MPIAttentionKernel expects:")
print("=" * 70)
print("wq: shape[0]=d_model=896, shape[1]=total_head_dim=896")
print("wk: shape[0]=d_model=896, shape[1]=n_head_kv*head_dim=128")
print("wv: shape[0]=d_model=896, shape[1]=n_head_kv*head_dim=128")
print("wo: shape[0]=total_head_dim=896, shape[1]=d_model=896")
