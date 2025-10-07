#!/usr/bin/env python3
"""
Check raw GGUF dimension order for key tensors to understand the dimension reversal issue.
"""

import sys
sys.path.insert(0, 'llama.cpp')

from gguf import GGUFReader

gguf_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf"

print("="*80)
print("GGUF RAW DIMENSION ORDER CHECK")
print("="*80)

reader = GGUFReader(gguf_path)

# Check key tensors
key_tensors = [
    "token_embd.weight",           # Embedding
    "output.weight",               # LM head  
    "blk.0.attn_q.weight",        # Attention Q projection
    "blk.0.attn_k.weight",        # Attention K projection
    "blk.0.attn_v.weight",        # Attention V projection
    "blk.0.attn_output.weight",   # Attention O projection ← THE KEY ONE
    "blk.0.ffn_gate.weight",      # FFN gate
    "blk.0.ffn_up.weight",        # FFN up
    "blk.0.ffn_down.weight",      # FFN down
]

print("\nRAW dimensions as stored in GGUF (BEFORE any reversal):\n")

for tensor in reader.tensors:
    if tensor.name in key_tensors:
        print(f"{tensor.name:30s}: {tensor.shape}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

print("""
GGUF Spec dimension order:
- GGUF stores dimensions in a specific order
- Reader.tensors[i].shape is what the GGUF file contains DIRECTLY

PyTorch expectation:
- Linear layer weights: [out_features, in_features]
- For o_proj: [d_model, d_model] = [896, 896]

Llaminar expectation (from MPIAttentionKernel):
- local_wo should be: [local_head_dim, d_model] = [K, N] for C = A @ B
- Where A is [seq_len, local_head_dim] and C is [seq_len, d_model]

The question:
1. What does GGUF actually store for blk.0.attn_output.weight?
2. Does it match PyTorch [896, 896] or llaminar's expected [896, 896]?
3. Do we need to reverse/transpose?

Expected PyTorch behavior:
  y = x @ weight.T + bias
  where weight is [out_features, in_features] = [896, 896]
  x is [batch, seq, 896], y is [batch, seq, 896]

Expected llaminar behavior (from scalar validation):
  C[i][j] = sum_k A[i][k] * B[k][j]
  where B is [K, N] = [896, 896]
  A is [M, K] = [seq_len, 896], C is [M, N] = [seq_len, 896]

So if PyTorch weight is [N, K] = [out, in] = [896, 896]
And llaminar needs [K, N] = [in, out] = [896, 896]
Then they're TRANSPOSED relative to each other!

The fix should be to transpose B when loading, OR transpose during matmul.
""")

# Now let's verify with actual PyTorch model
print("\n" + "="*80)
print("PYTORCH MODEL WEIGHTS")
print("="*80)

from transformers import AutoModel
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"\nLoading: {model_name}")
hf_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)

print("\nPyTorch layer 0 weights:")
print(f"  o_proj.weight shape: {hf_model.layers[0].self_attn.o_proj.weight.shape}")
print(f"  q_proj.weight shape: {hf_model.layers[0].self_attn.q_proj.weight.shape}")
print(f"  k_proj.weight shape: {hf_model.layers[0].self_attn.k_proj.weight.shape}")
print(f"  v_proj.weight shape: {hf_model.layers[0].self_attn.v_proj.weight.shape}")

print("\nPyTorch nn.Linear convention:")
print("  weight.shape = [out_features, in_features]")
print("  forward: y = F.linear(x, weight, bias) = x @ weight.T + bias")

