#!/usr/bin/env python3
"""
Verify that transposing the o_proj weight fixes the parity issue.
This script simulates what llaminar should do vs. what PyTorch does.
"""

import numpy as np

# Load the actual snapshots
pytorch_context = np.load("/tmp/pytorch_snapshots_openblas/ATTENTION_CONTEXT_0.npy")
llaminar_context = np.load("/tmp/test_snapshots/ATTENTION_CONTEXT_0.npy")
pytorch_output = np.load("/tmp/pytorch_snapshots_openblas/ATTENTION_OUTPUT_0.npy")

print("="*80)
print("TRANSPOSE FIX VERIFICATION")
print("="*80)

# Load PyTorch o_proj weight
from transformers import AutoModel
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
hf_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
o_proj_weight = hf_model.layers[0].self_attn.o_proj.weight.detach().cpu().numpy()

print(f"\nPyTorch o_proj weight shape: {o_proj_weight.shape}")  # [896, 896]
print(f"Context shape: {pytorch_context.shape}")  # [1, 5, 896]
print(f"Expected output shape: {pytorch_output.shape}")  # [1, 5, 896]

# Remove batch dimension for simplicity
context = pytorch_context[0]  # [5, 896]
expected_output = pytorch_output[0]  # [5, 896]

print(f"\nWorking with:")
print(f"  context: {context.shape}")
print(f"  o_proj: {o_proj_weight.shape}")

# Simulate what llaminar currently does (WRONG)
print("\n" + "-"*80)
print("CURRENT LLAMINAR (WITHOUT TRANSPOSE)")
print("-"*80)

wrong_output = context @ o_proj_weight  # [5, 896] @ [896, 896] = [5, 896]
error_wrong = np.abs(wrong_output - expected_output).max()
rel_l2_wrong = np.linalg.norm(wrong_output - expected_output) / np.linalg.norm(expected_output)

print(f"Output shape: {wrong_output.shape}")
print(f"Max abs error: {error_wrong:.6f}")
print(f"Relative L2: {rel_l2_wrong:.6f}")
print(f"Sample [0, 842]: {wrong_output[0, 842]:.6f} (expected: {expected_output[0, 842]:.6f})")

# Simulate what llaminar should do (CORRECT)
print("\n" + "-"*80)
print("FIXED LLAMINAR (WITH TRANSPOSE)")
print("-"*80)

correct_output = context @ o_proj_weight.T  # [5, 896] @ [896, 896].T = [5, 896]
error_correct = np.abs(correct_output - expected_output).max()
rel_l2_correct = np.linalg.norm(correct_output - expected_output) / np.linalg.norm(expected_output)

print(f"Output shape: {correct_output.shape}")
print(f"Max abs error: {error_correct:.6f}")
print(f"Relative L2: {rel_l2_correct:.6f}")
print(f"Sample [0, 842]: {correct_output[0, 842]:.6f} (expected: {expected_output[0, 842]:.6f})")

# Comparison
print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nWithout transpose:")
print(f"  ❌ Max error: {error_wrong:.6f}")
print(f"  ❌ Rel L2: {rel_l2_wrong:.6f}")

print(f"\nWith transpose:")
if rel_l2_correct < 1e-4:
    print(f"  ✅ Max error: {error_correct:.6e}")
    print(f"  ✅ Rel L2: {rel_l2_correct:.6e}")
    print(f"\n✅ TRANSPOSE FIX CONFIRMED!")
    print(f"   Adding transpose_B=true will fix the parity issue.")
else:
    print(f"  ⚠️  Max error: {error_correct:.6f}")
    print(f"  ⚠️  Rel L2: {rel_l2_correct:.6f}")
    print(f"\n⚠️  Transpose helps but doesn't fully fix it.")
    print(f"   May be additional issues (quantization, etc.)")
