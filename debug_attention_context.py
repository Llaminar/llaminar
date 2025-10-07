#!/usr/bin/env python3
"""
Debug script to check attention context values from PyTorch reference.

This helps us understand what the attention mechanism produces before the output projection.
"""

import numpy as np
import sys

# Load PyTorch snapshots
snapshot_dir = "/tmp/pytorch_snapshots_openblas"

print("="*80)
print("ATTENTION MECHANISM DEBUG - Layer 0")
print("="*80)

# Load intermediate stages for layer 0
embedding = np.load(f"{snapshot_dir}/EMBEDDING_-1.npy")
attn_norm = np.load(f"{snapshot_dir}/ATTENTION_NORM_0.npy")
q_proj = np.load(f"{snapshot_dir}/Q_PROJECTION_0.npy")
k_proj = np.load(f"{snapshot_dir}/K_PROJECTION_0.npy")
v_proj = np.load(f"{snapshot_dir}/V_PROJECTION_0.npy")
q_rope = np.load(f"{snapshot_dir}/Q_ROPE_0.npy")
k_rope = np.load(f"{snapshot_dir}/K_ROPE_0.npy")
attn_scores = np.load(f"{snapshot_dir}/ATTENTION_SCORES_0.npy")
attn_softmax = np.load(f"{snapshot_dir}/ATTENTION_SOFTMAX_0.npy")
attn_context = np.load(f"{snapshot_dir}/ATTENTION_CONTEXT_0.npy")
attn_output = np.load(f"{snapshot_dir}/ATTENTION_OUTPUT_0.npy")

print(f"\nEMBEDDING shape: {embedding.shape}")
print(f"ATTENTION_NORM shape: {attn_norm.shape}")
print(f"Q_PROJECTION shape: {q_proj.shape}")
print(f"K_PROJECTION shape: {k_proj.shape}")
print(f"V_PROJECTION shape: {v_proj.shape}")
print(f"Q_ROPE shape: {q_rope.shape}")
print(f"K_ROPE shape: {k_rope.shape}")
print(f"ATTENTION_SCORES shape: {attn_scores.shape}")
print(f"ATTENTION_SOFTMAX shape: {attn_softmax.shape}")
print(f"ATTENTION_CONTEXT shape: {attn_context.shape}")
print(f"ATTENTION_OUTPUT shape: {attn_output.shape}")

# Check attention context values at position 0, dimension 842
print(f"\n" + "="*80)
print("ATTENTION CONTEXT VALUES (before output projection)")
print("="*80)

# Flatten to [seq_len, hidden_dim] for easier inspection
context_flat = attn_context.reshape(5, 896)  # [batch=1, seq=5, hidden=896] -> [5, 896]

print(f"\nContext at position 0:")
print(f"  Shape: {context_flat[0].shape}")
print(f"  Min: {context_flat[0].min():.6f}")
print(f"  Max: {context_flat[0].max():.6f}")
print(f"  Mean: {context_flat[0].mean():.6f}")
print(f"  Std: {context_flat[0].std():.6f}")
print(f"\n  Values around dimension 842:")
for i in range(840, 846):
    print(f"    context[0, {i}] = {context_flat[0, i]:.6f}")

# Check output projection values
print(f"\n" + "="*80)
print("ATTENTION OUTPUT VALUES (after o_proj)")
print("="*80)

output_flat = attn_output.reshape(5, 896)

print(f"\nOutput at position 0:")
print(f"  Shape: {output_flat[0].shape}")
print(f"  Min: {output_flat[0].min():.6f}")
print(f"  Max: {output_flat[0].max():.6f}")
print(f"  Mean: {output_flat[0].mean():.6f}")
print(f"  Std: {output_flat[0].std():.6f}")
print(f"\n  Values around dimension 842 (where error was detected):")
for i in range(840, 846):
    print(f"    output[0, {i}] = {output_flat[0, i]:.6f}  (PyTorch expected)")

# Show the delta from context to output (what o_proj did)
print(f"\n" + "="*80)
print("OUTPUT PROJECTION EFFECT")
print("="*80)
print(f"\nDimension 842:")
print(f"  Context:  {context_flat[0, 842]:.6f}")
print(f"  Output:   {output_flat[0, 842]:.6f}")
print(f"  Delta:    {output_flat[0, 842] - context_flat[0, 842]:.6f}")

# Statistics on attention softmax (to check if attention is reasonable)
print(f"\n" + "="*80)
print("ATTENTION WEIGHTS (softmax probabilities)")
print("="*80)

# Shape: [batch=1, heads=14, seq_q=5, seq_k=5]
print(f"\nAttention weights shape: {attn_softmax.shape}")
print(f"Head 0, position 0 attends to:")
for j in range(5):
    weight = attn_softmax[0, 0, 0, j]
    print(f"  position {j}: {weight:.6f}")

print(f"\nHead 13 (last head), position 0 attends to:")
for j in range(5):
    weight = attn_softmax[0, 13, 0, j]
    print(f"  position {j}: {weight:.6f}")

# Save a summary for easy viewing
print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✓ PyTorch snapshots successfully loaded")
print(f"✓ ATTENTION_CONTEXT exists (before o_proj)")
print(f"✓ This is the key comparison point")
print(f"\nNext step: Compare llaminar's attention context with PyTorch's")
print(f"Expected: llaminar attention context should match PyTorch")
print(f"If context matches, problem is in o_proj (output projection)")
print(f"If context differs, problem is earlier (QKV, RoPE, or attention scores)")
