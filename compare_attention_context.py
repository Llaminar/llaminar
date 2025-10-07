#!/usr/bin/env python3
"""
Compare llaminar vs PyTorch ATTENTION_CONTEXT to diagnose divergence.

This script compares the attention context (before o_proj) between llaminar
and PyTorch to determine if the problem is in the output projection or earlier.
"""

import numpy as np
import sys
import os

print("="*80)
print("ATTENTION CONTEXT COMPARISON")
print("="*80)

# Load PyTorch reference
pytorch_dir = "/tmp/pytorch_snapshots_openblas"
pytorch_context = np.load(f"{pytorch_dir}/ATTENTION_CONTEXT_0.npy")

print(f"\n✓ PyTorch ATTENTION_CONTEXT loaded: {pytorch_context.shape}")

# Check if llaminar snapshot exists
# The test framework should have captured it, but let's check
llaminar_snapshot_path = "/tmp/test_snapshots/ATTENTION_CONTEXT_0.npy"

if not os.path.exists(llaminar_snapshot_path):
    print(f"\n❌ Llaminar snapshot not found at: {llaminar_snapshot_path}")
    print("\nPossible reasons:")
    print("1. Parity test segfaulted before saving snapshots")
    print("2. LLAMINAR_PARITY_CAPTURE environment variable not set")
    print("3. Snapshot directory not configured correctly")
    print("\nChecking for alternate locations...")
    
    # Check common alternate locations
    alternate_paths = [
        "/tmp/llaminar_parity_snapshots/ATTENTION_CONTEXT_0.npy",
        "./llaminar_snapshots/ATTENTION_CONTEXT_0.npy",
        "./build/ATTENTION_CONTEXT_0.npy",
    ]
    
    found = False
    for path in alternate_paths:
        if os.path.exists(path):
            llaminar_snapshot_path = path
            found = True
            print(f"✓ Found at: {path}")
            break
    
    if not found:
        print("\n❌ Could not find llaminar ATTENTION_CONTEXT snapshot")
        print("\nTo capture llaminar snapshots, run:")
        print("  export LLAMINAR_PARITY_CAPTURE=1")
        print("  mpirun -np 2 ./build/test_parity_framework --gtest_filter=ParityFramework.OpenBLASPrefillVsPyTorch")
        sys.exit(1)

# Load llaminar snapshot
llaminar_context = np.load(llaminar_snapshot_path)
print(f"✓ Llaminar ATTENTION_CONTEXT loaded: {llaminar_context.shape}")

# Check shapes match
if pytorch_context.shape != llaminar_context.shape:
    print(f"\n⚠️  SHAPE MISMATCH!")
    print(f"  PyTorch: {pytorch_context.shape}")
    print(f"  Llaminar: {llaminar_context.shape}")
    print("\nNote: This could be due to multi-rank partial heads")
    print("PyTorch has full 896 features, llaminar might have partial heads")

# Flatten for comparison
pt_flat = pytorch_context.reshape(-1)
ll_flat = llaminar_context.reshape(-1)

# Compute differences
min_len = min(len(pt_flat), len(ll_flat))
diff = np.abs(pt_flat[:min_len] - ll_flat[:min_len])
max_abs = np.max(diff)
mean_abs = np.mean(diff)

# Relative error
mask = np.abs(pt_flat[:min_len]) > 1e-8
if np.any(mask):
    rel_l2_sq = np.sum(diff[mask]**2) / np.sum(pt_flat[mask]**2)
    rel_l2 = np.sqrt(rel_l2_sq)
else:
    rel_l2 = 0.0

print(f"\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

print(f"\nMax absolute difference: {max_abs:.6e}")
print(f"Mean absolute difference: {mean_abs:.6e}")
print(f"Relative L2 error: {rel_l2:.6e}")

# Define tolerance (same as parity test)
MAX_ABS_TOL = 0.1
REL_L2_TOL = 0.05

if max_abs < MAX_ABS_TOL and rel_l2 < REL_L2_TOL:
    print(f"\n✅ ATTENTION_CONTEXT MATCHES!")
    print(f"   max_abs={max_abs:.6e} < {MAX_ABS_TOL}")
    print(f"   rel_l2={rel_l2:.6e} < {REL_L2_TOL}")
    print(f"\n🎯 CONCLUSION: Problem is in OUTPUT PROJECTION (o_proj)")
    print(f"   - Attention computation is correct")
    print(f"   - QKV projections, RoPE, attention scores are correct")
    print(f"   - Issue is in the final linear projection weights")
    print(f"\n   Next steps:")
    print(f"   1. Check o_proj weight loading from GGUF")
    print(f"   2. Verify weight orientation/transpose")
    print(f"   3. Check head concatenation order")
else:
    print(f"\n❌ ATTENTION_CONTEXT DIVERGES!")
    print(f"   max_abs={max_abs:.6e} >= {MAX_ABS_TOL}")
    print(f"   rel_l2={rel_l2:.6e} >= {REL_L2_TOL}")
    print(f"\n🎯 CONCLUSION: Problem is BEFORE output projection")
    print(f"   - Issue in QKV projection, RoPE, or attention computation")
    print(f"\n   Next steps:")
    print(f"   1. Compare Q_PROJECTION, K_PROJECTION, V_PROJECTION")
    print(f"   2. Compare Q_ROPE, K_ROPE")
    print(f"   3. Compare ATTENTION_SCORES, ATTENTION_SOFTMAX")

# Show specific values at position 0, dimension 842 (where error was detected)
print(f"\n" + "="*80)
print("SPECIFIC VALUES AT [position=0, dim=842]")
print("="*80)

# Reshape to [seq_len, features]
if len(pytorch_context.shape) == 3:
    pt_context_2d = pytorch_context.reshape(pytorch_context.shape[1], -1)
else:
    pt_context_2d = pytorch_context

if len(llaminar_context.shape) == 3:
    ll_context_2d = llaminar_context.reshape(llaminar_context.shape[1], -1)
else:
    ll_context_2d = llaminar_context

if pt_context_2d.shape[1] > 842 and ll_context_2d.shape[1] > 842:
    print(f"\nPyTorch context[0, 842]:  {pt_context_2d[0, 842]:.6f}")
    print(f"Llaminar context[0, 842]: {ll_context_2d[0, 842]:.6f}")
    print(f"Difference:               {abs(pt_context_2d[0, 842] - ll_context_2d[0, 842]):.6f}")
    
    # Also check output for reference
    pytorch_output = np.load(f"{pytorch_dir}/ATTENTION_OUTPUT_0.npy")
    pt_output_2d = pytorch_output.reshape(pytorch_output.shape[1], -1)
    print(f"\nFor reference:")
    print(f"PyTorch output[0, 842]:   {pt_output_2d[0, 842]:.6f}")
    print(f"Expected delta (o_proj):  {pt_output_2d[0, 842] - pt_context_2d[0, 842]:.6f}")
else:
    print(f"\n⚠️  Cannot check dimension 842 (shapes too small)")

print(f"\n" + "="*80)
