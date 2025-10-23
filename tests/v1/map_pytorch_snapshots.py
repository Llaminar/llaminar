#!/usr/bin/env python3
"""
Map PyTorch snapshot files to Llaminar test expected naming convention.

PyTorch format: embeddings.npy, layer_0_attn_out.npy, final_norm_out.npy, logits.npy
Llaminar format: EMBEDDING_-1.npy, ATTENTION_OUTPUT_0.npy, FINAL_NORM_-1.npy, LM_HEAD_-1.npy

@author David Sanftenberg
"""

import os
import sys
import shutil
from pathlib import Path


def map_pytorch_to_llaminar(pytorch_name: str) -> str:
    """
    Map PyTorch snapshot name to Llaminar expected name.
    
    Args:
        pytorch_name: PyTorch snapshot filename (without .npy)
        
    Returns:
        Llaminar format name (without .npy), or None if unmapped
    """
    # Global snapshots (no layer index)
    if pytorch_name == "embeddings":
        return "EMBEDDING_-1"
    elif pytorch_name == "final_norm_out":
        return "FINAL_NORM_-1"
    elif pytorch_name == "logits":
        return "LM_HEAD_-1"
    
    # Layer-specific snapshots
    # Format: layer_N_<stage>_out
    if pytorch_name.startswith("layer_"):
        parts = pytorch_name.split("_")
        if len(parts) < 3:
            return None
        
        try:
            layer_idx = int(parts[1])
        except ValueError:
            return None
        
        stage = "_".join(parts[2:])  # Join remaining parts (e.g., "attn_out", "post_attn_norm_out")
        
        # Map stage names
        if stage == "attn_out":
            return f"ATTENTION_OUTPUT_{layer_idx}"
        elif stage == "ffn_out":
            return f"FFN_DOWN_{layer_idx}"
        elif stage == "input_norm_out":
            return f"ATTENTION_NORM_{layer_idx}"
        elif stage == "post_attn_norm_out":
            return f"FFN_NORM_{layer_idx}"
        elif stage == "out":
            # Full layer output (after attention + FFN residuals)
            return f"FFN_RESIDUAL_{layer_idx}"
    
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python map_pytorch_snapshots.py <pytorch_snapshots_dir> [output_dir]")
        print("\nExample:")
        print("  python tests/map_pytorch_snapshots.py pytorch_snapshots/ pytorch_snapshots_mapped/")
        print("\nOr in-place (careful!):")
        print("  python tests/map_pytorch_snapshots.py pytorch_snapshots/")
        return 1
    
    source_dir = Path(sys.argv[1])
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return 1
    
    # Default: create mapped directory alongside source
    if len(sys.argv) >= 3:
        target_dir = Path(sys.argv[2])
    else:
        target_dir = source_dir.parent / (source_dir.name + "_mapped")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Mapping PyTorch snapshots:")
    print(f"  Source: {source_dir}")
    print(f"  Target: {target_dir}")
    print()
    
    mapped_count = 0
    skipped_count = 0
    
    for npy_file in sorted(source_dir.glob("*.npy")):
        pytorch_name = npy_file.stem  # filename without .npy
        llaminar_name = map_pytorch_to_llaminar(pytorch_name)
        
        if llaminar_name:
            target_file = target_dir / f"{llaminar_name}.npy"
            shutil.copy2(npy_file, target_file)
            print(f"  ✓ {pytorch_name}.npy → {llaminar_name}.npy")
            mapped_count += 1
        else:
            print(f"  - {pytorch_name}.npy (skipped - no mapping)")
            skipped_count += 1
    
    print()
    print(f"Summary:")
    print(f"  Mapped: {mapped_count} files")
    print(f"  Skipped: {skipped_count} files")
    print()
    print(f"Mapped snapshots available in: {target_dir}")
    print(f"\nTo run the test:")
    print(f"  export PYTORCH_SNAPSHOT_DIR={target_dir}")
    print(f"  export PYTORCH_SNAPSHOT_TOKENS=1639,266,285,17,10,17,30")
    print(f"  ./build/test_parity_framework --gtest_filter=*PyTorchReference*")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
