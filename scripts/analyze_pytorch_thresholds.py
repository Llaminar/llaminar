#!/usr/bin/env python3
"""
Analyze PyTorch snapshot magnitudes to suggest parity thresholds.

This script loads individual .npy snapshot files and computes magnitude-based
thresholds for parity testing.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple
import argparse


def load_snapshots_from_directory(snapshot_dir: str) -> Dict[str, np.ndarray]:
    """Load all .npy snapshot files from directory."""
    snapshot_path = Path(snapshot_dir)
    print(f"Loading PyTorch snapshots from: {snapshot_path}")
    
    snapshots = {}
    npy_files = sorted(snapshot_path.glob("*.npy"))
    
    for npy_file in npy_files:
        stage_name = npy_file.stem  # Filename without .npy extension
        tensor = np.load(npy_file)
        snapshots[stage_name] = tensor
    
    print(f"  ✓ Loaded {len(snapshots)} stages")
    return snapshots


def compute_threshold_from_magnitude(tensor: np.ndarray, max_abs_factor: float = 0.01) -> float:
    """
    Compute max_abs threshold based on tensor magnitude.
    
    Uses RMS (root mean square) as characteristic scale.
    """
    rms = np.sqrt(np.mean(tensor ** 2))
    # Max abs threshold: 1% of RMS by default, with floor at 1e-6
    return max(rms * max_abs_factor, 1e-6)


def analyze_snapshots(
    snapshots: Dict[str, np.ndarray],
    max_abs_factor: float = 0.01,
    rel_l2_threshold: float = 0.05
) -> Dict[str, Tuple[float, float]]:
    """Compute thresholds for each stage based on magnitude."""
    
    thresholds = {}
    
    print(f"\nComputing thresholds...")
    print(f"  Max abs factor: {max_abs_factor * 100:.1f}% of RMS")
    print(f"  Rel L2 threshold: {rel_l2_threshold * 100:.1f}%\n")
    
    for stage_name, tensor in sorted(snapshots.items()):
        max_abs_thresh = compute_threshold_from_magnitude(tensor, max_abs_factor)
        rel_l2_thresh = rel_l2_threshold
        thresholds[stage_name] = (max_abs_thresh, rel_l2_thresh)
    
    return thresholds


def print_threshold_summary(
    snapshots: Dict[str, np.ndarray],
    thresholds: Dict[str, Tuple[float, float]]
):
    """Print grouped threshold summary."""
    
    # Group stages by type
    stage_groups = {
        "EMBEDDING": [],
        "ATTENTION_NORM": [],
        "Q_PROJECTION": [],
        "K_PROJECTION": [],
        "V_PROJECTION": [],
        "ROPE_APPLICATION": [],
        "ATTENTION_SCORES": [],
        "ATTENTION_SOFTMAX": [],
        "ATTENTION_CONTEXT": [],
        "ATTENTION_OUTPUT": [],
        "ATTENTION_RESIDUAL": [],
        "FFN_NORM": [],
        "FFN_GATE": [],
        "FFN_UP": [],
        "FFN_SWIGLU": [],
        "FFN_DOWN": [],
        "FFN_RESIDUAL": [],
        "FINAL_NORM": [],
        "LM_HEAD": []
    }
    
    for stage_name in snapshots.keys():
        # Find which group this stage belongs to
        grouped = False
        for group_name in stage_groups.keys():
            if group_name in stage_name:
                stage_groups[group_name].append(stage_name)
                grouped = True
                break
        if not grouped:
            print(f"WARNING: Ungroup stage: {stage_name}")
    
    print(f"\n{'='*110}")
    print(f"MAGNITUDE-BASED THRESHOLDS")
    print(f"{'='*110}\n")
    
    for group_name, stage_list in stage_groups.items():
        if not stage_list:
            continue
        
        # Get representative stage (layer 0 if available)
        rep_stage = None
        for s in stage_list:
            if "_0" in s or "EMBEDDING" in s or "FINAL" in s or "LM_HEAD" in s:
                rep_stage = s
                break
        if not rep_stage:
            rep_stage = stage_list[0]
        
        tensor = snapshots[rep_stage]
        max_abs_thresh, rel_l2_thresh = thresholds[rep_stage]
        rms = np.sqrt(np.mean(tensor ** 2))
        
        print(f"{group_name:<25} (n={len(stage_list):>2})  RMS={rms:>12.4e}  max_abs={max_abs_thresh:>12.4e}  rel_l2={rel_l2_thresh:.4f}")


def save_json_thresholds(
    thresholds: Dict[str, Tuple[float, float]],
    output_file: str
):
    """Save thresholds as JSON."""
    thresholds_dict = {
        stage: {"max_abs": float(max_abs), "rel_l2": float(rel_l2)}
        for stage, (max_abs, rel_l2) in thresholds.items()
    }
    
    with open(output_file, 'w') as f:
        json.dump(thresholds_dict, f, indent=2)
    
    print(f"\n✓ Saved JSON thresholds to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PyTorch snapshots to suggest parity thresholds"
    )
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default="parity_data",
        help="Directory containing .npy snapshot files (default: parity_data)"
    )
    parser.add_argument(
        "--max-abs-factor",
        type=float,
        default=0.01,  # 1% of RMS
        help="Max absolute error as fraction of RMS (default: 0.01 = 1%%)"
    )
    parser.add_argument(
        "--rel-l2-threshold",
        type=float,
        default=0.05,  # 5%
        help="Relative L2 error threshold (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="parity_data/suggested_thresholds.json",
        help="Output JSON file (default: parity_data/suggested_thresholds.json)"
    )
    
    args = parser.parse_args()
    
    print(f"{'='*110}")
    print(f"Parity Threshold Analysis from PyTorch Magnitudes")
    print(f"{'='*110}\n")
    print(f"Snapshot directory: {args.snapshot_dir}")
    print(f"Max abs factor: {args.max_abs_factor * 100:.1f}% of RMS")
    print(f"Rel L2 threshold: {args.rel_l2_threshold * 100:.1f}%")
    print()
    
    # Load snapshots
    snapshots = load_snapshots_from_directory(args.snapshot_dir)
    
    # Compute thresholds
    thresholds = analyze_snapshots(
        snapshots,
        max_abs_factor=args.max_abs_factor,
        rel_l2_threshold=args.rel_l2_threshold
    )
    
    # Print summary
    print_threshold_summary(snapshots, thresholds)
    
    # Save outputs
    output_dir = Path(args.output_json).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json_thresholds(thresholds, args.output_json)
    
    print(f"\n{'='*110}")
    print("SUMMARY")
    print(f"{'='*110}")
    print(f"✓ Analyzed {len(snapshots)} stages")
    print(f"✓ Generated magnitude-based thresholds")
    print(f"✓ Key insight: Thresholds scale with tensor RMS magnitude")
    print(f"✓ Large activations (FFN) get larger absolute tolerance")
    print(f"✓ Small activations (norms) get tighter absolute tolerance")
    print(f"✓ All stages use same relative L2 threshold (magnitude-independent)")
    print(f"{'='*110}\n")
    
    # Print note about ATTENTION_CONTEXT
    if any("ATTENTION_CONTEXT" in s for s in snapshots.keys()):
        print("✓ ATTENTION_CONTEXT snapshots found! Parity test can now compare this critical stage.")
        print("  To enable in C++ test, add ATTENTION_CONTEXT snapshot emission in MPIAttentionKernel.\n")


if __name__ == "__main__":
    main()
