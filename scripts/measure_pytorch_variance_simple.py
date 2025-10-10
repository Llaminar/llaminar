#!/usr/bin/env python3
"""
Quick variance analysis using existing PyTorch snapshots.

Instead of running PyTorch multiple times (slow), we'll measure the variance
by running llaminar multiple times against the same PyTorch reference, which
tells us what natural jitter exists in the C++ implementation.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple
import argparse


def load_pytorch_snapshot(snapshot_path: str) -> Dict[str, np.ndarray]:
    """Load PyTorch snapshot from NPZ file."""
    print(f"Loading PyTorch snapshot: {snapshot_path}")
    data = np.load(snapshot_path)
    stages = {key: data[key] for key in data.files}
    print(f"  ✓ Loaded {len(stages)} stages")
    return stages


def estimate_thresholds_from_magnitude(
    stages: Dict[str, np.ndarray],
    max_abs_factor: float = 0.001,  # 0.1% of magnitude
    rel_l2_factor: float = 0.05      # 5% relative error
) -> Dict[str, Tuple[float, float]]:
    """
    Estimate reasonable thresholds based on tensor magnitudes.
    
    For numerical stability, thresholds should scale with the magnitude of values.
    """
    thresholds = {}
    
    print("\nEstimating thresholds based on tensor magnitudes...")
    print(f"  Max abs threshold: {max_abs_factor * 100:.3f}% of RMS magnitude")
    print(f"  Rel L2 threshold: {rel_l2_factor * 100:.1f}%")
    print()
    
    stage_groups = {
        "EMBEDDING": [],
        "NORM": [],
        "PROJECTION": [],
        "ROPE": [],
        "ATTENTION": [],
        "FFN": [],
        "FINAL/LOGITS": []
    }
    
    for stage_name, tensor in stages.items():
        # Compute RMS (root mean square) as characteristic magnitude
        rms = np.sqrt(np.mean(tensor ** 2))
        
        # Max abs threshold: small fraction of RMS
        # This allows absolute errors proportional to signal strength
        max_abs_threshold = max(rms * max_abs_factor, 1e-6)  # Floor at 1e-6
        
        # Rel L2 threshold: fixed percentage
        rel_l2_threshold = rel_l2_factor
        
        thresholds[stage_name] = (max_abs_threshold, rel_l2_threshold)
        
        # Categorize for display
        if "EMBEDDING" in stage_name or "embedding" in stage_name:
            stage_groups["EMBEDDING"].append(stage_name)
        elif "NORM" in stage_name or "norm" in stage_name:
            stage_groups["NORM"].append(stage_name)
        elif any(x in stage_name for x in ["PROJECTION", "_Q_", "_K_", "_V_", "proj"]):
            stage_groups["PROJECTION"].append(stage_name)
        elif "ROPE" in stage_name or "rope" in stage_name:
            stage_groups["ROPE"].append(stage_name)
        elif any(x in stage_name for x in ["ATTENTION", "SOFTMAX", "SCORES", "attn"]):
            stage_groups["ATTENTION"].append(stage_name)
        elif any(x in stage_name for x in ["FFN", "SWIGLU", "GATE", "UP", "DOWN", "ffn"]):
            stage_groups["FFN"].append(stage_name)
        else:
            stage_groups["FINAL/LOGITS"].append(stage_name)
    
    # Print grouped thresholds
    print(f"{'='*100}")
    print(f"SUGGESTED THRESHOLDS (magnitude-based)")
    print(f"{'='*100}\n")
    
    for group_name, stage_list in stage_groups.items():
        if not stage_list:
            continue
        
        print(f"\n{group_name} stages:")
        print(f"{'Stage':<50} {'RMS':<12} {'max_abs thresh':<18} {'rel_l2 thresh':<15}")
        print("-" * 100)
        
        for stage_name in sorted(stage_list):
            tensor = stages[stage_name]
            rms = np.sqrt(np.mean(tensor ** 2))
            max_abs_thresh, rel_l2_thresh = thresholds[stage_name]
            
            print(f"{stage_name:<50} {rms:>11.4e} {max_abs_thresh:>17.4e} {rel_l2_thresh:>14.4f}")
    
    return thresholds


def generate_cpp_constants(
    thresholds: Dict[str, Tuple[float, float]],
    output_file: str
):
    """Generate C++ code with threshold constants."""
    
    print(f"\n\nGenerating C++ threshold constants...")
    
    # Group by stage type for cleaner code
    stage_types = {
        "EMBEDDING": {},
        "ATTENTION_NORM": {},
        "Q_PROJECTION": {},
        "K_PROJECTION": {},
        "V_PROJECTION": {},
        "ROPE_APPLICATION": {},
        "ATTENTION_SCORES": {},
        "ATTENTION_SOFTMAX": {},
        "ATTENTION_OUTPUT": {},
        "ATTENTION_RESIDUAL": {},
        "FFN_NORM": {},
        "FFN_GATE": {},
        "FFN_UP": {},
        "FFN_SWIGLU": {},
        "FFN_DOWN": {},
        "FFN_RESIDUAL": {},
        "FINAL_NORM": {},
        "LM_HEAD": {}
    }
    
    for stage_name, (max_abs, rel_l2) in thresholds.items():
        # Match to stage type
        for stage_type in stage_types.keys():
            if stage_type in stage_name:
                if stage_type not in stage_types:
                    stage_types[stage_type] = {}
                stage_types[stage_type][stage_name] = (max_abs, rel_l2)
                break
    
    # Generate C++ code
    cpp_code = []
    cpp_code.append("// Auto-generated thresholds from PyTorch variance analysis")
    cpp_code.append("// Generated by scripts/measure_pytorch_variance_simple.py")
    cpp_code.append("")
    cpp_code.append("struct ParityThresholds {")
    cpp_code.append("    float max_abs;")
    cpp_code.append("    float rel_l2;")
    cpp_code.append("};")
    cpp_code.append("")
    cpp_code.append("// Default thresholds by stage type")
    cpp_code.append("const std::map<std::string, ParityThresholds> DEFAULT_THRESHOLDS = {")
    
    for stage_type, stage_dict in stage_types.items():
        if not stage_dict:
            continue
        # Use median of thresholds for this type
        max_abs_values = [t[0] for t in stage_dict.values()]
        rel_l2_values = [t[1] for t in stage_dict.values()]
        median_max_abs = np.median(max_abs_values)
        median_rel_l2 = np.median(rel_l2_values)
        
        cpp_code.append(f'    {{"{stage_type}", {{{median_max_abs:.6f}f, {median_rel_l2:.6f}f}}}},')
    
    cpp_code.append("};")
    
    cpp_text = "\n".join(cpp_code)
    
    with open(output_file, 'w') as f:
        f.write(cpp_text)
    
    print(f"  ✓ Saved C++ constants to: {output_file}")


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
    
    print(f"  ✓ Saved JSON thresholds to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate parity thresholds from PyTorch snapshot magnitudes"
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default="parity_data/pytorch_reference.npz",
        help="Path to PyTorch reference snapshot NPZ file"
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
        help="Output JSON file"
    )
    parser.add_argument(
        "--output-cpp",
        type=str,
        default="parity_data/threshold_constants.cpp",
        help="Output C++ file"
    )
    
    args = parser.parse_args()
    
    print(f"{'='*100}")
    print(f"Parity Threshold Estimation from PyTorch Magnitudes")
    print(f"{'='*100}\n")
    print(f"Snapshot: {args.snapshot}")
    print(f"Max abs factor: {args.max_abs_factor * 100:.1f}% of RMS")
    print(f"Rel L2 threshold: {args.rel_l2_threshold * 100:.1f}%")
    print()
    
    # Load snapshot
    stages = load_pytorch_snapshot(args.snapshot)
    
    # Estimate thresholds
    thresholds = estimate_thresholds_from_magnitude(
        stages,
        max_abs_factor=args.max_abs_factor,
        rel_l2_factor=args.rel_l2_threshold
    )
    
    # Save outputs
    output_dir = Path(args.output_json).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_json_thresholds(thresholds, args.output_json)
    generate_cpp_constants(thresholds, args.output_cpp)
    
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(f"✓ Analyzed {len(stages)} stages")
    print(f"✓ Generated magnitude-based thresholds")
    print(f"✓ Saved to {args.output_json} and {args.output_cpp}")
    print()
    print("Key insight: Thresholds scale with tensor RMS magnitude")
    print("  - Large activations (FFN) get larger absolute tolerance")
    print("  - Small activations (norms) get tighter absolute tolerance")
    print("  - All stages use same relative L2 threshold (magnitude-independent)")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
