#!/usr/bin/env python3
"""
Measure run-to-run variance in PyTorch model to establish baseline jitter.

This helps us set appropriate parity test thresholds - we should not expect
llaminar to be closer to PyTorch than PyTorch is to itself across runs.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add python directory to path for imports
script_dir = Path(__file__).parent.absolute()
workspace_dir = script_dir.parent.absolute()
python_dir = workspace_dir / "python"
sys.path.insert(0, str(python_dir))

from reference import ModelRegistry
from reference.capture_pytorch_layers import QwenLayerCapture


def compute_tensor_stats(tensors: List[np.ndarray]) -> Dict[str, float]:
    """Compute statistical measures across multiple runs of the same tensor."""
    # Stack tensors: [num_runs, ...tensor_shape]
    stacked = np.stack(tensors, axis=0)
    
    # Compute pairwise differences between all runs
    num_runs = len(tensors)
    all_abs_diffs = []
    all_rel_l2 = []
    
    for i in range(num_runs):
        for j in range(i + 1, num_runs):
            diff = tensors[i] - tensors[j]
            abs_diff = np.abs(diff)
            all_abs_diffs.append(abs_diff)
            
            # Relative L2 norm
            l2_diff = np.linalg.norm(diff)
            l2_ref = np.linalg.norm(tensors[i])
            if l2_ref > 1e-8:
                all_rel_l2.append(l2_diff / l2_ref)
    
    # Aggregate statistics
    all_abs_diffs_flat = np.concatenate([d.flatten() for d in all_abs_diffs])
    
    return {
        "max_abs_diff": float(np.max(all_abs_diffs_flat)),
        "mean_abs_diff": float(np.mean(all_abs_diffs_flat)),
        "median_abs_diff": float(np.median(all_abs_diffs_flat)),
        "std_abs_diff": float(np.std(all_abs_diffs_flat)),
        "p95_abs_diff": float(np.percentile(all_abs_diffs_flat, 95)),
        "p99_abs_diff": float(np.percentile(all_abs_diffs_flat, 99)),
        "max_rel_l2": float(np.max(all_rel_l2)) if all_rel_l2 else 0.0,
        "mean_rel_l2": float(np.mean(all_rel_l2)) if all_rel_l2 else 0.0,
        "median_rel_l2": float(np.median(all_rel_l2)) if all_rel_l2 else 0.0,
        "num_runs": num_runs,
        "num_comparisons": len(all_abs_diffs)
    }


def run_pytorch_multiple_times(
    model_path: str,
    tokens: List[int],
    num_runs: int = 10
) -> Dict[str, List[np.ndarray]]:
    """Run PyTorch model multiple times and collect all intermediate tensors."""
    print(f"Running PyTorch model {num_runs} times to measure variance...")
    print(f"Model: {model_path}")
    print(f"Tokens: {tokens}")
    print()
    
    # Storage for all runs: {stage_name: [run0_tensor, run1_tensor, ...]}
    all_runs = {}
    
    for run_idx in range(num_runs):
        print(f"Run {run_idx + 1}/{num_runs}...", end=" ", flush=True)
        
        # Create capture instance
        capture = QwenLayerCapture(model_path)
        capture.load_model()
        capture.setup_hooks()
        
        # Run inference
        try:
            snapshots = capture.capture_forward_pass(tokens)
            
            # Convert to numpy and store
            for stage_name, tensor in snapshots.items():
                if isinstance(tensor, np.ndarray):
                    np_tensor = tensor.astype(np.float32)
                elif isinstance(tensor, torch.Tensor):
                    np_tensor = tensor.cpu().numpy().astype(np.float32)
                else:
                    continue
                    
                if stage_name not in all_runs:
                    all_runs[stage_name] = []
                all_runs[stage_name].append(np_tensor)
            
            print(f"✓ ({len(snapshots)} stages)")
        finally:
            # Clean up hooks
            capture.hook_manager.clear()
            del capture
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\nCollected {len(all_runs)} unique stages across {num_runs} runs")
    return all_runs


def analyze_variance(
    all_runs: Dict[str, List[np.ndarray]]
) -> Dict[str, Dict[str, float]]:
    """Analyze variance for each stage."""
    print("\nAnalyzing variance across runs...")
    
    variance_stats = {}
    
    for stage_name in sorted(all_runs.keys()):
        tensors = all_runs[stage_name]
        
        # Compute statistics
        stats = compute_tensor_stats(tensors)
        variance_stats[stage_name] = stats
        
        # Print summary
        print(f"\n{stage_name}:")
        print(f"  Shape: {tensors[0].shape}")
        print(f"  Max abs diff:    {stats['max_abs_diff']:.6e}")
        print(f"  Mean abs diff:   {stats['mean_abs_diff']:.6e}")
        print(f"  Median abs diff: {stats['median_abs_diff']:.6e}")
        print(f"  P95 abs diff:    {stats['p95_abs_diff']:.6e}")
        print(f"  P99 abs diff:    {stats['p99_abs_diff']:.6e}")
        print(f"  Max rel L2:      {stats['max_rel_l2']:.6e}")
        print(f"  Mean rel L2:     {stats['mean_rel_l2']:.6e}")
        print(f"  Median rel L2:   {stats['median_rel_l2']:.6e}")
    
    return variance_stats


def suggest_thresholds(
    variance_stats: Dict[str, Dict[str, float]],
    safety_margin: float = 3.0
) -> Dict[str, Tuple[float, float]]:
    """
    Suggest parity test thresholds based on observed variance.
    
    Args:
        variance_stats: Statistics from variance analysis
        safety_margin: Multiplier for suggested thresholds (default 3x observed variance)
    
    Returns:
        Dict mapping stage_name -> (max_abs_threshold, rel_l2_threshold)
    """
    print(f"\n{'='*80}")
    print(f"SUGGESTED THRESHOLDS (safety margin: {safety_margin}x observed variance)")
    print(f"{'='*80}\n")
    
    thresholds = {}
    
    # Group stages by type for better organization
    stage_groups = {
        "EMBEDDING": [],
        "NORM": [],
        "PROJECTION": [],
        "ROPE": [],
        "ATTENTION": [],
        "FFN": [],
        "OTHER": []
    }
    
    for stage_name in sorted(variance_stats.keys()):
        stats = variance_stats[stage_name]
        
        # Use P99 + safety margin for max_abs (to handle outliers)
        # Use max + safety margin for rel_l2 (more stable metric)
        suggested_max_abs = stats['p99_abs_diff'] * safety_margin
        suggested_rel_l2 = stats['max_rel_l2'] * safety_margin
        
        thresholds[stage_name] = (suggested_max_abs, suggested_rel_l2)
        
        # Categorize stage
        if "EMBEDDING" in stage_name:
            stage_groups["EMBEDDING"].append(stage_name)
        elif "NORM" in stage_name:
            stage_groups["NORM"].append(stage_name)
        elif "PROJECTION" in stage_name or "_Q_" in stage_name or "_K_" in stage_name or "_V_" in stage_name:
            stage_groups["PROJECTION"].append(stage_name)
        elif "ROPE" in stage_name:
            stage_groups["ROPE"].append(stage_name)
        elif "ATTENTION" in stage_name or "SOFTMAX" in stage_name or "SCORES" in stage_name:
            stage_groups["ATTENTION"].append(stage_name)
        elif "FFN" in stage_name or "SWIGLU" in stage_name or "GATE" in stage_name or "UP" in stage_name or "DOWN" in stage_name:
            stage_groups["FFN"].append(stage_name)
        else:
            stage_groups["OTHER"].append(stage_name)
    
    # Print grouped recommendations
    for group_name, stage_list in stage_groups.items():
        if not stage_list:
            continue
        
        print(f"\n{group_name} stages:")
        print(f"{'Stage':<40} {'P99 abs':<12} {'Suggested max_abs':<18} {'Max rel_l2':<12} {'Suggested rel_l2':<18}")
        print("-" * 100)
        
        for stage_name in sorted(stage_list):
            stats = variance_stats[stage_name]
            max_abs_thresh, rel_l2_thresh = thresholds[stage_name]
            
            print(f"{stage_name:<40} {stats['p99_abs_diff']:>11.4e} {max_abs_thresh:>17.4e} {stats['max_rel_l2']:>11.4e} {rel_l2_thresh:>17.4e}")
    
    return thresholds


def save_results(
    variance_stats: Dict[str, Dict[str, float]],
    thresholds: Dict[str, Tuple[float, float]],
    output_dir: str
):
    """Save variance analysis results to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save variance statistics
    variance_file = output_path / "pytorch_variance_stats.json"
    with open(variance_file, 'w') as f:
        json.dump(variance_stats, f, indent=2)
    print(f"\nVariance statistics saved to: {variance_file}")
    
    # Save suggested thresholds
    thresholds_dict = {
        stage: {"max_abs": max_abs, "rel_l2": rel_l2}
        for stage, (max_abs, rel_l2) in thresholds.items()
    }
    thresholds_file = output_path / "suggested_thresholds.json"
    with open(thresholds_file, 'w') as f:
        json.dump(thresholds_dict, f, indent=2)
    print(f"Suggested thresholds saved to: {thresholds_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Measure PyTorch model run-to-run variance to establish parity test thresholds"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/qwen2.5-0.5b-instruct-fp16.gguf",
        help="Path to GGUF model file"
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated token IDs (default: 1,2,3,4,5)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of runs to measure variance (default: 10)"
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=3.0,
        help="Safety margin multiplier for thresholds (default: 3.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="parity_data",
        help="Directory to save results (default: parity_data)"
    )
    
    args = parser.parse_args()
    
    # Parse tokens
    tokens = [int(t.strip()) for t in args.tokens.split(",")]
    
    print(f"{'='*80}")
    print(f"PyTorch Variance Measurement")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Tokens: {tokens}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Safety margin: {args.safety_margin}x")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # Run multiple times
    all_runs = run_pytorch_multiple_times(args.model, tokens, args.num_runs)
    
    # Analyze variance
    variance_stats = analyze_variance(all_runs)
    
    # Suggest thresholds
    thresholds = suggest_thresholds(variance_stats, args.safety_margin)
    
    # Save results
    save_results(variance_stats, thresholds, args.output_dir)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"✓ Completed {args.num_runs} runs")
    print(f"✓ Analyzed {len(variance_stats)} stages")
    print(f"✓ Generated thresholds with {args.safety_margin}x safety margin")
    print(f"✓ Results saved to {args.output_dir}/")
    print()
    print("Next steps:")
    print("1. Review suggested thresholds in suggested_thresholds.json")
    print("2. Update test_parity_framework.cpp with new thresholds")
    print("3. Re-run parity tests with updated thresholds")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
