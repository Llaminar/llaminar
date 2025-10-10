#!/usr/bin/env python3
"""
Generate variance-based thresholds for parity tests by running PyTorch multiple times.

This script:
1. Runs PyTorch model N times with identical inputs
2. Measures variance across runs for each stage
3. Generates dynamic thresholds based on observed variance
4. Saves both reference snapshots and threshold metadata

Usage:
    python scripts/generate_variance_thresholds.py \
        -m models/qwen2.5-0.5b-instruct-fp16.gguf \
        --tokens "1,2,3,4,5" \
        -o parity_data \
        --num-runs 3 \
        --safety-margin 5.0
"""

import argparse
import sys
import os
import json
import numpy as np
from pathlib import Path

# Add python directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from reference import PipelineStage
from reference.generate_test_snapshots import PipelineStageCapture


def stage_key_to_str(key):
    """
    Convert stage key to C++ test format: STAGE_<layer_idx>.npy
    
    Handles both tuple keys (PipelineStage, int) and string keys like "STAGE_layerN".
    """
    if isinstance(key, tuple):
        stage, layer = key
        stage_name = stage.name if hasattr(stage, 'name') else str(stage)
        return f"{stage_name}_{layer}"
    
    # Handle string keys from PipelineStageCapture like "ATTENTION_OUTPUT_layer5"
    if isinstance(key, str):
        if '_layer' in key:
            # Parse "STAGE_layerN" -> "STAGE_N"
            parts = key.split('_layer')
            if len(parts) == 2:
                try:
                    layer_idx = int(parts[1])
                    stage_name = parts[0]
                    return f"{stage_name}_{layer_idx}"
                except ValueError:
                    pass
        # Global stages like "EMBEDDING", "FINAL_NORM", "LM_HEAD" without layer index
        # These should have -1 as layer index
        return f"{key}_-1"
    
    return str(key)


def run_pytorch_inference(model_path: str, tokens: list[int], verbose: bool = False):
    """
    Run PyTorch inference and capture ALL intermediate tensors (detailed stages).
    
    Returns:
        dict: Mapping from stage_name -> numpy array
    """
    if verbose:
        print(f"Loading model from {model_path}...")
    
    # Use detailed PipelineStageCapture for comprehensive snapshot collection
    capturer = PipelineStageCapture(model_path, verbose=verbose)
    
    if verbose:
        print(f"Running inference on tokens: {tokens}")
    
    # Run inference and capture all stages
    capturer.capture_stages(tokens)
    
    # Get captures (dict of stage_name -> numpy array)
    snapshots = capturer.captures.copy()
    
    # Clean up
    capturer.cleanup()
    
    if verbose:
        print(f"Captured {len(snapshots)} snapshots")
    
    return snapshots


def compute_variance_statistics(runs: list[dict], verbose: bool = False):
    """
    Compute variance statistics across multiple PyTorch runs.
    
    Args:
        runs: List of snapshot dicts from each run
        
    Returns:
        tuple: (mean_snapshots, variance_stats)
            - mean_snapshots: dict mapping stage_name -> mean tensor
            - variance_stats: dict mapping stage_name -> variance metadata
    """
    if not runs:
        raise ValueError("No runs provided")
    
    # Ensure all runs have the same stages
    stage_names = set(runs[0].keys())
    for run in runs[1:]:
        if set(run.keys()) != stage_names:
            raise ValueError("Inconsistent stages across runs")
    
    mean_snapshots = {}
    variance_stats = {}
    
    # Sort stages by string representation (enum values can't be sorted directly)
    for stage_name in sorted(stage_names, key=str):
        # Collect tensors from all runs
        tensors = [run[stage_name] for run in runs]
        
        # Ensure all tensors have same shape
        shape = tensors[0].shape
        for i, t in enumerate(tensors[1:], 1):
            if t.shape != shape:
                raise ValueError(f"Shape mismatch for {stage_name}: run0={shape}, run{i}={t.shape}")
        
        # Stack tensors: (num_runs, *original_shape)
        stacked = np.stack(tensors, axis=0)
        
        # Compute statistics
        mean_tensor = np.mean(stacked, axis=0)
        
        # Deviation from mean for each run
        deviations = stacked - mean_tensor
        
        # Max absolute deviation across all runs
        max_abs_dev = np.max(np.abs(deviations))
        
        # RMS deviation
        rms_dev = np.sqrt(np.mean(deviations ** 2))
        
        # Mean absolute deviation
        mean_abs_dev = np.mean(np.abs(deviations))
        
        # 95th percentile of absolute deviations
        percentile_95 = np.percentile(np.abs(deviations), 95)
        
        # Tensor magnitude (RMS of mean tensor)
        tensor_rms = np.sqrt(np.mean(mean_tensor ** 2))
        
        mean_snapshots[stage_name] = mean_tensor
        variance_stats[stage_name] = {
            "num_runs": len(runs),
            "shape": list(shape),
            "max_abs_deviation": float(max_abs_dev),
            "rms_deviation": float(rms_dev),
            "mean_abs_deviation": float(mean_abs_dev),
            "percentile_95_deviation": float(percentile_95),
            "tensor_rms": float(tensor_rms),
        }
        
        # Handle tuple keys (PipelineStage, layer_idx)
        stage_str = str(stage_name[0]) if isinstance(stage_name, tuple) else str(stage_name)
        if verbose and "ATTENTION_SCORES" in stage_str:
            print(f"{stage_name}: max_abs_dev={max_abs_dev:.6e}, rms_dev={rms_dev:.6e}, tensor_rms={tensor_rms:.6e}")
    
    return mean_snapshots, variance_stats


def compute_dynamic_thresholds(variance_stats: dict, 
                               safety_margin: float = 5.0,
                               min_rel_l2: float = 0.05,
                               verbose: bool = False):
    """
    Compute dynamic thresholds based on variance statistics.
    
    Args:
        variance_stats: Variance metadata from compute_variance_statistics
        safety_margin: Multiplier for variance-based threshold
        min_rel_l2: Minimum relative L2 threshold
        
    Returns:
        dict: Mapping from stage_name -> {"max_abs": X, "rel_l2": Y}
    """
    thresholds = {}
    
    for stage_name, stats in variance_stats.items():
        # Use max of different variance metrics
        variance_metric = max(
            stats["max_abs_deviation"],
            stats["rms_deviation"] * 1.5,  # RMS often underestimates extremes
            stats["percentile_95_deviation"],
        )
        
        # Absolute threshold: variance × safety margin
        max_abs_threshold = variance_metric * safety_margin
        
        # Also consider magnitude-based threshold (1.5% of tensor RMS)
        magnitude_threshold = stats["tensor_rms"] * 0.015
        
        # Use the larger of variance-based and magnitude-based
        final_max_abs = max(max_abs_threshold, magnitude_threshold)
        
        # Ensure minimum threshold (avoid division by zero issues)
        final_max_abs = max(final_max_abs, 1e-6)
        
        thresholds[stage_name] = {
            "max_abs": final_max_abs,
            "rel_l2": min_rel_l2,
            "variance_metric": variance_metric,
            "magnitude_threshold": magnitude_threshold,
            "safety_margin": safety_margin,
        }
        
        if verbose and "ATTENTION" in stage_name:
            print(f"{stage_name}: variance={variance_metric:.6e}, "
                  f"magnitude={magnitude_threshold:.6e}, "
                  f"final_max_abs={final_max_abs:.6e}")
    
    return thresholds


def save_snapshots_and_thresholds(mean_snapshots: dict,
                                  variance_stats: dict,
                                  thresholds: dict,
                                  output_dir: Path,
                                  verbose: bool = False):
    """
    Save mean snapshots as .npy files and metadata as JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual snapshot files
    for stage_name, tensor in mean_snapshots.items():
        stage_str = stage_key_to_str(stage_name)
        snapshot_path = output_dir / f"{stage_str}.npy"
        np.save(snapshot_path, tensor)
        if verbose:
            print(f"Saved {stage_str}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    # Convert keys to strings for JSON serialization
    variance_stats_serializable = {
        stage_key_to_str(k): v for k, v in variance_stats.items()
    }
    thresholds_serializable = {
        stage_key_to_str(k): v for k, v in thresholds.items()
    }
    
    # Save variance statistics
    variance_path = output_dir / "variance_statistics.json"
    with open(variance_path, 'w') as f:
        json.dump(variance_stats_serializable, f, indent=2)
    print(f"Saved variance statistics to {variance_path}")
    
    # Save dynamic thresholds
    thresholds_path = output_dir / "dynamic_thresholds.json"
    with open(thresholds_path, 'w') as f:
        json.dump(thresholds_serializable, f, indent=2)
    print(f"Saved dynamic thresholds to {thresholds_path}")
    
    # Generate summary report
    summary_path = output_dir / "threshold_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DYNAMIC THRESHOLD SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Use serializable thresholds (already have string keys)
        # Group by stage type
        stage_groups = {}
        for stage_name in sorted(thresholds_serializable.keys()):
            stage_type = stage_name.rsplit('_', 1)[0] if '_' in stage_name else stage_name
            if stage_type not in stage_groups:
                stage_groups[stage_type] = []
            stage_groups[stage_type].append(stage_name)
        
        for stage_type, stages in sorted(stage_groups.items()):
            # Compute statistics across this group
            max_abs_values = [thresholds_serializable[s]["max_abs"] for s in stages]
            variance_values = [thresholds_serializable[s]["variance_metric"] for s in stages]
            
            avg_max_abs = np.mean(max_abs_values)
            avg_variance = np.mean(variance_values)
            
            f.write(f"{stage_type:30s} (n={len(stages):2d})  ")
            f.write(f"max_abs={avg_max_abs:10.6e}  ")
            f.write(f"variance={avg_variance:10.6e}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED THRESHOLDS\n")
        f.write("=" * 80 + "\n\n")
        
        for stage_name in sorted(thresholds_serializable.keys()):
            t = thresholds_serializable[stage_name]
            s = variance_stats_serializable[stage_name]
            f.write(f"{stage_name}:\n")
            f.write(f"  max_abs_threshold: {t['max_abs']:.6e}\n")
            f.write(f"  rel_l2_threshold:  {t['rel_l2']:.6f}\n")
            f.write(f"  variance_metric:   {t['variance_metric']:.6e}\n")
            f.write(f"  magnitude_metric:  {t['magnitude_threshold']:.6e}\n")
            f.write(f"  tensor_rms:        {s['tensor_rms']:.6e}\n")
            f.write(f"  max_abs_deviation: {s['max_abs_deviation']:.6e}\n")
            f.write("\n")
    
    print(f"Saved threshold summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate variance-based thresholds for parity tests"
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to GGUF model file"
    )
    parser.add_argument(
        "--tokens",
        required=True,
        help="Comma-separated token IDs (e.g., '1,2,3,4,5')"
    )
    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Output directory for snapshots and thresholds"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of PyTorch runs for variance measurement (default: 3)"
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=5.0,
        help="Safety margin multiplier for variance-based thresholds (default: 5.0)"
    )
    parser.add_argument(
        "--min-rel-l2",
        type=float,
        default=0.05,
        help="Minimum relative L2 threshold (default: 0.05)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Parse tokens
    tokens = [int(t.strip()) for t in args.tokens.split(',')]
    output_dir = Path(args.output_dir)
    
    print(f"=" * 80)
    print(f"GENERATING VARIANCE-BASED THRESHOLDS")
    print(f"=" * 80)
    print(f"Model:         {args.model}")
    print(f"Tokens:        {tokens}")
    print(f"Num runs:      {args.num_runs}")
    print(f"Safety margin: {args.safety_margin}")
    print(f"Output dir:    {output_dir}")
    print(f"=" * 80)
    print()
    
    # Run PyTorch multiple times
    runs = []
    for i in range(args.num_runs):
        print(f"Run {i+1}/{args.num_runs}...")
        snapshots = run_pytorch_inference(args.model, tokens, verbose=args.verbose)
        runs.append(snapshots)
        print(f"  Captured {len(snapshots)} snapshots")
    
    print()
    print(f"Computing variance statistics across {len(runs)} runs...")
    mean_snapshots, variance_stats = compute_variance_statistics(runs, verbose=args.verbose)
    
    print()
    print(f"Computing dynamic thresholds (safety_margin={args.safety_margin})...")
    thresholds = compute_dynamic_thresholds(
        variance_stats,
        safety_margin=args.safety_margin,
        min_rel_l2=args.min_rel_l2,
        verbose=args.verbose
    )
    
    print()
    print(f"Saving results to {output_dir}...")
    save_snapshots_and_thresholds(
        mean_snapshots,
        variance_stats,
        thresholds,
        output_dir,
        verbose=args.verbose
    )
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Generated {len(mean_snapshots)} reference snapshots")
    print(f"Generated {len(thresholds)} dynamic thresholds")
    print(f"Variance measured across {args.num_runs} runs")
    print(f"Safety margin: {args.safety_margin}x")
    print()
    print("Output files:")
    print(f"  - {len(mean_snapshots)} .npy snapshot files")
    print(f"  - variance_statistics.json (variance metrics)")
    print(f"  - dynamic_thresholds.json (for C++ tests)")
    print(f"  - threshold_summary.txt (human-readable)")
    print("=" * 80)


if __name__ == "__main__":
    main()
