"""
Utility Functions for Reference Implementation

Helper functions for snapshot comparison, data conversion, and debugging.

@author David Sanftenberg
"""

from typing import Dict, Tuple, Optional
import numpy as np

from .pipeline_stages import PipelineStage, stage_to_string


def compare_snapshots(
    snapshot1: Dict[Tuple[PipelineStage, int], np.ndarray],
    snapshot2: Dict[Tuple[PipelineStage, int], np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> Dict[str, any]:
    """
    Compare two snapshot dictionaries and compute comparison metrics.
    
    Args:
        snapshot1: First snapshot dictionary
        snapshot2: Second snapshot dictionary
        rtol: Relative tolerance for allclose
        atol: Absolute tolerance for allclose
    
    Returns:
        Dictionary containing:
            - "matching_stages": List of (stage, layer) keys present in both
            - "only_in_1": Keys only in snapshot1
            - "only_in_2": Keys only in snapshot2
            - "metrics": Per-stage metrics (max_abs_diff, rel_l2, allclose)
            - "overall_match": Boolean indicating if all stages match
    """
    keys1 = set(snapshot1.keys())
    keys2 = set(snapshot2.keys())
    
    matching = sorted(keys1 & keys2)
    only_1 = sorted(keys1 - keys2)
    only_2 = sorted(keys2 - keys1)
    
    metrics = {}
    all_match = True
    
    for key in matching:
        arr1 = snapshot1[key]
        arr2 = snapshot2[key]
        
        # Check shapes
        if arr1.shape != arr2.shape:
            metrics[key] = {
                "shape_mismatch": True,
                "shape1": arr1.shape,
                "shape2": arr2.shape
            }
            all_match = False
            continue
        
        # Compute metrics
        diff = arr1 - arr2
        max_abs_diff = np.max(np.abs(diff))
        
        # Relative L2 norm
        norm1 = np.linalg.norm(arr1.flatten())
        norm2 = np.linalg.norm(arr2.flatten())
        norm_diff = np.linalg.norm(diff.flatten())
        rel_l2 = norm_diff / (max(norm1, norm2) + 1e-10)
        
        # Check allclose
        is_close = np.allclose(arr1, arr2, rtol=rtol, atol=atol)
        
        stage, layer = key
        metrics[key] = {
            "stage": stage_to_string(stage),
            "layer": layer,
            "shape": arr1.shape,
            "max_abs_diff": float(max_abs_diff),
            "rel_l2": float(rel_l2),
            "allclose": is_close
        }
        
        if not is_close:
            all_match = False
    
    return {
        "matching_stages": matching,
        "only_in_1": only_1,
        "only_in_2": only_2,
        "metrics": metrics,
        "overall_match": all_match
    }


def load_snapshots_npz(path: str) -> Dict[Tuple[PipelineStage, int], np.ndarray]:
    """
    Load snapshots from .npz file.
    
    Args:
        path: Path to .npz file
    
    Returns:
        Dictionary mapping (stage, layer_idx) to numpy arrays
    """
    from .pipeline_stages import string_to_stage
    
    data = np.load(path)
    snapshots = {}
    
    for key in data.files:
        # Parse key: "STAGE_NAME_layer_idx"
        parts = key.rsplit("_", 1)
        if len(parts) != 2:
            continue
        
        stage_name, layer_str = parts
        try:
            stage = string_to_stage(stage_name)
            layer_idx = int(layer_str)
            snapshots[(stage, layer_idx)] = data[key]
        except (KeyError, ValueError):
            continue
    
    return snapshots


def print_snapshot_summary(
    snapshots: Dict[Tuple[PipelineStage, int], np.ndarray],
    title: str = "Snapshot Summary"
) -> None:
    """
    Print a human-readable summary of captured snapshots.
    
    Args:
        snapshots: Snapshot dictionary
        title: Title for the summary
    """
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")
    print(f"Total snapshots: {len(snapshots)}\n")
    
    if not snapshots:
        print("No snapshots captured.")
        return
    
    # Group by stage
    by_stage = {}
    for (stage, layer_idx), array in snapshots.items():
        if stage not in by_stage:
            by_stage[stage] = []
        by_stage[stage].append((layer_idx, array))
    
    for stage in sorted(by_stage.keys(), key=lambda s: s.value):
        stage_name = stage_to_string(stage)
        layers = sorted(by_stage[stage])
        
        print(f"{stage_name}:")
        for layer_idx, array in layers:
            layer_str = f"layer {layer_idx}" if layer_idx >= 0 else "global"
            print(f"  {layer_str:12} : shape={str(array.shape):20} "
                  f"dtype={array.dtype} "
                  f"range=[{array.min():.4f}, {array.max():.4f}]")
        print()
    
    print(f"{'=' * 60}\n")


def tensor_stats(array: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics for a tensor.
    
    Args:
        array: NumPy array
    
    Returns:
        Dictionary with min, max, mean, std, norm
    """
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "norm": float(np.linalg.norm(array.flatten()))
    }
