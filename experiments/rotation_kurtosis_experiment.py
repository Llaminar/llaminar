#!/usr/bin/env python3
"""
Rotation-based Kurtosis Reduction Experiment for Q8_1 Quantization

Tests whether applying an orthogonal rotation (TurboQuant-style) before int8
quantisation can reduce activation kurtosis and improve quantisation fidelity.

Hypothesis: High-kurtosis activations (kurtosis >100) concentrate energy in a
few outlier dimensions. An orthogonal rotation spreads this energy uniformly
across all dimensions, reducing kurtosis toward ~0 (Gaussian). This transforms
the quantisation problem from "a few dimensions dominate the absmax" to
"all dimensions contribute equally", improving int8 round-trip accuracy.

Usage:
    python3 experiments/rotation_kurtosis_experiment.py
"""

import numpy as np
from pathlib import Path
import sys

SNAPSHOT_DIR = Path("pytorch_qwen35_4b_snapshots")
SNAPSHOT_DIR_08B = Path("pytorch_qwen35_snapshots")

# ── Rotation matrix generation (matches TurboQuantRotation.h exactly) ──

def generate_rotation_matrix(dim: int, seed: int = 31) -> np.ndarray:
    """Generate a deterministic d×d orthogonal rotation matrix via QR.
    
    Matches the C++ Modified Gram-Schmidt implementation in TurboQuantRotation.h.
    Uses numpy's QR for numerical stability (equivalent result).
    """
    rng = np.random.RandomState(seed)
    A = rng.randn(dim, dim).astype(np.float32)
    Q, R = np.linalg.qr(A)
    # Fix sign ambiguity (Haar measure): ensure R[i,i] > 0
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs[np.newaxis, :]
    return Q.astype(np.float32)


# ── Q8_1 quantization simulation ──

def quantize_q8_symmetric(x: np.ndarray, block_size: int = 32) -> np.ndarray:
    """Simulate per-block symmetric int8 quantization (Q8_1-style).
    
    For each block of `block_size` elements:
        scale = max(|x|) / 127
        quantized = round(x / scale)
        dequantized = quantized * scale
    """
    flat = x.flatten()
    n = len(flat)
    # Pad to multiple of block_size
    pad = (block_size - n % block_size) % block_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    
    blocks = flat.reshape(-1, block_size)
    scales = np.max(np.abs(blocks), axis=1, keepdims=True) / 127.0
    scales = np.maximum(scales, 1e-10)  # avoid divide by zero
    
    quantized = np.clip(np.round(blocks / scales), -128, 127)
    dequantized = quantized * scales
    
    return dequantized.flatten()[:n].reshape(x.shape)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flat arrays."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(dot / (norm_a * norm_b))


def excess_kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis (Fisher's definition): E[(x-μ)⁴]/σ⁴ - 3."""
    flat = x.flatten().astype(np.float64)
    mean = np.mean(flat)
    var = np.var(flat)
    if var < 1e-30:
        return 0.0
    m4 = np.mean((flat - mean) ** 4)
    return float(m4 / (var ** 2) - 3.0)


def max_over_std(x: np.ndarray) -> float:
    """max(|x|) / std(x) — ratio that determines int8 dynamic range waste."""
    flat = x.flatten().astype(np.float64)
    s = np.std(flat)
    if s < 1e-30:
        return 0.0
    return float(np.max(np.abs(flat)) / s)


# ── Per-head rotation experiment ──

def rotate_per_head(x: np.ndarray, head_dim: int, rotation: np.ndarray) -> np.ndarray:
    """Apply orthogonal rotation to each head-sized chunk of x.
    
    x shape: (..., hidden_dim) where hidden_dim = n_heads * head_dim
    rotation: (head_dim, head_dim) orthogonal matrix
    
    Returns: rotated x with same shape
    """
    shape = x.shape
    hidden_dim = shape[-1]
    assert hidden_dim % head_dim == 0
    n_heads = hidden_dim // head_dim
    
    # Reshape to (..., n_heads, head_dim)
    reshaped = x.reshape(-1, n_heads, head_dim)
    # Apply rotation: each head gets rotated independently
    # rotated[i, h, :] = rotation @ reshaped[i, h, :]
    rotated = np.einsum('ij,...j->...i', rotation, reshaped)
    return rotated.reshape(shape)


def inverse_rotate_per_head(x: np.ndarray, head_dim: int, rotation: np.ndarray) -> np.ndarray:
    """Apply inverse rotation (transpose) to each head-sized chunk."""
    shape = x.shape
    hidden_dim = shape[-1]
    n_heads = hidden_dim // head_dim
    reshaped = x.reshape(-1, n_heads, head_dim)
    # Inverse rotation = transpose: x = R^T @ y
    inv_rotated = np.einsum('ji,...j->...i', rotation, reshaped)
    return inv_rotated.reshape(shape)


# ── Full-dimension rotation experiment ──

def rotate_full(x: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply full hidden_dim × hidden_dim rotation."""
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    rotated = flat @ rotation.T
    return rotated.reshape(shape)


def inverse_rotate_full(x: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply inverse full rotation."""
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    inv_rotated = flat @ rotation
    return inv_rotated.reshape(shape)


# ── Main experiment ──

def run_layer_experiment(data: np.ndarray, layer_idx: int, hidden_dim: int):
    """Run the rotation + quantization experiment on one layer's activations."""
    
    flat_data = data.reshape(-1, hidden_dim)  # (seq_len, hidden_dim)
    
    # ── Baseline: raw Q8_1 round-trip ──
    dq_raw = quantize_q8_symmetric(flat_data)
    cos_raw = cosine_similarity(flat_data, dq_raw)
    kurt_raw = excess_kurtosis(flat_data)
    maxstd_raw = max_over_std(flat_data)
    
    results = {
        'layer': layer_idx,
        'kurtosis_raw': kurt_raw,
        'max_std_raw': maxstd_raw,
        'cosine_q8_raw': cos_raw,
    }
    
    # ── Per-head rotation experiments (head_dim = 64, 128, 256) ──
    for head_dim in [64, 128, 256]:
        if hidden_dim % head_dim != 0:
            continue
        
        rot = generate_rotation_matrix(head_dim, seed=31)
        
        # Rotate
        rotated = rotate_per_head(flat_data, head_dim, rot)
        kurt_rot = excess_kurtosis(rotated)
        maxstd_rot = max_over_std(rotated)
        
        # Q8_1 round-trip in rotated space, then inverse rotate
        dq_rotated = quantize_q8_symmetric(rotated)
        reconstructed = inverse_rotate_per_head(dq_rotated, head_dim, rot)
        cos_rotated = cosine_similarity(flat_data, reconstructed)
        
        results[f'kurtosis_rot_h{head_dim}'] = kurt_rot
        results[f'max_std_rot_h{head_dim}'] = maxstd_rot
        results[f'cosine_q8_rot_h{head_dim}'] = cos_rotated
    
    # ── Full-dimension rotation (hidden_dim × hidden_dim) ──
    if hidden_dim <= 4096:  # Don't generate huge rotation matrices
        rot_full = generate_rotation_matrix(hidden_dim, seed=42)
        rotated_full = rotate_full(flat_data, rot_full)
        kurt_full = excess_kurtosis(rotated_full)
        maxstd_full = max_over_std(rotated_full)
        
        dq_full = quantize_q8_symmetric(rotated_full)
        recon_full = inverse_rotate_full(dq_full, rot_full)
        cos_full = cosine_similarity(flat_data, recon_full)
        
        results['kurtosis_rot_full'] = kurt_full
        results['max_std_rot_full'] = maxstd_full
        results['cosine_q8_rot_full'] = cos_full
    
    return results


def print_experiment(snapshot_dir: Path, model_name: str, n_layers: int, hidden_dim: int):
    """Run and print the experiment for all layers of a model."""
    
    print(f"\n{'='*100}")
    print(f"  ROTATION KURTOSIS REDUCTION EXPERIMENT — {model_name}")
    print(f"  hidden_dim={hidden_dim}, layers={n_layers}")
    print(f"{'='*100}\n")
    
    # Header
    print(f"{'Layer':>7} │ {'Kurt(raw)':>10} │ {'Kurt(h64)':>10} │ {'Kurt(h128)':>10} │ {'Kurt(full)':>10} │"
          f" {'Cos(raw)':>10} │ {'Cos(h64)':>10} │ {'Cos(h128)':>10} │ {'Cos(full)':>10} │ {'MaxStd(r)':>10} │ {'MaxStd(f)':>10}")
    print("─" * 140)
    
    all_results = []
    
    for layer_idx in range(n_layers):
        snapshot_path = snapshot_dir / f"layer{layer_idx}_ATTENTION_NORM.npy"
        if not snapshot_path.exists():
            continue
        
        data = np.load(str(snapshot_path))
        results = run_layer_experiment(data, layer_idx, hidden_dim)
        all_results.append(results)
        
        # Print row
        kurt_raw = results['kurtosis_raw']
        kurt_h64 = results.get('kurtosis_rot_h64', float('nan'))
        kurt_h128 = results.get('kurtosis_rot_h128', float('nan'))
        kurt_full = results.get('kurtosis_rot_full', float('nan'))
        cos_raw = results['cosine_q8_raw']
        cos_h64 = results.get('cosine_q8_rot_h64', float('nan'))
        cos_h128 = results.get('cosine_q8_rot_h128', float('nan'))
        cos_full = results.get('cosine_q8_rot_full', float('nan'))
        maxstd_raw = results['max_std_raw']
        maxstd_full = results.get('max_std_rot_full', float('nan'))
        
        print(f"  L{layer_idx:>3} │ {kurt_raw:>10.0f} │ {kurt_h64:>10.1f} │ {kurt_h128:>10.1f} │ {kurt_full:>10.1f} │"
              f" {cos_raw:>10.6f} │ {cos_h64:>10.6f} │ {cos_h128:>10.6f} │ {cos_full:>10.6f} │ {maxstd_raw:>10.1f} │ {maxstd_full:>10.1f}")
    
    if not all_results:
        print("  No snapshots found!")
        return
    
    # Summary
    print("─" * 140)
    avg_cos_raw = np.mean([r['cosine_q8_raw'] for r in all_results])
    avg_cos_h64 = np.mean([r.get('cosine_q8_rot_h64', r['cosine_q8_raw']) for r in all_results])
    avg_cos_h128 = np.mean([r.get('cosine_q8_rot_h128', r['cosine_q8_raw']) for r in all_results])
    avg_cos_full = np.mean([r.get('cosine_q8_rot_full', r['cosine_q8_raw']) for r in all_results])
    avg_kurt_raw = np.mean([r['kurtosis_raw'] for r in all_results])
    avg_kurt_full = np.mean([r.get('kurtosis_rot_full', r['kurtosis_raw']) for r in all_results])
    
    print(f"  {'AVG':>3} │ {avg_kurt_raw:>10.0f} │ {'':>10} │ {'':>10} │ {avg_kurt_full:>10.1f} │"
          f" {avg_cos_raw:>10.6f} │ {avg_cos_h64:>10.6f} │ {avg_cos_h128:>10.6f} │ {avg_cos_full:>10.6f} │")
    
    # Improvement summary
    print(f"\n  ═══ IMPROVEMENT SUMMARY ═══")
    print(f"  Average kurtosis: {avg_kurt_raw:.0f} → {avg_kurt_full:.1f} (full rotation)")
    print(f"  Average Q8 cosine: {avg_cos_raw:.6f} → {avg_cos_full:.6f} (full rotation)")
    print(f"  Cosine improvement: +{(avg_cos_full - avg_cos_raw)*1e6:.0f} ppm")
    
    # Also show per-head rotation improvements
    if any('cosine_q8_rot_h128' in r for r in all_results):
        print(f"  Average Q8 cosine: {avg_cos_raw:.6f} → {avg_cos_h128:.6f} (h128 rotation)")
        print(f"  Cosine improvement: +{(avg_cos_h128 - avg_cos_raw)*1e6:.0f} ppm")
    
    # Find worst-case improvements
    worst_raw = min(all_results, key=lambda r: r['cosine_q8_raw'])
    worst_full = min(all_results, key=lambda r: r.get('cosine_q8_rot_full', 1.0))
    print(f"\n  Worst-case raw cosine: L{worst_raw['layer']} = {worst_raw['cosine_q8_raw']:.6f}")
    if 'cosine_q8_rot_full' in worst_raw:
        print(f"  Same layer with full rotation: {worst_raw['cosine_q8_rot_full']:.6f}")


def main():
    # ── 4B model ──
    if SNAPSHOT_DIR.exists():
        print_experiment(SNAPSHOT_DIR, "Qwen3.5-4B (Q8_0)", n_layers=32, hidden_dim=2560)
    else:
        print(f"4B snapshots not found at {SNAPSHOT_DIR}")
    
    # ── 0.8B model for comparison ──
    if SNAPSHOT_DIR_08B.exists():
        print_experiment(SNAPSHOT_DIR_08B, "Qwen3.5-0.8B (Q4_0)", n_layers=24, hidden_dim=1024)
    else:
        print(f"0.8B snapshots not found at {SNAPSHOT_DIR_08B}")


if __name__ == "__main__":
    main()
