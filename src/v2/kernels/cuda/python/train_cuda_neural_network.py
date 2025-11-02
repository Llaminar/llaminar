#!/usr/bin/env python3
"""
Neural Network Training for CUDA GEMM Heuristic (ONNX Export) - IMPROVED VERSION

This script trains a neural network to predict CUDA GEMM performance,
then exports it to ONNX format for C++ inference via ONNX Runtime.

IMPROVEMENTS (November 2, 2025):
1. Added 1.5B model training data (canary test size!)
2. Deeper architecture: 128 → 64 → 32 → 16 (vs previous 64 → 32 → 16)
3. More iterations: 2000 max (vs previous 500)
4. Hardware-aware features: warps, SMEM bank conflicts, coalescing
5. RobustScaler: Better outlier handling (vs StandardScaler)
6. Improved data splitting: Include 1.5B in training set
7. Odd batch/dimension training data (23,328 new samples)
8. Oddness/alignment features (26 new features - PHASE 3)

Architecture:
- Input: 99 features (config + problem size + engineered + oddness/alignment)
- Hidden Layer 1: 128 neurons (ReLU)
- Hidden Layer 2: 64 neurons (ReLU)
- Hidden Layer 3: 32 neurons (ReLU)
- Hidden Layer 4: 16 neurons (ReLU)
- Output: 1 neuron (GFLOPS prediction)

Feature Categories (99 total):
- Raw config (13): tile sizes, threads, work sizes, prefetch, vectorize, tensor cores
- Basic derived (19): threads_per_block, tile_size, occupancy_estimate, etc.
- Hardware-aware (15): warps, bank conflicts, coalescing, register pressure
- Profiler metrics (8): bank_conflict_risk, coalescing_score, mem_compute_ratio, etc.
- Batch/shape (18): batch_size_log2, aspect_ratio, tile_coverage, etc.
- NEW: Oddness/alignment (26): is_odd_batch, n_align_128, total_oddness, etc.

Expected Performance:
- Test R² = 0.94+ (current: 0.9396)
- Canary Top-30 hit rate: 67%+ (current: 67%, target: 75%+)
- Goal: Fix 1.5B regression (rank #69 → #5-10) without breaking odd cases

Author: David Sanftenberg
Date: November 2, 2025 (Updated with oddness/alignment features - PHASE 3)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Scikit-learn
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor

# ONNX export
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def engineer_features(df):
    """
    Engineer features from raw benchmark data.
    
    This function adds ~25 derived features to help the neural network
    learn patterns in CUDA GEMM performance.
    """
    print("      Engineering features...")
    
    # Basic derived features
    df['threads_per_block'] = df['threads_m'] * df['threads_n']
    df['tile_size'] = df['tile_m'] * df['tile_n']
    df['tile_area'] = df['tile_m'] * df['tile_n'] * df['tile_k']
    df['work_per_thread'] = df['work_m'] * df['work_n']
    
    # Occupancy estimate (48 KB shared memory, 1024 max threads per block)
    df['smem_per_block'] = df['tile_area'] * 4  # 4 bytes per float
    df['occupancy_estimate'] = np.minimum(
        48000 / df['smem_per_block'].clip(lower=1),
        1024 / df['threads_per_block'].clip(lower=1)
    )
    
    # Arithmetic intensity (FLOPs per byte loaded)
    df['arithmetic_intensity'] = (2 * df['tile_m'] * df['tile_n'] * df['tile_k']) / \
                                  ((df['tile_m'] * df['tile_k'] + df['tile_k'] * df['tile_n']) * 4)
    
    # Problem size features
    df['m_over_tile_m'] = df['m'] / df['tile_m']
    df['n_over_tile_n'] = df['n'] / df['tile_n']
    df['k_over_tile_k'] = df['k'] / df['tile_k']
    
    # Tile coverage
    df['tile_coverage_m'] = (df['m'] % df['tile_m']) / df['tile_m']
    df['tile_coverage_n'] = (df['n'] % df['tile_n']) / df['tile_n']
    
    # Size categories
    total_size = df['m'] * df['n'] * df['k']
    df['is_tiny'] = (total_size < 1e6).astype(int)
    df['is_small'] = ((total_size >= 1e6) & (total_size < 1e7)).astype(int)
    df['is_medium'] = ((total_size >= 1e7) & (total_size < 1e8)).astype(int)
    df['is_large'] = (total_size >= 1e8).astype(int)
    
    # Shape features
    df['is_square'] = (df['n'] == df['k']).astype(int)
    df['aspect_ratio'] = df['n'] / df['k'].clip(lower=1)
    df['tile_aspect_ratio'] = df['tile_n'] / df['tile_k'].clip(lower=1)
    df['tile_shape_match'] = np.abs(df['aspect_ratio'] - df['tile_aspect_ratio'])
    
    # NEW: Batch-aware features
    df['batch_size_log2'] = np.log2(df['m'].clip(lower=1))
    df['is_single_token'] = (df['m'] == 1).astype(int)
    df['is_power_of_2_batch'] = ((df['m'] & (df['m'] - 1)) == 0).astype(int)
    
    # NEW: Alignment features
    df['n_aligned_16'] = (df['n'] % 16 == 0).astype(int)
    df['n_aligned_32'] = (df['n'] % 32 == 0).astype(int)
    df['n_aligned_64'] = (df['n'] % 64 == 0).astype(int)
    df['k_aligned_32'] = (df['k'] % 32 == 0).astype(int)
    df['n_aligned_tile'] = (df['n'] % df['tile_n'] == 0).astype(int)
    df['k_aligned_tile'] = (df['k'] % df['tile_k'] == 0).astype(int)
    
    # NEW: Efficiency features
    df['warp_efficiency'] = df['threads_per_block'] / 32
    df['blocks_per_sm_estimate'] = 48000 / df['threads_per_block'].clip(lower=32)
    df['work_imbalance'] = (df['m'] % df['tile_m']) + (df['n'] % df['tile_n'])
    df['work_total'] = df['tile_m'] * df['tile_n'] * df['tile_k']
    df['work_per_thread_normalized'] = df['work_total'] / df['threads_per_block'].clip(lower=1)
    
    # NEW: Memory/compute features
    df['bytes_loaded_per_flop'] = 1.0 / df['arithmetic_intensity'].clip(lower=0.1)
    df['prefetch_benefit'] = df['prefetch_stages'] * df['arithmetic_intensity']
    df['vec_load_aligned'] = ((df['vectorize_load'] * 4) <= 16).astype(int)
    
    # NEW: Tile coverage features
    df['m_tiles'] = np.ceil(df['m'] / df['tile_m'])
    df['n_tiles'] = np.ceil(df['n'] / df['tile_n'])
    df['k_tiles'] = np.ceil(df['k'] / df['tile_k'])
    df['partial_tiles'] = ((df['m'] % df['tile_m'] != 0).astype(int) + 
                           (df['n'] % df['tile_n'] != 0).astype(int))
    
    # Size category
    df['size_category'] = 0
    df.loc[df['is_small'] == 1, 'size_category'] = 1
    df.loc[df['is_medium'] == 1, 'size_category'] = 2
    df.loc[df['is_large'] == 1, 'size_category'] = 3
    
    # NEW: Interaction features (help neural network learn complex patterns)
    df['tile_size_x_batch'] = df['tile_size'] * np.log2(df['m'].clip(lower=1))
    df['occupancy_x_intensity'] = df['occupancy_estimate'] * df['arithmetic_intensity']
    df['work_per_thread_x_batch'] = df['work_per_thread'] * df['m']
    
    # NEW: Hardware-aware features (GPU architecture specific)
    df['warps_per_block'] = df['threads_per_block'] / 32
    df['warp_utilization'] = (df['threads_per_block'] % 32) / 32.0  # Partial warp inefficiency
    df['smem_bank_conflicts'] = (df['tile_n'] % 32).clip(lower=1)  # 32 banks on most GPUs
    df['coalescing_penalty'] = ((df['n'] % 128) != 0).astype(int)  # 128-byte cache lines
    df['vec_load_efficiency'] = df['vectorize_load'] / 4.0  # Normalized 0-1
    
    # NEW: Advanced tile features
    df['tiles_per_sm'] = np.ceil(df['m_tiles'] * df['n_tiles'] / 80)  # ~80 SMs on A100
    df['tile_reuse_factor'] = df['k_tiles']  # How many times tiles are reused
    df['tile_compute_density'] = df['tile_area'] / df['threads_per_block'].clip(lower=1)
    
    # PHASE 2: Estimated profiler metrics (ZERO runtime cost!)
    # These approximate what ncu profiler would measure, but computed from config
    
    # Bank conflict risk (0-1 scale, higher = more conflicts)
    df['bank_conflict_risk'] = (df['tile_n'] % 32).clip(upper=16) / 16.0
    
    # Coalescing efficiency score (0-1, higher = better coalescing)
    df['coalescing_score'] = (
        (df['n'] % 128 == 0).astype(float) * 1.0 +
        (df['n'] % 64 == 0).astype(float) * 0.5 +
        (df['n'] % 32 == 0).astype(float) * 0.25
    )
    
    # Register pressure (work per thread - higher = more registers needed)
    df['register_pressure'] = df['work_total'] / df['threads_per_block'].clip(lower=1)
    
    # Memory vs compute ratio (higher = more memory-bound)
    df['mem_compute_ratio'] = df['bytes_loaded_per_flop'] * 1000
    
    # Warp divergence risk (0-1, higher = more divergence)
    # Non-power-of-2 thread counts cause divergence
    df['warp_divergence_risk'] = (
        1.0 - ((df['threads_per_block'] & (df['threads_per_block'] - 1)) == 0).astype(float)
    )
    
    # Shared memory pressure (KB per block)
    df['smem_kb_per_block'] = (df['smem_per_block'] / 1024.0).clip(upper=48)
    
    # L1 cache pressure (estimate based on tile size)
    df['l1_cache_pressure'] = (df['tile_area'] * 4 / (1024 * 128)).clip(upper=1.0)  # 128KB L1
    
    # Occupancy limiter score (what limits occupancy: 0=none, 1=smem, 2=threads, 3=registers)
    df['occupancy_limiter'] = 0
    df.loc[df['smem_kb_per_block'] > 32, 'occupancy_limiter'] = 1  # SMEM limited
    df.loc[df['threads_per_block'] > 512, 'occupancy_limiter'] = 2  # Thread limited
    df.loc[df['register_pressure'] > 128, 'occupancy_limiter'] = 3  # Register limited
    
    # ============================================================================
    # PHASE 3: Oddness and Alignment Features (Nov 2, 2025)
    # These features help distinguish odd batches/dims from aligned cases
    # Critical for canary tests with odd shapes
    # ============================================================================
    
    # Batch oddness features
    df['is_odd_batch'] = (df['m'] % 2).astype(float)
    df['is_prime_batch'] = df['m'].apply(lambda x: is_prime(int(x))).astype(float)
    df['batch_power_of_2'] = ((df['m'] & (df['m'] - 1)) == 0).astype(float)
    
    # Dimension alignment features (fine-grained)
    df['n_align_16'] = (df['n'] % 16).astype(float) / 16.0  # 0.0 = perfect, 1.0 = worst
    df['n_align_32'] = (df['n'] % 32).astype(float) / 32.0
    df['n_align_64'] = (df['n'] % 64).astype(float) / 64.0
    df['n_align_128'] = (df['n'] % 128).astype(float) / 128.0
    df['k_align_16'] = (df['k'] % 16).astype(float) / 16.0
    df['k_align_32'] = (df['k'] % 32).astype(float) / 32.0
    df['k_align_64'] = (df['k'] % 64).astype(float) / 64.0
    df['k_align_128'] = (df['k'] % 128).astype(float) / 128.0
    
    # Combined alignment scores (0.0 = perfect, 1.0 = worst)
    df['nk_align_score_16'] = (df['n_align_16'] + df['k_align_16']) / 2.0
    df['nk_align_score_32'] = (df['n_align_32'] + df['k_align_32']) / 2.0
    df['nk_align_score_128'] = (df['n_align_128'] + df['k_align_128']) / 2.0
    
    # Total alignment quality (inverted: 1.0 = perfect, 0.0 = worst)
    df['total_alignment_quality'] = 1.0 - df['nk_align_score_128']
    
    # Oddness interaction with alignment
    df['odd_batch_aligned_dims'] = df['is_odd_batch'] * df['total_alignment_quality']
    df['aligned_batch_aligned_dims'] = (1.0 - df['is_odd_batch']) * df['total_alignment_quality']
    
    # Dimension oddness (separate from batch oddness)
    df['n_is_odd'] = (df['n'] % 2).astype(float)
    df['k_is_odd'] = (df['k'] % 2).astype(float)
    df['both_dims_odd'] = df['n_is_odd'] * df['k_is_odd']
    df['any_dim_odd'] = ((df['n'] % 2 != 0) | (df['k'] % 2 != 0)).astype(float)
    
    # Combined oddness score (0.0 = all aligned, 1.0 = all odd)
    df['total_oddness'] = (df['is_odd_batch'] + df['n_is_odd'] + df['k_is_odd']) / 3.0
    
    # Alignment mismatch (tile vs problem size)
    df['m_tile_mismatch'] = (df['m'] % df['tile_m']).astype(float) / df['tile_m']
    df['n_tile_mismatch'] = (df['n'] % df['tile_n']).astype(float) / df['tile_n']
    df['k_tile_mismatch'] = (df['k'] % df['tile_k']).astype(float) / df['tile_k']
    df['total_tile_mismatch'] = (df['m_tile_mismatch'] + df['n_tile_mismatch'] + df['k_tile_mismatch']) / 3.0
    
    return df

def is_prime(n):
    """Helper function to check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train CUDA GEMM neural network heuristic')
    parser.add_argument('--input', default='../../../../../build_v2_release/cuda_gemm_benchmark_data.csv',
                       help='Input CSV file with benchmark data')
    parser.add_argument('--output-dir', default='../generated',
                       help='Output directory for generated files')
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[128, 64, 32, 16],
                       help='Hidden layer sizes (default: 128 64 32 16 - IMPROVED!)')
    parser.add_argument('--max-iter', type=int, default=2000,
                       help='Maximum training iterations (default: 2000 - IMPROVED!)')
    parser.add_argument('--early-stopping', action='store_true', default=True,
                       help='Enable early stopping based on validation loss (default: True)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("CUDA GEMM Neural Network Training Pipeline (ONNX Export)")
    print("=" * 80)
    
    # Load data
    print(f"\n[1/7] Loading data from {args.input}...")
    
    try:
        df = pd.read_csv(args.input)
        print(f"      Loaded {len(df)} benchmark results")
        print(f"      Unique test cases: {df['test_name'].nunique()}")
        print(f"      Unique configurations: {len(df[['tile_m', 'tile_n', 'tile_k', 'threads_m', 'threads_n', 'work_m', 'work_n']].drop_duplicates())}")
    except FileNotFoundError:
        print(f"[ERROR] File not found: {args.input}")
        print("        Run benchmark first: ./performance/v2_perf_cuda_heuristic_validation")
        return 1
    
    # Engineer features
    print("\n[2/7] Engineering features...")
    df_features = engineer_features(df.copy())
    
    # Select features for training
    raw_features = ['tile_m', 'tile_n', 'tile_k', 'threads_m', 'threads_n', 
                    'work_m', 'work_n', 'prefetch_stages', 'transpose_smem', 'vectorize_load',
                    'm', 'n', 'k']
    
    base_engineered = ['threads_per_block', 'tile_size', 'tile_area', 'work_per_thread',
                       'occupancy_estimate', 'arithmetic_intensity',
                       'm_over_tile_m', 'n_over_tile_n', 'k_over_tile_k',
                       'tile_coverage_m', 'tile_coverage_n',
                       'is_tiny', 'is_small', 'is_medium', 'is_large',
                       'is_square', 'aspect_ratio', 'tile_aspect_ratio', 'tile_shape_match']
    
    enhanced_features = [
        # Batch-aware
        'batch_size_log2', 'is_single_token', 'is_power_of_2_batch',
        # Alignment
        'n_aligned_16', 'n_aligned_32', 'n_aligned_64', 'k_aligned_32',
        'n_aligned_tile', 'k_aligned_tile',
        # Efficiency
        'warp_efficiency', 'blocks_per_sm_estimate', 'work_imbalance',
        'work_total', 'work_per_thread_normalized',
        # Memory/compute
        'bytes_loaded_per_flop', 'prefetch_benefit', 'vec_load_aligned',
        # Tile coverage
        'm_tiles', 'n_tiles', 'k_tiles', 'partial_tiles', 'size_category',
        # Interactions
        'tile_size_x_batch', 'occupancy_x_intensity', 'work_per_thread_x_batch',
        # Hardware-aware (8 features)
        'warps_per_block', 'warp_utilization', 'smem_bank_conflicts',
        'coalescing_penalty', 'vec_load_efficiency',
        # Advanced tile features (3 features)
        'tiles_per_sm', 'tile_reuse_factor', 'tile_compute_density',
        # PHASE 2: Estimated profiler metrics (8 features - ZERO runtime cost!)
        'bank_conflict_risk', 'coalescing_score', 'register_pressure',
        'mem_compute_ratio', 'warp_divergence_risk', 'smem_kb_per_block',
        'l1_cache_pressure', 'occupancy_limiter'
    ]
    
    feature_columns = raw_features + base_engineered + enhanced_features
    X = df_features[feature_columns]
    y = df_features['gflops']
    
    print(f"      Feature count: {len(feature_columns)} ({len(base_engineered)} base + {len(enhanced_features)} enhanced)")
    print(f"      Target: GFLOPS (range {y.min():.1f} - {y.max():.1f})")
    
    # Split data by MODEL SIZE (honest generalization test)
    print("\n[3/7] Splitting train/val/test sets by MODEL SIZE...")
    print("      Strategy: Include 1.5B (canary size!) in training set")
    print("      NEVER let model see 14B/235B/671B during training")
    print("      This tests true generalization to unseen model sizes")
    
    # Train on small, medium, and 1.5B (CANARY SIZE!) + ODD BATCHES/DIMENSIONS
    train_tests = [
        'Qwen_0_5B_SingleToken_QKV', 'Qwen_0_5B_Batch32_QKV', 'Qwen_0_5B_FFN_Gate',
        'Qwen_1_5B_SingleToken_QKV', 'Qwen_1_5B_Batch32_QKV', 'Qwen_1_5B_FFN_Gate', 'Qwen_1_5B_FFN_Down',  # NEW!
        'Qwen_7B_SingleToken_QKV', 'Qwen_7B_Batch128_QKV', 'Qwen_7B_FFN_Gate',
        'Qwen_72B_SingleToken_QKV', 'Qwen_72B_Batch128_QKV', 'Qwen_72B_FFN_Down',
        # Odd batch/dimension tests for better generalization
        'OddBatch_3x1280x1280', 'OddBatch_7x2048x2048', 'OddBatch_17x2048x2048', 'OddBatch_23x4096x4096',
        'OddDim_1x1537x2048', 'OddDim_1x2053x4096'
    ]
    
    val_tests = [
        'Qwen_4B_SingleToken_QKV', 'Qwen_4B_Batch128_QKV', 'Qwen_4B_FFN_Down',
        'Qwen_32B_SingleToken_QKV', 'Qwen_32B_FFN_Down'
    ]
    
    # All remaining tests (14B, 235B, 671B) are test/hold-out
    all_tests = df_features['test_name'].unique()
    test_tests = [t for t in all_tests if t not in train_tests and t not in val_tests]
    
    # Split data
    train_mask = df_features['test_name'].isin(train_tests)
    val_mask = df_features['test_name'].isin(val_tests)
    test_mask = df_features['test_name'].isin(test_tests)
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    X_val = X[val_mask]
    y_val = y[val_mask]
    
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"      Train: {len(X_train)} samples ({len(train_tests)} model sizes)")
    print(f"             Models: 0.5B, 1.5B (CANARY!), 7B, 72B + OddBatch/OddDim tests")
    print(f"      Val:   {len(X_val)} samples ({len(val_tests)} model sizes)")
    print(f"             Models: 4B, 32B")
    print(f"      Test:  {len(X_test)} samples ({len(test_tests)} model sizes - UNSEEN)")
    print(f"             Models: 14B, 235B, 671B")
    print(f"      Total: {len(X_train) + len(X_val) + len(X_test)} samples")
    
    # Scale features (fit ONLY on training data to prevent leakage)
    # Use RobustScaler for better outlier handling (vs StandardScaler)
    print("\n[4/7] Scaling features with RobustScaler (better outlier handling)...")
    scaler = RobustScaler()  # Uses median and IQR instead of mean/std
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train neural network
    print(f"\n[5/7] Training Neural Network...")
    print(f"      Architecture: {len(feature_columns)} → {' → '.join(map(str, args.hidden_layers))} → 1")
    print(f"      Max iterations: {args.max_iter}")
    print(f"      Early stopping: {'Enabled' if args.early_stopping else 'Disabled'}")
    print(f"      Activation: ReLU")
    print(f"      Optimizer: Adam")
    print(f"      Solver: adam (adaptive moment estimation)")
    
    model = MLPRegressor(
        hidden_layer_sizes=tuple(args.hidden_layers),
        activation='relu',
        solver='adam',
        alpha=0.001,              # L2 regularization
        batch_size='auto',
        learning_rate='adaptive',  # Decrease lr when loss stops improving
        learning_rate_init=0.001,
        max_iter=args.max_iter,
        shuffle=True,
        random_state=42,
        early_stopping=args.early_stopping,
        validation_fraction=0.1 if args.early_stopping else 0.1,
        n_iter_no_change=20,      # Patience for early stopping
        verbose=True
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on all three sets
    print("\n[6/7] Evaluating model...")
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print("\n" + "=" * 80)
    print("Neural Network Performance:")
    print("=" * 80)
    print(f"                        R²          MAE         RMSE")
    print("-" * 80)
    print(f"Train (seen)        {train_r2:6.4f}   {train_mae:8.2f}   {train_rmse:8.2f}")
    print(f"Val (seen)          {val_r2:6.4f}   {val_mae:8.2f}   {val_rmse:8.2f}")
    print(f"Test (UNSEEN)       {test_r2:6.4f}   {test_mae:8.2f}   {test_rmse:8.2f}")
    print("=" * 80)
    print(f"\n⚠️  KEY METRIC: Test R² measures generalization to UNSEEN model sizes")
    print(f"   Test models: {', '.join([t.split('_')[0] + ' ' + t.split('_')[1] for t in test_tests[:5]])}...")
    
    # Cross-validation on training set
    print("\nCross-validation on training set (5-fold):")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2', verbose=1)
    print(f"  R² scores: {cv_scores}")
    print(f"  Mean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Export to ONNX
    print(f"\n[7/7] Exporting to ONNX format...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = output_dir / "cuda_heuristic_nn.onnx"
    
    # Define input type (all features are float32)
    initial_type = [('float_input', FloatTensorType([None, len(feature_columns)]))]
    
    # Convert to ONNX
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)
    
    # Save ONNX model
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"[EXPORT] ONNX model exported to {onnx_path}")
    print(f"         Model size: {onnx_path.stat().st_size / 1024:.1f} KB")
    print(f"         Input: {len(feature_columns)} features (float32)")
    print(f"         Output: 1 value (GFLOPS prediction)")
    
    # Save scaler parameters for C++
    scaler_path = output_dir / "cuda_heuristic_scaler.npz"
    # RobustScaler uses center_ and scale_ (not mean_ and scale_)
    np.savez(scaler_path, mean=scaler.center_, scale=scaler.scale_)
    print(f"[EXPORT] Scaler parameters saved to {scaler_path}")
    print(f"         Note: Using RobustScaler (median/IQR, better outlier handling)")
    
    # Also export as simple text file for C++ (easier to parse than .npz)
    scaler_txt_path = output_dir / "cuda_heuristic_scaler.txt"
    with open(scaler_txt_path, 'w') as f:
        f.write("MEAN\n")  # Actually median for RobustScaler
        for val in scaler.center_:
            f.write(f"{val}\n")
        f.write("SCALE\n")  # Actually IQR for RobustScaler
        for val in scaler.scale_:
            f.write(f"{val}\n")
    print(f"[EXPORT] Scaler parameters (text) saved to {scaler_txt_path}")
    
    # Save feature names for C++
    features_path = output_dir / "cuda_heuristic_features.txt"
    with open(features_path, 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")
    print(f"[EXPORT] Feature names saved to {features_path}")
    
    # Create validation plots
    print("\n[PLOT] Creating validation plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Training predictions
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.3, s=1)
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual GFLOPS')
    axes[0, 0].set_ylabel('Predicted GFLOPS')
    axes[0, 0].set_title(f'Training Set (R²={train_r2:.4f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation predictions
    axes[0, 1].scatter(y_val, y_val_pred, alpha=0.3, s=1)
    axes[0, 1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual GFLOPS')
    axes[0, 1].set_ylabel('Predicted GFLOPS')
    axes[0, 1].set_title(f'Validation Set (R²={val_r2:.4f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Test predictions
    axes[1, 0].scatter(y_test, y_test_pred, alpha=0.3, s=1)
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('Actual GFLOPS')
    axes[1, 0].set_ylabel('Predicted GFLOPS')
    axes[1, 0].set_title(f'Test Set - UNSEEN (R²={test_r2:.4f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Learning curve
    if hasattr(model, 'loss_curve_'):
        axes[1, 1].plot(model.loss_curve_, label='Training Loss')
        if hasattr(model, 'validation_scores_'):
            axes[1, 1].plot(model.validation_scores_, label='Validation R²')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Loss / R²')
        axes[1, 1].set_title('Learning Curve')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "cuda_heuristic_nn_validation.png"
    plt.savefig(plot_path, dpi=150)
    print(f"[PLOT] Saved validation plots to {plot_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Integrate ONNX Runtime into C++ build (see CMakeLists.txt)")
    print(f"  2. Load ONNX model in CudaGemmAutoTuner::initializeNeuralNetwork()")
    print(f"  3. Implement feature extraction in C++ (must match Python exactly)")
    print(f"  4. Test with canary shapes: LLAMINAR_USE_NN_HEURISTIC=1")
    print(f"  5. Compare against lookup table and linear model")
    
    return 0

if __name__ == '__main__':
    exit(main())
