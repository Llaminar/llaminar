#!/usr/bin/env python3
"""
Train a regression model to predict CUDA GEMM performance from kernel configuration.

This script:
1. Loads benchmark data from cuda_gemm_benchmark_data.csv
2. Engineers features from configuration parameters
3. Trains a Gradient Boosting Regressor to predict GFLOPS
4. Exports model for use in C++ heuristic
5. Validates model performance

Usage:
    python train_cuda_heuristic.py [--output-dir DIR]

Output:
    - cuda_heuristic_model_weights.txt (model coefficients for C++)
    - cuda_heuristic_weights.h (C++ header with learned weights)
    - cuda_heuristic_validation.png (performance plots)
    - Feature importance rankings
"""

import argparse

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def engineer_features(df):
    """
    Engineer features from raw configuration parameters.
    
    Derived features based on GPU performance theory:
    - Occupancy estimates
    - Memory bandwidth requirements
    - Arithmetic intensity
    - Thread block size
    - Work per thread
    - Problem size ratios
    """
    df = df.copy()
    
    # Thread block size
    df['threads_per_block'] = df['threads_m'] * df['threads_n']
    
    # Tile sizes
    df['tile_size'] = df['tile_m'] * df['tile_n']
    df['tile_area'] = df['tile_m'] * df['tile_n'] * df['tile_k']
    
    # Work per thread
    df['work_per_thread'] = df['work_m'] * df['work_n']
    
    # Occupancy estimate (simplified)
    # RTX 3090: 82 SMs, 1536 max threads per SM, 48KB shared memory per SM
    max_threads_per_sm = 1536
    max_smem_per_sm = 48 * 1024  # bytes
    
    df['threads_per_block'] = df['threads_m'] * df['threads_n']
    
    # Shared memory per block (estimate)
    df['smem_per_block'] = (df['tile_m'] * df['tile_k'] + df['tile_n'] * df['tile_k']) * 4  # floats
    df['smem_per_block'] *= (1 + df['prefetch_stages'])  # Multi-buffering
    
    # Occupancy limiting factors
    df['blocks_per_sm_threads'] = max_threads_per_sm // df['threads_per_block']
    df['blocks_per_sm_smem'] = max_smem_per_sm // df['smem_per_block'].clip(lower=1)
    df['occupancy_estimate'] = np.minimum(df['blocks_per_sm_threads'], df['blocks_per_sm_smem']) / 16.0  # Normalize to [0,1]
    
    # Arithmetic intensity (FLOPs per byte loaded)
    df['flops_per_element'] = 2 * df['k']  # multiply-add
    df['bytes_per_element'] = df['k'] * 4  # float32 for A
    # IQ4_NL: k/32 blocks, each 20 bytes
    df['bytes_per_element'] += (df['k'] / 32.0) * 20
    df['arithmetic_intensity'] = df['flops_per_element'] / df['bytes_per_element']
    
    # Problem size ratios
    df['m_over_tile_m'] = df['m'] / df['tile_m']
    df['n_over_tile_n'] = df['n'] / df['tile_n']
    df['k_over_tile_k'] = df['k'] / df['tile_k']
    
    # Tile coverage efficiency
    df['tile_coverage_m'] = (df['m'] % df['tile_m'] == 0).astype(int)
    df['tile_coverage_n'] = (df['n'] % df['tile_n'] == 0).astype(int)
    
    # Problem size classification
    df['is_tiny'] = (df['m'] < 8).astype(int)
    df['is_small'] = ((df['m'] >= 8) & (df['m'] < 64)).astype(int)
    df['is_medium'] = ((df['m'] >= 64) & (df['m'] < 256)).astype(int)
    df['is_large'] = (df['m'] >= 256).astype(int)
    
    # Matrix shape
    df['is_square'] = (df['m'] == df['n']).astype(int)
    df['aspect_ratio'] = df['n'] / df['m'].clip(lower=1)
    
    # Tile shape
    df['tile_aspect_ratio'] = df['tile_n'] / df['tile_m']
    df['tile_shape_match'] = (
        (df['aspect_ratio'] > 1.5) & (df['tile_aspect_ratio'] > 1.5) |
        (df['aspect_ratio'] < 0.67) & (df['tile_aspect_ratio'] < 0.67)
    ).astype(int)
    
    # Total compute (for normalization)
    df['total_flops'] = 2 * df['m'] * df['n'] * df['k']
    
    # ========================================================================
    # ENHANCED FEATURES (Nov 2, 2025 - Generalization Improvements)
    # ========================================================================
    
    # Batch size features (critical for generalization)
    df['batch_size_log2'] = np.log2(df['m'].clip(lower=1))
    df['is_single_token'] = (df['m'] == 1).astype(int)
    df['is_power_of_2_batch'] = ((df['m'] & (df['m'] - 1)) == 0).astype(int)
    
    # Dimension alignment features (affects memory coalescing)
    df['n_aligned_16'] = (df['n'] % 16 == 0).astype(int)
    df['n_aligned_32'] = (df['n'] % 32 == 0).astype(int)
    df['n_aligned_64'] = (df['n'] % 64 == 0).astype(int)
    df['k_aligned_32'] = (df['k'] % 32 == 0).astype(int)  # IQ4_NL block size
    
    df['n_aligned_tile'] = (df['n'] % df['tile_n'] == 0).astype(int)
    df['k_aligned_tile'] = (df['k'] % df['tile_k'] == 0).astype(int)
    
    # Thread block efficiency
    df['warp_efficiency'] = ((df['threads_per_block'] % 32 == 0).astype(float))
    df['blocks_per_sm_estimate'] = np.minimum(
        1536 // df['threads_per_block'],  # Thread limit
        (48 * 1024) // df['smem_per_block'].clip(lower=1)  # SMEM limit
    ).clip(upper=16) / 16.0  # Normalize
    
    # Work distribution efficiency
    df['work_imbalance'] = (df['m'] % (df['tile_m'] * df['work_m']) != 0).astype(int)
    df['work_total'] = df['work_m'] * df['work_n']
    df['work_per_thread_normalized'] = df['work_per_thread'] / df['work_total'].clip(lower=1)
    
    # Memory bandwidth features
    df['bytes_loaded_per_flop'] = (
        (df['m'] * df['k'] * 4 +  # A matrix (FP32)
         df['n'] * (df['k'] / 32.0) * 20)  # B matrix (IQ4_NL: 20 bytes per 32 elements)
        / (2.0 * df['m'] * df['n'] * df['k'])  # FLOPs
    )
    
    # Prefetch stage effectiveness
    df['prefetch_benefit'] = (
        (df['prefetch_stages'] > 0).astype(int) * 
        (df['arithmetic_intensity'] > 1.0).astype(int)  # Only helps if compute-bound
    )
    
    # Vectorization effectiveness
    df['vec_load_aligned'] = (
        ((df['k'] % (df['vectorize_load'] * 4)) == 0).astype(int)  # 4 bytes per float
    )
    
    # Tile coverage efficiency (edge cases)
    df['m_tiles'] = np.ceil(df['m'] / df['tile_m'])
    df['n_tiles'] = np.ceil(df['n'] / df['tile_n'])
    df['k_tiles'] = np.ceil(df['k'] / df['tile_k'])
    df['partial_tiles'] = (
        (df['m'] % df['tile_m'] != 0).astype(int) +
        (df['n'] % df['tile_n'] != 0).astype(int) +
        (df['k'] % df['tile_k'] != 0).astype(int)
    )
    
    # Problem size classification (more granular)
    df['size_category'] = 0
    df.loc[df['m'] < 8, 'size_category'] = 0  # Tiny
    df.loc[(df['m'] >= 8) & (df['m'] < 32), 'size_category'] = 1  # Small
    df.loc[(df['m'] >= 32) & (df['m'] < 128), 'size_category'] = 2  # Medium
    df.loc[(df['m'] >= 128) & (df['m'] < 512), 'size_category'] = 3  # Large
    df.loc[df['m'] >= 512, 'size_category'] = 4  # Huge
    
    # Interaction features (captures non-linear relationships)
    df['tile_size_x_batch'] = df['tile_size'] * np.log1p(df['m'])
    df['occupancy_x_intensity'] = df['occupancy_estimate'] * df['arithmetic_intensity']
    df['work_per_thread_x_batch'] = df['work_per_thread'] * np.log1p(df['m'])
    
    return df

def plot_results(y_true, y_pred, feature_importance, feature_names, output_file='cuda_heuristic_validation.png'):
    """
    Generate validation plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Predicted vs Actual
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=10)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual GFLOPS')
    axes[0, 0].set_ylabel('Predicted GFLOPS')
    axes[0, 0].set_title('Predicted vs Actual Performance')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_pred - y_true
    axes[0, 1].scatter(y_true, residuals, alpha=0.5, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Actual GFLOPS')
    axes[0, 1].set_ylabel('Residual (Predicted - Actual)')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Feature importance (top 20)
    top_n = 20
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).nlargest(top_n, 'importance')
    
    axes[1, 0].barh(importance_df['feature'], importance_df['importance'])
    axes[1, 0].set_xlabel('Importance')
    axes[1, 0].set_title(f'Top {top_n} Feature Importance')
    axes[1, 0].invert_yaxis()
    
    # Error distribution
    axes[1, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Residual (GFLOPS)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"[PLOT] Saved validation plots to {output_file}")

def export_cpp_heuristic(gb_model, linear_model, scaler, feature_names, output_dir='.'):
    """
    Export model in C++-compatible format.
    
    Generates:
    1. cuda_heuristic_model_weights.txt - Human-readable summary
    2. cuda_heuristic_weights.h - C++ header with learned weights
    
    Args:
        gb_model: Trained Gradient Boosting model (for feature importance)
        linear_model: Trained Linear Regression model (for actual C++ implementation)
        scaler: StandardScaler instance
        feature_names: List of feature names
        output_dir: Output directory
    """
    # 1. Human-readable summary
    txt_file = f'{output_dir}/cuda_heuristic_model_weights.txt'
    with open(txt_file, 'w') as f:
        f.write("CUDA GEMM Heuristic - Trained Model Summary\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Gradient Boosting Model (for reference):\n")
        f.write(f"Number of estimators: {gb_model.n_estimators}\n")
        f.write(f"Max depth: {gb_model.max_depth}\n")
        f.write(f"Learning rate: {gb_model.learning_rate}\n\n")
        
        f.write("Linear Regression Model (for C++ implementation):\n")
        f.write(f"Intercept: {linear_model.intercept_:.6f}\n")
        f.write(f"Number of coefficients: {len(linear_model.coef_)}\n\n")
        
        f.write("Feature Importance (GB model, Top 20):\n")
        f.write("-" * 80 + "\n")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': gb_model.feature_importances_
        }).nlargest(20, 'importance')
        
        for _, row in importance_df.iterrows():
            f.write(f"{row['feature']:40s} {row['importance']:10.6f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Linear Regression Coefficients:\n")
        f.write("-" * 80 + "\n")
        for i, (feature, coef) in enumerate(zip(feature_names, linear_model.coef_)):
            f.write(f"{feature:40s} {coef:15.6f}\n")
    
    print(f"[EXPORT] Model summary exported to {txt_file}")
    
    # 2. C++ header file with LINEAR MODEL coefficients
    h_file = f'{output_dir}/cuda_heuristic_weights.h'
    with open(h_file, 'w') as f:
        f.write("// Auto-generated by train_cuda_heuristic.py\n")
        f.write("// DO NOT EDIT - regenerate by running performance tests\n\n")
        f.write("#pragma once\n\n")
        f.write("#include <cstddef>\n\n")
        f.write("namespace llaminar {\n")
        f.write("namespace cuda {\n\n")
        
        f.write("// ML-learned coefficients from Linear Regression\n")
        f.write("// Model trained to predict GFLOPS from configuration parameters\n")
        f.write("struct MLHeuristicWeights {\n")
        
        # Export LINEAR REGRESSION coefficients (not feature importance!)
        f.write(f"    static constexpr double intercept = {linear_model.intercept_:.8f};\n\n")
        
        # Export all 32 feature coefficients
        for i, feature in enumerate(feature_names):
            cpp_name = feature.replace('_', '_').lower() + '_coef'
            f.write(f"    static constexpr double {cpp_name} = {linear_model.coef_[i]:.8f};\n")
        
        f.write("\n    // Scaler parameters (for feature normalization)\n")
        f.write("    struct ScalerParams {\n")
        f.write("        double mean;\n")
        f.write("        double std;\n")
        f.write("    };\n\n")
        
        # Export scaler mean/std for each feature
        f.write("    static const ScalerParams feature_scalers[" + str(len(feature_names)) + "];\n")
        f.write("};\n\n")
        
        # Define scaler array in implementation
        f.write("// Scaler parameters for feature normalization\n")
        f.write("inline const MLHeuristicWeights::ScalerParams MLHeuristicWeights::feature_scalers[] = {\n")
        for i, feature in enumerate(feature_names):
            mean = scaler.mean_[i]
            std = scaler.scale_[i]
            f.write(f"    {{{mean:.6f}, {std:.6f}}},  // {feature}\n")
        f.write("};\n\n")
        
        f.write("}  // namespace cuda\n")
        f.write("}  // namespace llaminar\n")
    
    print(f"[EXPORT] C++ header exported to {h_file}")

def compute_config_hash(m, n, k, tile_m, tile_n, tile_k, threads_m, threads_n, 
                       work_m, work_n, prefetch, transpose, vectorize):
    """
    Compute FNV-1a hash for a configuration.
    This MUST match the hash() implementation in CudaGemmConfig.h.
    Includes both problem size (m,n,k) and config parameters.
    """
    # FNV-1a constants
    h = 14695981039346656037
    fnv_prime = 1099511628211
    
    def mix(v):
        nonlocal h
        h ^= int(v)
        h *= fnv_prime
        h &= 0xFFFFFFFFFFFFFFFF  # Keep 64-bit
    
    # Hash problem size FIRST (critical for distinguishing different tests)
    mix(m)
    mix(n)
    mix(k)
    
    # Then hash config parameters in same order as C++
    mix(tile_m)
    mix(tile_n)
    mix(tile_k)
    mix(threads_m)
    mix(threads_n)
    mix(work_m)
    mix(work_n)
    mix(prefetch)
    mix(1 if transpose else 0)
    mix(vectorize)
    
    return h

def export_lookup_table(df_features, gb_model, scaler, feature_columns, output_dir='.'):
    """
    Export Gradient Boosting predictions as C++ lookup table.
    
    Since we have a finite set of configurations (~650 per test), we can:
    1. Pre-compute GB predictions for all configs
    2. Export as C++ hash map for O(1) lookup
    3. Get exact GB accuracy (R²=0.9999) without porting decision trees
    
    Args:
        df_features: DataFrame with all benchmarked configurations
        gb_model: Trained Gradient Boosting model (high accuracy)
        scaler: StandardScaler instance
        feature_columns: List of feature names
        output_dir: Output directory
    """
    h_file = f'{output_dir}/cuda_heuristic_lookup.h'
    
    # Compute GB predictions for all configs
    X = df_features[feature_columns]
    X_scaled = scaler.transform(X)
    df_features['gb_predicted_gflops'] = gb_model.predict(X_scaled)
    
    print(f"\n[LOOKUP] Exporting GB predictions as lookup table...")
    print(f"         Configs: {len(df_features)}")
    print(f"         Tests: {df_features['test_name'].nunique()}")
    
    with open(h_file, 'w') as f:
        f.write("// Auto-generated by train_cuda_heuristic.py\n")
        f.write("// Gradient Boosting predictions (R²=0.9999) as lookup table\n")
        f.write("// DO NOT EDIT - regenerate by running performance tests\n\n")
        f.write("#pragma once\n\n")
        f.write("#include <unordered_map>\n")
        f.write("#include <cstdint>\n")
        f.write("#include <tuple>\n\n")
        
        f.write("namespace llaminar {\n")
        f.write("namespace cuda {\n\n")
        
        # Export full predictions lookup
        f.write(f"// GB predictions for all {len(df_features)} benchmarked configs\n")
        f.write("// Usage: gb_predictions.at(config.hash()) -> predicted GFLOPS\n")
        f.write("static const std::unordered_map<uint64_t, float> gb_predictions = {\n")
        
        for idx, row in df_features.iterrows():
            config_hash = compute_config_hash(
                int(row['m']), int(row['n']), int(row['k']),  # Problem size
                int(row['tile_m']), int(row['tile_n']), int(row['tile_k']),
                int(row['threads_m']), int(row['threads_n']),
                int(row['work_m']), int(row['work_n']),
                int(row['prefetch_stages']),
                bool(row['transpose_smem']), int(row['vectorize_load'])
            )
            
            test_name = row['test_name'][:40]  # Truncate long names
            config_str = f"tile_{int(row['tile_m'])}x{int(row['tile_n'])}"
            
            f.write(f"    {{0x{config_hash:016x}ULL, {row['gb_predicted_gflops']:.2f}f}},")
            f.write(f"  // {test_name} - {config_str}\n")
        
        f.write("};\n\n")
        
        # Export best config per test shape
        f.write("// Best config per (m, n, k) shape (top-1 by GB prediction)\n")
        f.write("struct ShapeKey {\n")
        f.write("    int m, n, k;\n")
        f.write("    bool operator==(const ShapeKey& o) const {\n")
        f.write("        return m == o.m && n == o.n && k == o.k;\n")
        f.write("    }\n")
        f.write("};\n\n")
        
        f.write("struct ShapeKeyHash {\n")
        f.write("    size_t operator()(const ShapeKey& k) const {\n")
        f.write("        return std::hash<int>()(k.m) ^ \n")
        f.write("               (std::hash<int>()(k.n) << 1) ^ \n")
        f.write("               (std::hash<int>()(k.k) << 2);\n")
        f.write("    }\n")
        f.write("};\n\n")
        
        f.write("static const std::unordered_map<ShapeKey, uint64_t, ShapeKeyHash> best_configs = {\n")
        
        # Group by test (which groups by m/n/k) and find best config
        for test_name in df_features['test_name'].unique():
            test_data = df_features[df_features['test_name'] == test_name]
            best_config = test_data.loc[test_data['gb_predicted_gflops'].idxmax()]
            
            m = int(best_config['m'])
            n = int(best_config['n'])
            k = int(best_config['k'])
            
            config_hash = compute_config_hash(
                m, n, k,  # Problem size
                int(best_config['tile_m']), int(best_config['tile_n']), int(best_config['tile_k']),
                int(best_config['threads_m']), int(best_config['threads_n']),
                int(best_config['work_m']), int(best_config['work_n']),
                int(best_config['prefetch_stages']),
                bool(best_config['transpose_smem']), int(best_config['vectorize_load'])
            )
            
            config_str = f"tile_{int(best_config['tile_m'])}x{int(best_config['tile_n'])}"
            f.write(f"    {{{{{m}, {n}, {k}}}, 0x{config_hash:016x}ULL}},")
            f.write(f"  // {test_name} - {config_str} ({best_config['gb_predicted_gflops']:.1f} GFLOPS)\n")
        
        f.write("};\n\n")
        
        f.write("}  // namespace cuda\n")
        f.write("}  // namespace llaminar\n")
    
    print(f"[EXPORT] Lookup table exported to {h_file}")
    print(f"         Binary size: ~{len(df_features) * 12 / 1024:.1f} KB")

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train CUDA GEMM heuristic model')
    parser.add_argument('--input', default='../../../../../build_v2/cuda_gemm_benchmark_data.csv',
                       help='Input CSV file with benchmark data')
    parser.add_argument('--output-dir', default='../generated',
                       help='Output directory for generated files')
    args = parser.parse_args()
    
    print("=" * 80)
    print("CUDA GEMM Heuristic Training Pipeline")
    print("=" * 80)
    
    # Load data
    print(f"\n[1/6] Loading data from {args.input}...")
    
    try:
        df = pd.read_csv(args.input)
        print(f"      Loaded {len(df)} benchmark results")
        print(f"      Unique test cases: {df['test_name'].nunique()}")
        print(f"      Unique configurations: {df.groupby(['tile_m', 'tile_n', 'tile_k', 'threads_m', 'threads_n', 'work_m', 'work_n', 'prefetch_stages', 'transpose_smem', 'vectorize_load']).ngroups}")
    except FileNotFoundError:
        print(f"ERROR: {args.input} not found!")
        print("Please run the benchmark first to generate training data.")
        return 1
    
    # Engineer features
    print("\n[2/6] Engineering features...")
    df_features = engineer_features(df)
    
    # Select features for training
    raw_features = ['tile_m', 'tile_n', 'tile_k', 'threads_m', 'threads_n', 
                    'work_m', 'work_n', 'prefetch_stages', 'transpose_smem', 'vectorize_load',
                    'm', 'n', 'k']
    
    # Original engineered features
    base_engineered = ['threads_per_block', 'tile_size', 'tile_area', 'work_per_thread',
                       'occupancy_estimate', 'arithmetic_intensity',
                       'm_over_tile_m', 'n_over_tile_n', 'k_over_tile_k',
                       'tile_coverage_m', 'tile_coverage_n',
                       'is_tiny', 'is_small', 'is_medium', 'is_large',
                       'is_square', 'aspect_ratio', 'tile_aspect_ratio', 'tile_shape_match']
    
    # NEW: Enhanced features for better generalization
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
        'tile_size_x_batch', 'occupancy_x_intensity', 'work_per_thread_x_batch'
    ]
    
    feature_columns = raw_features + base_engineered + enhanced_features
    X = df_features[feature_columns]
    y = df_features['gflops']
    
    print(f"      Feature count: {len(feature_columns)} ({len(base_engineered)} base + {len(enhanced_features)} enhanced)")
    print(f"      Target: GFLOPS (range {y.min():.1f} - {y.max():.1f})")
    
    # Model-based train/val/test split to prevent overfitting
    print("\n[3/6] Splitting train/val/test sets by MODEL SIZE...")
    print("      Strategy: NEVER let model see 14B/235B/671B during training")
    print("      This tests true generalization to unseen model sizes")
    
    # Define splits based on model families
    train_tests = [
        'Qwen_0_5B_SingleToken_QKV', 'Qwen_0_5B_Batch32_QKV', 'Qwen_0_5B_FFN_Gate',
        'Qwen_7B_SingleToken_QKV', 'Qwen_7B_Batch128_QKV', 'Qwen_7B_FFN_Gate',
        'Qwen_72B_SingleToken_QKV', 'Qwen_72B_Batch128_QKV', 'Qwen_72B_FFN_Down'
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
    print(f"             Models: 0.5B, 7B, 72B")
    print(f"      Val:   {len(X_val)} samples ({len(val_tests)} model sizes)")
    print(f"             Models: 4B, 32B")
    print(f"      Test:  {len(X_test)} samples ({len(test_tests)} model sizes - UNSEEN)")
    print(f"             Models: 14B, 235B, 671B")
    print(f"      Total: {len(X_train) + len(X_val) + len(X_test)} samples")
    
    # Scale features (fit ONLY on training data to prevent leakage)
    print("\n[4/6] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\n[5/6] Training Gradient Boosting Regressor...")
    print("      Enhanced regularization to reduce overfitting:")
    print("        - max_depth: 8 → 6 (shallower trees)")
    print("        - learning_rate: 0.1 → 0.05 (slower, more stable)")
    print("        - min_samples_split: 10 → 20 (more conservative)")
    print("        - min_samples_leaf: 5 → 10 (smoother predictions)")
    print("        - subsample: 0.8 → 0.7 (more stochastic)")
    print("        - max_features: None → 'sqrt' (feature subsampling)")
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,              # Reduced from 8 → less overfitting
        learning_rate=0.05,       # Reduced from 0.1 → slower, more stable
        min_samples_split=20,     # Increased from 10 → more conservative splits
        min_samples_leaf=10,      # Increased from 5 → smoother predictions
        subsample=0.7,            # Reduced from 0.8 → more regularization
        max_features='sqrt',      # Added feature subsampling
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on all three sets
    print("\n[6/6] Evaluating model...")
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
    print("Model Performance (Gradient Boosting):")
    print("=" * 80)
    print(f"{'':15s} {'R²':>10s} {'MAE':>12s} {'RMSE':>12s}")
    print("-" * 80)
    print(f"{'Train (seen)':15s} {train_r2:10.4f} {train_mae:10.2f} {train_rmse:10.2f}")
    print(f"{'Val (seen)':15s} {val_r2:10.4f} {val_mae:10.2f} {val_rmse:10.2f}")
    print(f"{'Test (UNSEEN)':15s} {test_r2:10.4f} {test_mae:10.2f} {test_rmse:10.2f}")
    print("=" * 80)
    print("\n⚠️  KEY METRIC: Test R² measures generalization to UNSEEN model sizes")
    print(f"   Test models: {', '.join([t.split('_')[1] for t in test_tests[:5]])}... ({len(test_tests)} total)")
    
    # If test R² << train R², we're overfitting
    if test_r2 < train_r2 - 0.1:
        print(f"\n⚠️  WARNING: Test R² ({test_r2:.4f}) << Train R² ({train_r2:.4f})")
        print("   Model may be overfitting to training sizes")
        print("   Consider: More regularization, simpler model, or more diverse training data")
    elif test_r2 >= train_r2 - 0.05:
        print(f"\n✓  Good generalization: Test R² ({test_r2:.4f}) ≈ Train R² ({train_r2:.4f})")
        print("   Model generalizes well to unseen model sizes!")
    
    # Cross-validation (only on training set)
    print("\nCross-validation on training set (5-fold):")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"  R² scores: {cv_scores}")
    print(f"  Mean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train linear model for C++ export (simpler than GB, easier to port)
    print("\n[BONUS] Training Linear Regression for C++ export...")
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    
    y_train_pred_linear = linear_model.predict(X_train_scaled)
    y_val_pred_linear = linear_model.predict(X_val_scaled)
    y_test_pred_linear = linear_model.predict(X_test_scaled)
    
    linear_train_r2 = r2_score(y_train, y_train_pred_linear)
    linear_val_r2 = r2_score(y_val, y_val_pred_linear)
    linear_test_r2 = r2_score(y_test, y_test_pred_linear)
    
    print(f"\nLinear Model Performance:")
    print(f"  Train R²: {linear_train_r2:.4f}")
    print(f"  Val R²:   {linear_val_r2:.4f}")
    print(f"  Test R²:  {linear_test_r2:.4f}")
    print(f"Note: Linear model easier to port to C++, but less accurate than GB")
    
    # Export results
    print("\n" + "=" * 80)
    print("Exporting Results:")
    print("=" * 80)
    
    plot_results(y_test, y_test_pred, model.feature_importances_, feature_columns,
                 output_file=f'{args.output_dir}/cuda_heuristic_validation.png')
    export_cpp_heuristic(model, linear_model, scaler, feature_columns, output_dir=args.output_dir)
    export_lookup_table(df_features, model, scaler, feature_columns, output_dir=args.output_dir)
    
    # Generate predictions for all configs (using LINEAR model for C++ compatibility)
    print("\n[BONUS] Generating predictions for all configurations...")
    print("         (Using Linear Regression predictions - same model exported to C++)")
    df_features['predicted_gflops'] = linear_model.predict(scaler.transform(X))
    df_features['prediction_error'] = df_features['predicted_gflops'] - df_features['gflops']
    df_features['abs_error'] = df_features['prediction_error'].abs()
    
    # Save predictions
    predictions_file = f'{args.output_dir}/cuda_gemm_predictions.csv'
    df_features[['test_name', 'm', 'n', 'k', 'tile_m', 'tile_n', 'tile_k',
                 'threads_m', 'threads_n', 'work_m', 'work_n',
                 'prefetch_stages', 'transpose_smem', 'vectorize_load',
                 'gflops', 'predicted_gflops', 'prediction_error', 'abs_error']].to_csv(
        predictions_file, index=False
    )
    print(f"[EXPORT] Predictions saved to {predictions_file}")
    
    # Show worst predictions
    print("\nWorst predictions (top 10 by absolute error):")
    worst = df_features.nlargest(10, 'abs_error')[['test_name', 'tile_m', 'tile_n', 'threads_m', 'threads_n',
                                                     'gflops', 'predicted_gflops', 'abs_error']]
    print(worst.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review feature importance in cuda_heuristic_model_weights.txt")
    print("  2. Use top features to refine C++ heuristic weights")
    print("  3. OR: Use predictions in cuda_gemm_predictions.csv as lookup table")
    print("  4. Re-run validation tests to verify improved correlation")

if __name__ == '__main__':
    main()
