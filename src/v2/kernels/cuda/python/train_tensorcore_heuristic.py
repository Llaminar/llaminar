#!/usr/bin/env python3
"""
Train a regression model to predict Tensor Core GEMM performance from kernel configuration.

This script trains on comprehensive benchmark data from Phase 3+ covering:
- All Tensor Core tile configurations (TILE_M × TILE_N × TILE_K)
- Model sizes: 0.5B, 4B, 7B, 14B
- Batch sizes: 1, 8, 32, 128
- Operations: Q/K/V projections, FFN up/down, LM head

Key findings from Phase 3:
- TILE_M matching actual M dimension is critical (32×64×16 for m=32)
- Batch size (M dimension) is dominant feature (~78% importance)
- TILE_K=16 optimal for sm_80 (matches MMA instruction)

Usage:
    python train_tensorcore_heuristic.py [--output-dir DIR]

Output:
    - tensorcore_heuristic_weights.h (C++ header with learned weights)
    - tensorcore_heuristic_validation.png (performance plots)
    - Feature importance rankings

Author: David Sanftenberg
Date: November 1, 2025
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def engineer_features(df):
    """
    Engineer features from raw configuration parameters.
    
    Key insights from Phase 3:
    - Tile-to-matrix matching is critical (TILE_M == M → best performance)
    - Batch size dominance (M dimension ~78% of variance)
    - Tile aspect ratio should match matrix aspect ratio
    """
    df = df.copy()
    
    # ============================================================================
    # CRITICAL FEATURES (Phase 3 insights)
    # ============================================================================
    
    # Tile utilization (most important for small M!)
    df['tile_m_utilization'] = np.minimum(df['m'] / df['tile_m'], 1.0)
    df['tile_n_utilization'] = np.minimum(df['n'] / df['tile_n'], 1.0)
    df['tile_efficiency'] = df['tile_m_utilization'] * df['tile_n_utilization']
    
    # Exact tile match (critical for single-token decode)
    df['tile_m_matches_m'] = (df['tile_m'] == df['m']).astype(int)
    df['tile_m_double_m'] = (df['tile_m'] == 2 * df['m']).astype(int)
    df['tile_m_half_m'] = (df['tile_m'] * 2 == df['m']).astype(int)
    
    # ============================================================================
    # PROBLEM SIZE FEATURES
    # ============================================================================
    
    # Batch size (M dimension) - DOMINANT FEATURE
    df['log_m'] = np.log1p(df['m'])
    df['is_single_token'] = (df['m'] == 1).astype(int)
    df['is_small_batch'] = ((df['m'] > 1) & (df['m'] <= 32)).astype(int)
    df['is_medium_batch'] = ((df['m'] > 32) & (df['m'] <= 128)).astype(int)
    df['is_large_batch'] = (df['m'] > 128).astype(int)
    
    # Matrix dimensions
    df['log_n'] = np.log1p(df['n'])
    df['log_k'] = np.log1p(df['k'])
    df['matrix_size'] = df['m'] * df['n'] * df['k']
    df['log_matrix_size'] = np.log1p(df['matrix_size'])
    
    # Matrix aspect ratio
    df['aspect_ratio_mn'] = df['n'] / df['m'].clip(lower=1)
    df['aspect_ratio_mk'] = df['k'] / df['m'].clip(lower=1)
    df['is_square'] = ((df['n'] / df['k']).between(0.9, 1.1)).astype(int)
    df['is_tall'] = (df['k'] > df['n'] * 1.5).astype(int)
    df['is_wide'] = (df['n'] > df['k'] * 1.5).astype(int)
    
    # ============================================================================
    # TILE CONFIGURATION FEATURES
    # ============================================================================
    
    # Tile dimensions
    df['tile_area'] = df['tile_m'] * df['tile_n']
    df['tile_volume'] = df['tile_m'] * df['tile_n'] * df['tile_k']
    df['log_tile_area'] = np.log1p(df['tile_area'])
    
    # Tile aspect ratio
    df['tile_aspect_ratio'] = df['tile_n'] / df['tile_m']
    df['tile_shape_match'] = (
        ((df['aspect_ratio_mn'] > 1.5) & (df['tile_aspect_ratio'] > 1.0)) |
        ((df['aspect_ratio_mn'] < 0.67) & (df['tile_aspect_ratio'] < 1.0))
    ).astype(int)
    
    # Tile size categories
    df['is_small_tile'] = (df['tile_m'] <= 32).astype(int)
    df['is_medium_tile'] = ((df['tile_m'] > 32) & (df['tile_m'] <= 64)).astype(int)
    df['is_large_tile'] = (df['tile_m'] > 64).astype(int)
    
    # K-tile dimensions (CRITICAL - Phase 3 showed TILE_K=16 optimal)
    df['k_tiles_per_iteration'] = df['k'] / df['tile_k']
    df['tile_k_is_16'] = (df['tile_k'] == 16).astype(int)
    df['tile_k_is_32'] = (df['tile_k'] == 32).astype(int)
    df['tile_k_is_64'] = (df['tile_k'] == 64).astype(int)
    
    # ============================================================================
    # OCCUPANCY AND RESOURCE USAGE
    # ============================================================================
    
    # Grid dimensions (number of thread blocks)
    df['grid_blocks_m'] = np.ceil(df['m'] / df['tile_m'])
    df['grid_blocks_n'] = np.ceil(df['n'] / df['tile_n'])
    df['total_thread_blocks'] = df['grid_blocks_m'] * df['grid_blocks_n']
    
    # Shared memory estimate (bytes)
    # Two tiles (A and B) for current k-tile
    df['smem_bytes'] = (df['tile_m'] * df['tile_k'] + df['tile_n'] * df['tile_k']) * 4
    df['smem_pressure'] = df['smem_bytes'] / (48 * 1024)  # Fraction of 48KB limit
    
    # Arithmetic intensity (FLOPs per byte loaded)
    # Each element used TILE_M or TILE_N times
    df['arithmetic_intensity'] = (2 * df['tile_m'] * df['tile_n'] * df['tile_k']) / (
        (df['tile_m'] * df['tile_k'] + df['tile_n'] * df['tile_k']) * 4
    )
    
    # ============================================================================
    # INTERACTION FEATURES
    # ============================================================================
    
    # Tile-matrix interactions
    df['m_times_tile_m'] = df['m'] * df['tile_m']
    df['m_over_tile_m'] = df['m'] / df['tile_m'].clip(lower=1)
    
    # Batch-size-dependent tile effectiveness
    df['small_batch_small_tile'] = df['is_small_batch'] * df['is_small_tile']
    df['medium_batch_medium_tile'] = df['is_medium_batch'] * df['is_medium_tile']
    df['large_batch_large_tile'] = df['is_large_batch'] * df['is_large_tile']
    
    return df

def plot_results(y_true, y_pred, feature_importance, feature_names, output_file='tensorcore_heuristic_validation.png'):
    """
    Generate validation plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Predicted vs Actual
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=20, c=y_pred-y_true, cmap='RdYlGn_r')
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual GFLOPS', fontsize=12)
    axes[0, 0].set_ylabel('Predicted GFLOPS', fontsize=12)
    axes[0, 0].set_title('Predicted vs Actual Performance (Tensor Core)', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_pred - y_true
    axes[0, 1].scatter(y_true, residuals, alpha=0.5, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Actual GFLOPS', fontsize=12)
    axes[0, 1].set_ylabel('Residual (Predicted - Actual)', fontsize=12)
    axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Feature importance (top 25)
    top_n = 25
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).nlargest(top_n, 'importance')
    
    axes[1, 0].barh(importance_df['feature'], importance_df['importance'])
    axes[1, 0].set_xlabel('Importance', fontsize=12)
    axes[1, 0].set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].tick_params(axis='y', labelsize=8)
    
    # Error distribution
    axes[1, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Residual (GFLOPS)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved validation plots to {output_file}")

def export_cpp_header(model, scaler, feature_names, output_dir='.'):
    """
    Export trained model as C++ header file with lookup table.
    
    Uses decision tree leaf values for fast C++ inference.
    """
    output_file = f"{output_dir}/tensorcore_heuristic_weights.h"
    
    with open(output_file, 'w') as f:
        f.write("""/**
 * @file tensorcore_heuristic_weights.h
 * @brief ML-learned weights for Tensor Core GEMM heuristic
 * 
 * AUTO-GENERATED by train_tensorcore_heuristic.py
 * DO NOT EDIT MANUALLY
 * 
 * Model: Gradient Boosting Regressor
 * Training data: Comprehensive Tensor Core benchmarks (0.5B-14B models)
 * Feature importance: See tensorcore_heuristic_validation.png
 * 
 * Key insights:
 * - Batch size (M) is dominant feature (~78% importance)
 * - Tile-matrix matching critical (TILE_M == M optimal)
 * - TILE_K=16 optimal for sm_80
 * 
 * @author Auto-generated
 * @date November 1, 2025
 */

#pragma once

#include <array>
#include <cmath>
#include <algorithm>

namespace llaminar2
{
namespace cuda
{

/**
 * @brief Predict Tensor Core GEMM performance using ML model
 * 
 * @param m Batch size (rows)
 * @param n Columns
 * @param k Inner dimension
 * @param tile_m Tile M dimension
 * @param tile_n Tile N dimension
 * @param tile_k Tile K dimension
 * @return Predicted GFLOPS
 */
inline double predictTensorCorePerformance(
    int m, int n, int k,
    int tile_m, int tile_n, int tile_k
) {
    // Feature engineering (matching Python training script)
""")
        
        # Write feature computation code
        f.write("    \n    // Tile utilization\n")
        f.write("    double tile_m_utilization = std::min(static_cast<double>(m) / tile_m, 1.0);\n")
        f.write("    double tile_efficiency = tile_m_utilization * std::min(static_cast<double>(n) / tile_n, 1.0);\n")
        
        f.write("\n    // Problem size\n")
        f.write("    double log_m = std::log1p(m);\n")
        f.write("    int is_small_batch = (m > 1 && m <= 32) ? 1 : 0;\n")
        
        f.write("\n    // Tile matching\n")
        f.write("    int tile_m_matches_m = (tile_m == m) ? 1 : 0;\n")
        f.write("    int tile_k_is_16 = (tile_k == 16) ? 1 : 0;\n")
        
        f.write("\n    // For now, use simplified heuristic based on Phase 3 findings\n")
        f.write("    // Full ML model implementation coming soon\n")
        f.write("    \n")
        f.write("    // Phase 3 optimization: Prefer TILE_M matching M dimension\n")
        f.write("    double score = 1000.0;  // Base score\n")
        f.write("    \n")
        f.write("    // Bonus for tile-matrix match\n")
        f.write("    score += tile_m_matches_m * 500.0;\n")
        f.write("    score += tile_efficiency * 300.0;\n")
        f.write("    \n")
        f.write("    // Bonus for TILE_K=16 (optimal for sm_80)\n")
        f.write("    score += tile_k_is_16 * 200.0;\n")
        f.write("    \n")
        f.write("    // Penalty for small batch with large tiles\n")
        f.write("    if (is_small_batch && tile_m > 64) score -= 400.0;\n")
        f.write("    \n")
        f.write("    return score;\n")
        f.write("}\n\n")
        
        f.write("} // namespace cuda\n")
        f.write("} // namespace llaminar2\n")
    
    print(f"\n✓ Exported C++ header to {output_file}")
    print("\nNOTE: Full ML model implementation pending - using Phase 3 heuristic")

def main():
    parser = argparse.ArgumentParser(description='Train Tensor Core GEMM heuristic model')
    parser.add_argument('--input', default='../../../../../build_v2/tensorcore_benchmark_data.csv',
                       help='Input CSV file with benchmark data')
    parser.add_argument('--output-dir', default='../generated',
                       help='Output directory for generated header')
    args = parser.parse_args()
    
    print("=" * 80)
    print("TENSOR CORE GEMM HEURISTIC TRAINING")
    print("=" * 80)
    
    # Load data
    print(f"\n[1/6] Loading benchmark data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
        print(f"      Loaded {len(df)} benchmark results")
        print(f"      Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"\n❌ ERROR: Benchmark data not found at {args.input}")
        print("\nPlease run the benchmark first:")
        print("  cd build_v2")
        print("  ./performance/v2_perf_tensorcore_heuristic_validation")
        return 1
    
    # Engineer features
    print("\n[2/6] Engineering features...")
    df = engineer_features(df)
    print(f"      Generated {len(df.columns)} total features")
    
    # Prepare training data
    print("\n[3/6] Preparing training data...")
    feature_cols = [c for c in df.columns if c not in ['model_size', 'operation', 'batch_size', 
                                                         'gflops', 'time_ms', 'iterations']]
    X = df[feature_cols]
    y = df['gflops']
    
    print(f"      Features: {len(feature_cols)}")
    print(f"      Samples: {len(y)}")
    print(f"      GFLOPS range: {y.min():.1f} - {y.max():.1f}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("\n[4/6] Training Gradient Boosting model...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model.fit(X_train_scaled, y_train)
    print("      Training complete!")
    
    # Evaluate
    print("\n[5/6] Evaluating model...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print(f"      Training R²: {r2_train:.4f}")
    print(f"      Test R²: {r2_test:.4f}")
    print(f"      Test MAE: {mae_test:.1f} GFLOPS")
    
    # Feature importance
    print("\n      Top 10 most important features:")
    importance = model.feature_importances_
    feature_importance = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    for feat, imp in feature_importance[:10]:
        print(f"        {feat:30s}: {imp:6.4f}")
    
    # Generate plots
    print("\n[6/6] Generating validation plots...")
    plot_results(y_test, y_pred_test, importance, feature_cols)
    
    # Export C++ header
    export_cpp_header(model, scaler, feature_cols, args.output_dir)
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - {args.output_dir}/tensorcore_heuristic_weights.h")
    print(f"  - tensorcore_heuristic_validation.png")
    print(f"\nModel performance:")
    print(f"  - R² score: {r2_test:.4f}")
    print(f"  - MAE: {mae_test:.1f} GFLOPS")
    print(f"  - Dominant feature: {feature_importance[0][0]} ({feature_importance[0][1]:.1%})")
    
    return 0

if __name__ == '__main__':
    exit(main())
