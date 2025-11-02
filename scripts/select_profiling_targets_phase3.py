#!/usr/bin/env python3
"""
PHASE 3: Model-Guided Profiling
Identifies high-uncertainty configs where the neural network is least confident,
then profiles only those specific configs for maximum information gain.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description='Phase 3: Model-guided profiling target selection')
    parser.add_argument('--benchmark-csv', required=True, help='Benchmark results CSV')
    parser.add_argument('--predictions-csv', required=True, help='Model predictions CSV')
    parser.add_argument('--output', default='profiling_targets_phase3.csv', help='Output CSV')
    parser.add_argument('--uncertainty-threshold', type=float, default=0.2,
                       help='Prediction uncertainty threshold (default: 0.2 = 20%)')
    parser.add_argument('--max-targets', type=int, default=200,
                       help='Maximum configs to profile (default: 200)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("PHASE 3: Model-Guided Profiling Target Selection")
    print("=" * 80)
    print()
    
    # Load data
    print(f"[1/4] Loading benchmark data from {args.benchmark_csv}...")
    try:
        df_bench = pd.read_csv(args.benchmark_csv)
        print(f"      Loaded {len(df_bench)} benchmark results")
    except Exception as e:
        print(f"[ERROR] Failed to load benchmark CSV: {e}")
        return 1
    
    print(f"\n[2/4] Loading model predictions from {args.predictions_csv}...")
    try:
        df_pred = pd.read_csv(args.predictions_csv)
        print(f"      Loaded {len(df_pred)} predictions")
    except Exception as e:
        print(f"[ERROR] Failed to load predictions CSV: {e}")
        print("        Generate predictions first:")
        print("        python3 src/v2/kernels/cuda/python/predict_uncertainty.py")
        return 1
    
    # Merge datasets
    print("\n[3/4] Computing prediction uncertainty...")
    
    # Merge on config features
    merge_cols = ['test_name', 'm', 'n', 'k', 'tile_m', 'tile_n', 'tile_k',
                  'threads_m', 'threads_n', 'work_m', 'work_n']
    df_merged = df_bench.merge(df_pred, on=merge_cols, how='inner', suffixes=('_actual', '_pred'))
    
    # Compute prediction error (uncertainty proxy)
    df_merged['pred_error'] = np.abs(df_merged['gflops_actual'] - df_merged['gflops_pred'])
    df_merged['pred_error_pct'] = df_merged['pred_error'] / df_merged['gflops_actual'].clip(lower=1.0)
    
    # Rank by uncertainty
    df_merged['uncertainty_rank'] = df_merged['pred_error_pct'].rank(ascending=False)
    
    # Select high-uncertainty configs
    threshold_pct = args.uncertainty_threshold * 100
    high_uncertainty = df_merged[df_merged['pred_error_pct'] >= args.uncertainty_threshold]
    
    print(f"      Total configs: {len(df_merged)}")
    print(f"      High uncertainty (>{threshold_pct:.0f}% error): {len(high_uncertainty)}")
    print(f"      Top uncertainty: {df_merged['pred_error_pct'].max():.1%}")
    print(f"      Median uncertainty: {df_merged['pred_error_pct'].median():.1%}")
    
    # Select diverse sample (stratified by test_name)
    print(f"\n[4/4] Selecting up to {args.max_targets} diverse targets...")
    
    targets = []
    per_test = args.max_targets // df_merged['test_name'].nunique()
    
    for test_name, group in high_uncertainty.groupby('test_name'):
        # Take top-N most uncertain from each test
        top_uncertain = group.nlargest(per_test, 'pred_error_pct')
        targets.append(top_uncertain)
        print(f"      {test_name}: {len(top_uncertain)} configs")
    
    df_targets = pd.concat(targets, ignore_index=True)
    
    # Trim to max
    if len(df_targets) > args.max_targets:
        df_targets = df_targets.nlargest(args.max_targets, 'pred_error_pct')
    
    # Select columns for profiling
    output_cols = merge_cols + ['gflops_actual', 'gflops_pred', 'pred_error_pct', 
                                 'prefetch_stages', 'transpose_smem', 'vectorize_load']
    df_output = df_targets[output_cols]
    
    # Save
    output_path = Path(args.output)
    df_output.to_csv(output_path, index=False)
    
    print(f"\n[OUTPUT] Saved {len(df_output)} profiling targets to {output_path}")
    print(f"         Average uncertainty: {df_output['pred_error_pct'].mean():.1%}")
    print(f"         Estimated profiling time: {len(df_output) * 2 / 60:.1f} minutes")
    print()
    print("NEXT STEPS:")
    print("1. Run targeted profiler on these specific configs")
    print("2. Collect hardware metrics (bank conflicts, occupancy, etc.)")
    print("3. Add profiler metrics as new features to training data")
    print("4. Re-train neural network with enriched dataset")
    print("5. Expect 5-15% improvement in canary performance")
    print()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
