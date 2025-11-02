#!/usr/bin/env python3
"""
Collect NVIDIA profiling data for top-N and worst-N configs per test case.

This script:
1. Reads benchmark results (cuda_gemm_benchmark_data.csv)
2. For each test case, identifies top-N and worst-N configs (default N=50)
3. Runs NVIDIA profiler (ncu) on those configs
4. Extracts key metrics: SM efficiency, memory bandwidth, occupancy, etc.
5. Saves profiling data to CSV for training

Usage:
    python collect_profiling_data.py \\
        --input cuda_gemm_benchmark_data.csv \\
        --executable /path/to/profile_cuda_config \\
        --output cuda_gemm_profiling_data.csv \\
        --top-n 50

Author: David Sanftenberg
Date: November 2, 2025
"""

import pandas as pd
import numpy as np
import subprocess
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# NVIDIA profiling metrics to collect for GEMM performance analysis
# Carefully selected metrics that exist on A100/Ampere and correlate with performance
#
# Metric categories:
# 1. Memory hierarchy (DRAM, L2, L1 cache hit rates and throughput)
# 2. Compute utilization (SM busy time, instruction throughput)
# 3. Memory access patterns (coalescing efficiency, bank conflicts)
# 4. Execution efficiency (warp occupancy, divergence, stalls)
#
PROFILE_METRICS = [
    # === Memory Hierarchy Throughput ===
    'dram__throughput.avg.pct_of_peak_sustained_elapsed',  # DRAM bandwidth (% of peak)
    'lts__t_sector_hit_rate.pct',  # L2 cache hit rate (LTS = L2/Texture/Shared)
    'l1tex__t_sector_hit_rate.pct',  # L1 cache hit rate
    
    # === Compute Utilization ===
    'sm__throughput.avg.pct_of_peak_sustained_elapsed',  # SM compute throughput (% of peak)
    'sm__instruction_throughput.avg.pct_of_peak_sustained_elapsed',  # Instruction issue rate
    
    # === Warp Occupancy ===
    'sm__warps_active.avg.pct_of_peak_sustained_elapsed',  # Active warp ratio
    
    # === Memory Access Efficiency ===
    'smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct',  # Global load coalescing
    'smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct',  # Global store coalescing
    
    # === Shared Memory Efficiency ===
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',  # Shared mem bank conflicts (ld)
    'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',  # Shared mem bank conflicts (st)
    
    # === Thread Divergence ===
    'smsp__thread_inst_executed_per_inst_executed.ratio',  # Thread divergence (1.0 = perfect)
]

# Total: 11 metrics covering memory, compute, and efficiency dimensions
# This provides a comprehensive view of what makes a GEMM config fast or slow


def run_ncu_profile(executable: str, m: int, n: int, k: int, 
                   tile_m: int, tile_n: int, tile_k: int,
                   threads_m: int, threads_n: int,
                   work_m: int, work_n: int,
                   prefetch: int, transpose: int,
                   vectorize_load: int = 1) -> Dict[str, float]:
    """
    Run NVIDIA Compute profiler on a specific GEMM configuration.
    
    Returns dict of profiling metrics.
    """
    
    # Build ncu command
    # Use --metrics to collect specific metrics
    # Use --csv to get parseable output
    metrics_str = ','.join(PROFILE_METRICS)
    
    # Use sudo and full path for ncu (required for GPU profiling permissions)
    # Convert relative executable path to absolute
    import os
    executable_abs = os.path.abspath(executable)
    
    cmd = [
        'sudo', '/usr/local/cuda/bin/ncu',
        '--metrics', metrics_str,
        '--csv',
        '--target-processes', 'all',
        executable_abs,
        str(m), str(n), str(k),
        str(tile_m), str(tile_n), str(tile_k),
        str(threads_m), str(threads_n),
        str(work_m), str(work_n),
        str(prefetch), str(transpose),
        str(vectorize_load)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout per profile
        )
        
        if result.returncode != 0:
            print(f"[WARNING] ncu failed for config:")
            print(f"  stdout (first 500 chars): {result.stdout[:500]}")
            print(f"  stderr (first 500 chars): {result.stderr[:500]}")
            return {}
        
        # Parse CSV output
        metrics = parse_ncu_csv(result.stdout)
        if not metrics:
            print(f"[DEBUG] No metrics parsed. CSV output length: {len(result.stdout)}")
            print(f"[DEBUG] First 500 chars of stdout: {result.stdout[:500]}")
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"[WARNING] ncu timeout for m={m}, n={n}, k={k}")
        return {}
    except Exception as e:
        print(f"[ERROR] ncu execution failed: {e}")
        return {}


def parse_ncu_csv(csv_output: str) -> Dict[str, float]:
    """Parse ncu CSV output and extract metric values."""
    import csv
    import io
    
    metrics = {}
    
    # Use CSV reader to properly handle quoted fields
    reader = csv.reader(io.StringIO(csv_output))
    
    for row in reader:
        # Skip headers, empty rows, and rows that don't have enough columns
        if not row or len(row) < 15:
            continue
        
        # Skip header row and PROF messages
        if row[0] == 'ID' or row[0].startswith('=='):
            continue
        
        # CSV format (0-indexed):
        # 0: ID, 1: Process ID, 2: Process Name, 3: Host Name, 4: Kernel Name,
        # 5: Context, 6: Stream, 7: Block Size, 8: Grid Size, 9: Device, 10: CC,
        # 11: Section Name, 12: Metric Name, 13: Metric Unit, 14: Metric Value
        metric_name = row[12]
        metric_unit = row[13]
        metric_value_str = row[14]
        
        try:
            # Handle percentage values
            if metric_unit == '%':
                value = float(metric_value_str) / 100.0
            else:
                value = float(metric_value_str)
            
            # Use shortened but unique metric names
            # Keep enough of the name to distinguish between similar metrics
            if 'dram__throughput' in metric_name:
                short_name = 'dram_throughput_pct'
            elif 'lts__t_sector_hit_rate' in metric_name:
                short_name = 'l2_cache_hit_rate'
            elif 'l1tex__t_sector_hit_rate' in metric_name:
                short_name = 'l1_cache_hit_rate'
            elif 'sm__throughput' in metric_name:
                short_name = 'sm_throughput_pct'
            elif 'sm__instruction_throughput' in metric_name:
                short_name = 'sm_instruction_throughput_pct'
            elif 'sm__warps_active' in metric_name:
                short_name = 'sm_warps_active_pct'
            elif 'mem_global_op_ld' in metric_name:
                short_name = 'global_load_coalescing_pct'
            elif 'mem_global_op_st' in metric_name:
                short_name = 'global_store_coalescing_pct'
            elif 'bank_conflicts' in metric_name and 'op_ld' in metric_name:
                short_name = 'smem_bank_conflicts_ld'
            elif 'bank_conflicts' in metric_name and 'op_st' in metric_name:
                short_name = 'smem_bank_conflicts_st'
            elif 'thread_inst_executed' in metric_name:
                short_name = 'warp_divergence_ratio'
            else:
                # Fallback to last component
                short_name = metric_name.split('.')[-1] if '.' in metric_name else metric_name
            
            metrics[short_name] = value
            
        except (ValueError, IndexError):
            continue
    
    return metrics


def collect_profiling_for_test(df_test: pd.DataFrame, test_name: str, 
                               executable: str, top_n: int = 10) -> pd.DataFrame:
    """
    Collect profiling data for top-N and worst-N configs of a test case.
    
    Returns DataFrame with profiling metrics added.
    """
    
    # Sort by GFLOPS (descending)
    df_sorted = df_test.sort_values('gflops', ascending=False).reset_index(drop=True)
    
    # Select top-N and worst-N
    top_configs = df_sorted.head(top_n)
    worst_configs = df_sorted.tail(top_n)
    
    configs_to_profile = pd.concat([top_configs, worst_configs])
    
    print(f"\n[{test_name}] Profiling {len(configs_to_profile)} configs...")
    print(f"  Top-{top_n} GFLOPS range: {top_configs['gflops'].min():.1f} - {top_configs['gflops'].max():.1f}")
    print(f"  Worst-{top_n} GFLOPS range: {worst_configs['gflops'].min():.1f} - {worst_configs['gflops'].max():.1f}")
    
    # Collect profiling data for each config
    profiling_results = []
    
    for idx, row in configs_to_profile.iterrows():
        # Run profiler
        vectorize_load = int(row['vectorize_load']) if 'vectorize_load' in row else 1
        metrics = run_ncu_profile(
            executable,
            int(row['m']), int(row['n']), int(row['k']),
            int(row['tile_m']), int(row['tile_n']), int(row['tile_k']),
            int(row['threads_m']), int(row['threads_n']),
            int(row['work_m']), int(row['work_n']),
            int(row['prefetch_stages']), int(row['transpose_smem']),
            vectorize_load
        )
        
        # Add config identifier
        metrics['config_idx'] = idx
        metrics['test_name'] = test_name
        metrics['gflops'] = row['gflops']
        metrics['rank'] = idx  # Rank by GFLOPS
        
        profiling_results.append(metrics)
        
        # Progress update
        if (len(profiling_results) % 5) == 0:
            print(f"  Profiled {len(profiling_results)}/{len(configs_to_profile)} configs")
    
    return pd.DataFrame(profiling_results)


def main():
    parser = argparse.ArgumentParser(description='Collect NVIDIA profiling data for GEMM configs')
    parser.add_argument('--input', required=True, help='Input CSV with benchmark data')
    parser.add_argument('--executable', required=True, help='Path to profiling executable')
    parser.add_argument('--output', default='cuda_gemm_profiling_data.csv', help='Output CSV')
    parser.add_argument('--top-n', type=int, default=50, help='Number of top/worst configs to profile per test (default: 50)')
    parser.add_argument('--tests', nargs='+', help='Specific tests to profile (default: all)')
    
    args = parser.parse_args()
    
    # Load benchmark data
    print(f"[1/3] Loading benchmark data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"      Loaded {len(df)} benchmark results")
    print(f"      Tests: {df['test_name'].nunique()}")
    
    # Filter to specific tests if requested
    if args.tests:
        df = df[df['test_name'].isin(args.tests)]
        print(f"      Filtered to {len(df)} results ({len(args.tests)} tests)")
    
    # Check if ncu is available
    try:
        result = subprocess.run(['ncu', '--version'], capture_output=True, timeout=5)
        if result.returncode != 0:
            print("[ERROR] NVIDIA Compute profiler (ncu) not found!")
            print("        Install: sudo apt install nvidia-nsight-compute")
            return 1
    except FileNotFoundError:
        print("[ERROR] ncu command not found!")
        return 1
    
    # Collect profiling data for each test
    print(f"\n[2/3] Collecting profiling data...")
    all_profiling_data = []
    
    for test_name in df['test_name'].unique():
        df_test = df[df['test_name'] == test_name]
        
        prof_data = collect_profiling_for_test(
            df_test, 
            test_name, 
            args.executable,
            top_n=args.top_n
        )
        
        all_profiling_data.append(prof_data)
    
    # Combine and save
    print(f"\n[3/3] Saving profiling data...")
    df_profiling = pd.concat(all_profiling_data, ignore_index=True)
    df_profiling.to_csv(args.output, index=False)
    
    print(f"\n✓ Profiling data saved to {args.output}")
    print(f"  Total configs profiled: {len(df_profiling)}")
    print(f"  Metrics per config: {len([col for col in df_profiling.columns if col not in ['config_idx', 'test_name', 'gflops', 'rank']])}")
    
    # Show sample metrics
    print(f"\nSample profiling metrics:")
    metric_cols = [col for col in df_profiling.columns if col not in ['config_idx', 'test_name', 'gflops', 'rank']]
    for col in metric_cols[:5]:
        print(f"  {col}: {df_profiling[col].mean():.3f} ± {df_profiling[col].std():.3f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
