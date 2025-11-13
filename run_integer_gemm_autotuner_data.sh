#!/bin/bash
#
# run_integer_gemm_autotuner_data.sh
# Complete pipeline for gathering Integer GEMM autotuner training data
#
# Workflow:
#   Phase 1: Run full config sweep, collect GFLOPS for all configs × workloads → CSV
#   Phase 2: For each workload, identify top-10 and bottom-10 configs
#   Phase 3: Run `perf stat` on those configs to gather cache/CPU counters
#   Phase 4: Merge performance + perf data into unified training CSV
#
# Usage:
#   ./run_integer_gemm_autotuner_data.sh [--skip-sweep] [--skip-perf]
#
# Options:
#   --skip-sweep   Skip Phase 1 (use existing CSV from previous run)
#   --skip-perf    Skip Phase 3 (don't collect perf counters, just use GFLOPS)
#
# Output:
#   integer_gemm_autotuner_training.csv - Final training data with all features
#
# Author: David Sanftenberg
# Date: November 2025

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SKIP_SWEEP=false
SKIP_PERF=false
PERF_TOP_N=10      # Number of top performers to profile per workload
PERF_BOTTOM_N=10   # Number of bottom performers to profile per workload

# Output files
SWEEP_CSV="integer_gemm_sweep_full.csv"
PERF_CSV="integer_gemm_perf_counters.csv"
TRAINING_CSV="integer_gemm_autotuner_training.csv"
LOG_FILE="integer_gemm_autotuner_data.log"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --skip-sweep)
            SKIP_SWEEP=true
            ;;
        --skip-perf)
            SKIP_PERF=true
            ;;
        --help|-h)
            echo "Usage: $0 [--skip-sweep] [--skip-perf]"
            echo ""
            echo "Options:"
            echo "  --skip-sweep   Skip Phase 1 (use existing $SWEEP_CSV)"
            echo "  --skip-perf    Skip Phase 3 (don't collect perf counters)"
            echo ""
            echo "Output:"
            echo "  $TRAINING_CSV - Training data with GFLOPS + perf counters"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Ensure we're in workspace root
cd "$(dirname "$0")"

echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Integer GEMM Autotuner Data Collection Pipeline${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Pipeline phases:"
echo "  [1] Full config sweep → $SWEEP_CSV"
echo "  [2] Identify top/bottom performers per workload"
echo "  [3] Run 'perf stat' on selected configs → $PERF_CSV"
echo "  [4] Merge data → $TRAINING_CSV"
echo ""

# Start logging
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Started at: $(date)"
echo ""

# ============================================================================
# Phase 1: Full Configuration Sweep
# ============================================================================
if [ "$SKIP_SWEEP" = true ]; then
    echo -e "${YELLOW}[Phase 1] Skipping sweep (using existing $SWEEP_CSV)${NC}"
    
    if [ ! -f "$SWEEP_CSV" ]; then
        echo -e "${RED}Error: $SWEEP_CSV not found (cannot skip sweep without existing data)${NC}"
        exit 1
    fi
    
    SWEEP_LINES=$(wc -l < "$SWEEP_CSV")
    echo "Found $SWEEP_LINES lines in existing sweep CSV"
else
    echo -e "${YELLOW}[Phase 1] Running full configuration sweep${NC}"
    echo "This will test all parameter combinations across all Qwen workloads..."
    echo "Estimated time: 30-60 minutes (depends on system)"
    echo ""
    
    # Clean rebuild for reproducibility
    if [ -d "build_v2_release" ]; then
        echo "Removing existing build_v2_release"
        rm -rf build_v2_release
    fi
    
    echo "Configuring Release build with CTest registration..."
    cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-march=native -mtune=native" > /dev/null 2>&1
    
    echo "Building performance test..."
    cmake --build build_v2_release --target v2_perf_integer_gemm_full_sweep --parallel > /dev/null 2>&1
    
    # Detect CPU topology for informational output
    NUM_SOCKETS=$(lscpu | grep 'Socket(s):' | awk '{print $2}')
    CORES_PER_SOCKET=$(lscpu | grep 'Core(s) per socket:' | awk '{print $4}')
    CORES_PER_SOCKET=${CORES_PER_SOCKET:-1}
    
    echo "CPU: $NUM_SOCKETS socket(s), $CORES_PER_SOCKET cores/socket"
    echo "Running full sweep via CTest (writes directly to CSV file)..."
    echo ""
    
    SWEEP_START=$(date +%s)
    
    # Run full sweep test via CTest
    # Test writes output directly to integer_gemm_sweep_full.csv
    cd build_v2_release
    ctest -R "V2_Perf_IntegerGEMM_FullSweep_AllConfigs" --verbose
    cd ..
    
    # Check if output file was created
    if [ ! -f "integer_gemm_sweep_full.csv" ]; then
        echo -e "${RED}Error: Output file integer_gemm_sweep_full.csv not found${NC}"
        exit 1
    fi
    
    # Move to expected location
    mv integer_gemm_sweep_full.csv "$SWEEP_CSV"
    
    SWEEP_END=$(date +%s)
    SWEEP_DURATION=$((SWEEP_END - SWEEP_START))
    
    SWEEP_LINES=$(wc -l < "$SWEEP_CSV")
    DATA_LINES=$((SWEEP_LINES - 1))
    
    echo -e "${GREEN}✓ Phase 1 complete${NC}"
    echo "  Duration: $((SWEEP_DURATION / 60))m $((SWEEP_DURATION % 60))s"
    echo "  Configs tested: $DATA_LINES"
    echo "  Output: $SWEEP_CSV"
    echo ""
fi

# ============================================================================
# Phase 2: Identify Top/Bottom Performers
# ============================================================================
echo -e "${YELLOW}[Phase 2] Identifying top/bottom performers per workload${NC}"

# Use Python for analysis
python3 << 'EOF'
import sys
import pandas as pd

# Read sweep results
try:
    df = pd.read_csv("integer_gemm_sweep_full.csv")
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)

# Group by workload (m, n, k)
df['workload'] = df['m'].astype(str) + 'x' + df['n'].astype(str) + 'x' + df['k'].astype(str)
workloads = df.groupby('workload')

print(f"Found {len(workloads)} unique workloads")
print()

# For each workload, find top and bottom performers
top_n = 10
bottom_n = 10
selected_configs = []

for workload_name, group in workloads:
    # Sort by GFLOPS
    sorted_group = group.sort_values('gflops', ascending=False)
    
    # Get top N
    top_performers = sorted_group.head(top_n)
    # Get bottom N  
    bottom_performers = sorted_group.tail(bottom_n)
    
    # Mark them
    top_performers = top_performers.copy()
    bottom_performers = bottom_performers.copy()
    top_performers['performance_tier'] = 'top'
    bottom_performers['performance_tier'] = 'bottom'
    
    selected_configs.append(top_performers)
    selected_configs.append(bottom_performers)
    
    print(f"Workload {workload_name}:")
    print(f"  Top-{top_n} GFLOPS: {top_performers['gflops'].min():.2f} - {top_performers['gflops'].max():.2f}")
    print(f"  Bottom-{bottom_n} GFLOPS: {bottom_performers['gflops'].min():.2f} - {bottom_performers['gflops'].max():.2f}")

# Combine all selected configs
selected_df = pd.concat(selected_configs, ignore_index=True)

# Save to temp file for Phase 3
selected_df.to_csv("integer_gemm_selected_for_perf.csv", index=False)

print()
print(f"Total configs selected for perf profiling: {len(selected_df)}")
print(f"Saved to: integer_gemm_selected_for_perf.csv")
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}Error in Phase 2${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Phase 2 complete${NC}"
echo ""

# ============================================================================
# Phase 3: Collect Perf Counters
# ============================================================================
if [ "$SKIP_PERF" = true ]; then
    echo -e "${YELLOW}[Phase 3] Skipping perf counter collection${NC}"
    echo "Will proceed with GFLOPS data only"
    echo ""
    
    # Just copy sweep CSV to training CSV
    cp "$SWEEP_CSV" "$TRAINING_CSV"
    
else
    echo -e "${YELLOW}[Phase 3] Collecting perf counters for selected configs${NC}"
    echo "This uses 'perf stat' to measure cache misses, IPC, etc."
    echo ""
    
    # Check if perf is available
    if ! command -v perf &> /dev/null; then
        echo -e "${RED}Error: 'perf' command not found${NC}"
        echo "Install with: sudo apt-get install linux-tools-generic linux-tools-\$(uname -r)"
        echo "Or run with --skip-perf to use GFLOPS data only"
        exit 1
    fi
    
    # Create perf data collection script
    cat > /tmp/collect_perf_data.py << 'PERF_SCRIPT'
#!/usr/bin/env python3
import os
import subprocess
import pandas as pd
import sys
import re
from pathlib import Path

# Read selected configs
df = pd.read_csv("integer_gemm_selected_for_perf.csv")

print(f"Profiling {len(df)} configurations with perf...")
print()

# Perf counters to collect
# Note: Some counters may not be available on all systems
PERF_EVENTS = [
    'cycles',
    'instructions',
    'cache-references',
    'cache-misses',
    'L1-dcache-loads',
    'L1-dcache-load-misses',
    'LLC-loads',
    'LLC-load-misses',
    'context-switches',
    'cpu-migrations',
    'branch-misses',
]

# Build test executable path
exe_path = Path("build_v2_release/performance/v2_perf_integer_gemm_full_sweep")
if not exe_path.exists():
    print(f"Error: {exe_path} not found")
    sys.exit(1)

# Prepare output
perf_results = []

for idx, row in df.iterrows():
    m, n, k = int(row['m']), int(row['n']), int(row['k'])
    mr, nr = int(row['mr']), int(row['nr'])
    k_blocks, unroll_k, prefetch = int(row['k_blocks']), int(row['unroll_k']), int(row['prefetch_dist'])
    mc, kc, nc = int(row['mc']), int(row['kc']), int(row['nc'])
    
    workload_id = f"{m}x{n}x{k}"
    config_id = f"MR{mr}_KB{k_blocks}_U{unroll_k}_P{prefetch}"
    
    print(f"[{idx+1}/{len(df)}] Profiling {workload_id} with {config_id}...", end=' ')
    sys.stdout.flush()
    
    # Build test filter (matches SingleConfig test)
    # Format: IntegerGEMM_FullSweep.SingleConfig_M<m>_N<n>_K<k>_MR<mr>_...
    test_filter = (f"*SingleConfig_M{m}_N{n}_K{k}_MR{mr}_NR{nr}_"
                  f"KB{k_blocks}_U{unroll_k}_P{prefetch}_"
                  f"MC{mc}_KC{kc}_NC{nc}")
    
    # Run with perf stat
    # NOTE: For Phase 3, we use direct execution because:
    #   1. perf stat needs to wrap the binary directly
    #   2. Only ~240 configs (vs 4,608 in Phase 1)
    #   3. Environment already configured from CTest build
    # The OMP environment is inherited from the shell where this script runs
    
    perf_cmd = [
        'perf', 'stat',
        '-e', ','.join(PERF_EVENTS),
        '--', str(exe_path), f'--gtest_filter={test_filter}'
    ]
    
    try:
        result = subprocess.run(
            perf_cmd,
            capture_output=True,
            text=True,
            timeout=60,  # 60s timeout per config
            env={**os.environ}  # Inherit environment (OMP vars set by user/script caller)
        )
        
        # Parse perf output (stderr)
        perf_output = result.stderr
        
        # Extract counter values
        counters = {}
        for line in perf_output.split('\n'):
            # Format: "      123,456      event-name"
            match = re.match(r'\s*([\d,]+)\s+(\S+)', line)
            if match:
                value_str = match.group(1).replace(',', '')
                event_name = match.group(2)
                
                # Clean event name (remove :u suffix, etc)
                event_name = event_name.split(':')[0]
                
                try:
                    counters[event_name] = int(value_str)
                except ValueError:
                    pass
        
        # Calculate derived metrics
        ipc = counters.get('instructions', 0) / max(counters.get('cycles', 1), 1)
        l1_miss_rate = counters.get('L1-dcache-load-misses', 0) / max(counters.get('L1-dcache-loads', 1), 1)
        llc_miss_rate = counters.get('LLC-load-misses', 0) / max(counters.get('LLC-loads', 1), 1)
        cache_miss_rate = counters.get('cache-misses', 0) / max(counters.get('cache-references', 1), 1)
        
        # Store results
        perf_row = row.to_dict()
        perf_row.update({
            'cycles': counters.get('cycles', 0),
            'instructions': counters.get('instructions', 0),
            'ipc': ipc,
            'cache_references': counters.get('cache-references', 0),
            'cache_misses': counters.get('cache-misses', 0),
            'cache_miss_rate': cache_miss_rate,
            'l1_dcache_loads': counters.get('L1-dcache-loads', 0),
            'l1_dcache_load_misses': counters.get('L1-dcache-load-misses', 0),
            'l1_miss_rate': l1_miss_rate,
            'llc_loads': counters.get('LLC-loads', 0),
            'llc_load_misses': counters.get('LLC-load-misses', 0),
            'llc_miss_rate': llc_miss_rate,
            'context_switches': counters.get('context-switches', 0),
            'cpu_migrations': counters.get('cpu-migrations', 0),
            'branch_misses': counters.get('branch-misses', 0),
        })
        
        perf_results.append(perf_row)
        print(f"IPC={ipc:.2f}, L1_miss={l1_miss_rate:.1%}, LLC_miss={llc_miss_rate:.1%}")
        
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
    except Exception as e:
        print(f"ERROR: {e}")

# Save results
perf_df = pd.DataFrame(perf_results)
perf_df.to_csv("integer_gemm_perf_counters.csv", index=False)

print()
print(f"Collected perf data for {len(perf_df)} configs")
print(f"Saved to: integer_gemm_perf_counters.csv")
PERF_SCRIPT
    
    chmod +x /tmp/collect_perf_data.py
    
    # Run perf collection
    PERF_START=$(date +%s)
    python3 /tmp/collect_perf_data.py
    PERF_END=$(date +%s)
    PERF_DURATION=$((PERF_END - PERF_START))
    
    echo -e "${GREEN}✓ Phase 3 complete${NC}"
    echo "  Duration: $((PERF_DURATION / 60))m $((PERF_DURATION % 60))s"
    echo "  Output: $PERF_CSV"
    echo ""
fi

# ============================================================================
# Phase 4: Merge Data
# ============================================================================
echo -e "${YELLOW}[Phase 4] Merging performance + perf counter data${NC}"

if [ "$SKIP_PERF" = true ]; then
    # Already copied in Phase 3
    echo "Using GFLOPS-only training data"
else
    # Merge sweep data with perf counter data
    python3 << 'EOF'
import pandas as pd

# Read both datasets
sweep_df = pd.read_csv("integer_gemm_sweep_full.csv")
perf_df = pd.read_csv("integer_gemm_perf_counters.csv")

print(f"Sweep data: {len(sweep_df)} rows")
print(f"Perf data: {len(perf_df)} rows")

# Merge on config parameters (m, n, k, mr, nr, k_blocks, ...)
merge_keys = ['m', 'n', 'k', 'mr', 'nr', 'k_blocks', 'unroll_k', 'prefetch_dist', 'mc', 'kc', 'nc']

# Left join (keep all sweep data, add perf where available)
merged_df = sweep_df.merge(perf_df, on=merge_keys, how='left', suffixes=('', '_perf'))

# For configs without perf data, fill with NaN (will be excluded from ML training)
print(f"Merged data: {len(merged_df)} rows")
print(f"  With perf data: {merged_df['ipc'].notna().sum()} rows")
print(f"  Without perf data: {merged_df['ipc'].isna().sum()} rows")

# Save
merged_df.to_csv("integer_gemm_autotuner_training.csv", index=False)
print(f"Saved to: integer_gemm_autotuner_training.csv")
EOF
fi

echo -e "${GREEN}✓ Phase 4 complete${NC}"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Pipeline Complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

TOTAL_LINES=$(wc -l < "$TRAINING_CSV")
echo "Output files:"
echo "  Training data: $TRAINING_CSV ($TOTAL_LINES lines)"
if [ "$SKIP_PERF" != true ]; then
    echo "  Perf counters: $PERF_CSV"
fi
echo "  Log file: $LOG_FILE"
echo ""

echo "Next steps:"
echo "  1. Analyze data:"
echo "     python3 src/v2/kernels/cpu/gemm/python/analyze_autotuner_data.py $TRAINING_CSV"
echo ""
echo "  2. Train heuristic model:"
echo "     python3 src/v2/kernels/cpu/gemm/python/train_integer_gemm_autotuner.py --input $TRAINING_CSV"
echo ""
echo "  3. Or train ML model (if pandas/sklearn/torch installed):"
echo "     python3 src/v2/kernels/cpu/gemm/python/train_integer_gemm_ml.py --input $TRAINING_CSV"
echo ""

echo "Finished at: $(date)"
