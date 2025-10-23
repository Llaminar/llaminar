#!/bin/bash
# BF16 tile size sweep - tests key tile sizes for BF16 activation path
# For full sweep, adjust TILE_CONFIGS

set -e

BENCHMARK_EXE="./build_release/benchmark_iq4nl_gemm"
RESULTS_DIR="tile_sweep_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/bf16_sweep_${TIMESTAMP}.csv"

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Initialize CSV with header
echo "M_TILE,N_TILE,Workload,m,n,k,GFLOPS,Time_ms" > "${RESULTS_FILE}"

# Quick sweep: test key tile sizes only
TILE_CONFIGS="32,32 48,48 64,64 96,96 128,128 32,64 64,32 48,96 96,48"

# Always enable microkernel for these tests
export LLAMINAR_IQ4_GEMM_MICROKERNEL=1

echo "=================================="
echo "IQ4_NL GEMM BF16 Tile Sweep"
echo "=================================="
echo "Microkernel: ENABLED"
echo "Testing BF16 activation path"
echo "Testing ${TILE_CONFIGS// /,} configurations"
echo "Results will be saved to: ${RESULTS_FILE}"
echo ""

# Function to extract BF16 metric from benchmark output for a specific test
# Usage: extract_metric "Prefill (1024 tokens" "Throughput"
extract_metric() {
    local pattern="$1"
    local metric="$2"
    # Find the "Running:" line, then look for BF16 section, extract metric
    # Use index() instead of regex to avoid escaping issues
    awk -v pat="$pattern" -v met="$metric" '
        index($0, pat) > 0 { found=1; next }
        found && /BF16 Activation Path/ { in_bf16=1; next }
        in_bf16 && index($0, met) > 0 {
            match($0, /[0-9]+\.[0-9]+/)
            print substr($0, RSTART, RLENGTH)
            exit
        }
    '
}

# Function to run benchmark with specific tile settings
run_tile_config() {
    local m_tile=$1
    local n_tile=$2
    
    echo "Testing M_TILE=${m_tile}, N_TILE=${n_tile}"
    
    export LLAMINAR_IQ4_M_TILE_BF16=${m_tile}
    export LLAMINAR_IQ4_N_TILE_BF16=${n_tile}
    
    # Run benchmark and capture output
    local output=$(./run_benchmark.sh benchmark_iq4nl_gemm 2>&1)
    
    # Extract key workloads only (not exhaustive)
    local qproj_1024=$(echo "$output" | extract_metric "Prefill (1024 tokens, Q Projection" "Throughput")
    local qproj_1024_time=$(echo "$output" | extract_metric "Prefill (1024 tokens, Q Projection" "Time per iter")
    local qproj_4096=$(echo "$output" | extract_metric "Prefill (4096 tokens, Q Projection" "Throughput")
    local qproj_4096_time=$(echo "$output" | extract_metric "Prefill (4096 tokens, Q Projection" "Time per iter")
    local ffn_b16=$(echo "$output" | extract_metric "Batch FFN Gate (16 tokens" "Throughput")
    local ffn_b16_time=$(echo "$output" | extract_metric "Batch FFN Gate (16 tokens" "Time per iter")
    local ffn_b256=$(echo "$output" | extract_metric "Batch FFN Gate (256 tokens" "Throughput")
    local ffn_b256_time=$(echo "$output" | extract_metric "Batch FFN Gate (256 tokens" "Time per iter")
    
    # Write key results to CSV
    echo "${m_tile},${n_tile},Q-Proj-1024,1024,896,896,${qproj_1024},${qproj_1024_time}" >> "${RESULTS_FILE}"
    echo "${m_tile},${n_tile},Q-Proj-4096,4096,896,896,${qproj_4096},${qproj_4096_time}" >> "${RESULTS_FILE}"
    echo "${m_tile},${n_tile},FFN-Batch-16,16,4864,2048,${ffn_b16},${ffn_b16_time}" >> "${RESULTS_FILE}"
    echo "${m_tile},${n_tile},FFN-Batch-256,256,4864,2048,${ffn_b256},${ffn_b256_time}" >> "${RESULTS_FILE}"
    
    echo "  -> Q-Proj 1024: ${qproj_1024} GFLOPS, Q-Proj 4096: ${qproj_4096} GFLOPS"
    echo "  -> FFN B16: ${ffn_b16} GFLOPS, FFN B256: ${ffn_b256} GFLOPS"
}

# Parse and run tile configurations
config_count=0
for config in ${TILE_CONFIGS}; do
    config_count=$((config_count + 1))
done

echo "Will test ${config_count} tile configurations"
echo ""

current=0
for config in ${TILE_CONFIGS}; do
    current=$((current + 1))
    m_tile=$(echo $config | cut -d',' -f1)
    n_tile=$(echo $config | cut -d',' -f2)
    echo "[${current}/${config_count}] Testing M_TILE=${m_tile}, N_TILE=${n_tile}"
    run_tile_config ${m_tile} ${n_tile}
    echo ""
done

echo "=================================="
echo "BF16 sweep complete!"
echo "Results saved to: ${RESULTS_FILE}"
echo "=================================="
echo ""
echo "Best configurations:"
echo ""
echo "Q-Proj 4096:"
grep "Q-Proj-4096" "${RESULTS_FILE}" | sort -t',' -k7 -rn | head -3 | column -t -s','
echo ""
echo "FFN Batch 16:"
grep "FFN-Batch-16" "${RESULTS_FILE}" | sort -t',' -k7 -rn | head -3 | column -t -s','
echo ""
echo "FFN Batch 256:"
grep "FFN-Batch-256" "${RESULTS_FILE}" | sort -t',' -k7 -rn | head -3 | column -t -s','
echo ""
echo "For detailed analysis: python3 analyze_tile_sweep.py ${RESULTS_FILE}"
