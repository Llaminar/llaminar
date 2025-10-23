#!/bin/bash
# Tile size sweep benchmark for IQ4_NL GEMM with microkernel enabled
# Tests various tile sizes across different workload shapes

set -e

BENCHMARK_EXE="./build_release/benchmark_iq4nl_gemm"
RESULTS_DIR="tile_sweep_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/tile_sweep_${TIMESTAMP}.csv"

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Initialize CSV with header
echo "M_TILE,N_TILE,Workload,m,n,k,GFLOPS,Time_ms" > "${RESULTS_FILE}"

# Tile sizes to test (powers of 2 and some intermediate values)
TILE_SIZES="16 24 32 48 64 96 128 192 256"

# Always enable microkernel for these tests
export LLAMINAR_IQ4_GEMM_MICROKERNEL=1

echo "=================================="
echo "IQ4_NL GEMM Tile Size Sweep Test"
echo "=================================="
echo "Microkernel: ENABLED"
echo "Results will be saved to: ${RESULTS_FILE}"
echo ""

# Function to extract FP32 metric from benchmark output for a specific test
# Usage: extract_metric "Prefill (1024 tokens" "Throughput"
extract_metric() {
    local pattern="$1"
    local metric="$2"
    # Find the "Running:" line, then look for FP32 section, extract metric
    # Use index() instead of regex to avoid escaping issues
    awk -v pat="$pattern" -v met="$metric" '
        index($0, pat) > 0 { found=1; next }
        found && /FP32 Activation Path/ { in_fp32=1; next }
        in_fp32 && index($0, met) > 0 {
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
    
    export LLAMINAR_IQ4_M_TILE=${m_tile}
    export LLAMINAR_IQ4_N_TILE=${n_tile}
    
    # Run benchmark and capture output
    local output=$(./run_benchmark.sh benchmark_iq4nl_gemm 2>&1)
    
    # Extract results for each workload
    # Q-Proj workloads (square-ish)
    local qproj_512=$(echo "$output" | extract_metric "Prefill (512 tokens, Q Projection" "Throughput")
    local qproj_512_time=$(echo "$output" | extract_metric "Prefill (512 tokens, Q Projection" "Time per iter")
    local qproj_1024=$(echo "$output" | extract_metric "Prefill (1024 tokens, Q Projection" "Throughput")
    local qproj_1024_time=$(echo "$output" | extract_metric "Prefill (1024 tokens, Q Projection" "Time per iter")
    local qproj_2048=$(echo "$output" | extract_metric "Prefill (2048 tokens, Q Projection" "Throughput")
    local qproj_2048_time=$(echo "$output" | extract_metric "Prefill (2048 tokens, Q Projection" "Time per iter")
    local qproj_4096=$(echo "$output" | extract_metric "Prefill (4096 tokens, Q Projection" "Throughput")
    local qproj_4096_time=$(echo "$output" | extract_metric "Prefill (4096 tokens, Q Projection" "Time per iter")
    local qproj_8192=$(echo "$output" | extract_metric "Prefill (8192 tokens, Q Projection" "Throughput")
    local qproj_8192_time=$(echo "$output" | extract_metric "Prefill (8192 tokens, Q Projection" "Time per iter")
    
    # Single token (very small)
    local single_token=$(echo "$output" | extract_metric "Single Token Decode (Q Projection" "Throughput")
    local single_token_time=$(echo "$output" | extract_metric "Single Token Decode (Q Projection" "Time per iter")
    
    # FFN workloads (wide)
    local ffn_b16=$(echo "$output" | extract_metric "Batch FFN Gate (16 tokens" "Throughput")
    local ffn_b16_time=$(echo "$output" | extract_metric "Batch FFN Gate (16 tokens" "Time per iter")
    local ffn_b64=$(echo "$output" | extract_metric "Batch FFN Gate (64 tokens" "Throughput")
    local ffn_b64_time=$(echo "$output" | extract_metric "Batch FFN Gate (64 tokens" "Time per iter")
    local ffn_b128=$(echo "$output" | extract_metric "Batch FFN Gate (128 tokens" "Throughput")
    local ffn_b128_time=$(echo "$output" | extract_metric "Batch FFN Gate (128 tokens" "Time per iter")
    local ffn_b256=$(echo "$output" | extract_metric "Batch FFN Gate (256 tokens" "Throughput")
    local ffn_b256_time=$(echo "$output" | extract_metric "Batch FFN Gate (256 tokens" "Time per iter")
    
    # Write results to CSV
    echo "${m_tile},${n_tile},Q-Proj-512,512,896,896,${qproj_512},${qproj_512_time}" >> "${RESULTS_FILE}"
    echo "${m_tile},${n_tile},Q-Proj-1024,1024,896,896,${qproj_1024},${qproj_1024_time}" >> "${RESULTS_FILE}"
    echo "${m_tile},${n_tile},Q-Proj-2048,2048,896,896,${qproj_2048},${qproj_2048_time}" >> "${RESULTS_FILE}"
    echo "${m_tile},${n_tile},Q-Proj-4096,4096,896,896,${qproj_4096},${qproj_4096_time}" >> "${RESULTS_FILE}"
    echo "${m_tile},${n_tile},Q-Proj-8192,8192,896,896,${qproj_8192},${qproj_8192_time}" >> "${RESULTS_FILE}"
    echo "${m_tile},${n_tile},Single-Token,1,896,896,${single_token},${single_token_time}" >> "${RESULTS_FILE}"
    echo "${m_tile},${n_tile},FFN-Batch-16,16,4864,2048,${ffn_b16},${ffn_b16_time}" >> "${RESULTS_FILE}"
    echo "${m_tile},${n_tile},FFN-Batch-64,64,4864,2048,${ffn_b64},${ffn_b64_time}" >> "${RESULTS_FILE}"
    echo "${m_tile},${n_tile},FFN-Batch-128,128,4864,2048,${ffn_b128},${ffn_b128_time}" >> "${RESULTS_FILE}"
    echo "${m_tile},${n_tile},FFN-Batch-256,256,4864,2048,${ffn_b256},${ffn_b256_time}" >> "${RESULTS_FILE}"
    
    echo "  -> Wrote 10 results to CSV"
}

# Test all tile size combinations
total_configs=0
for m_tile in ${TILE_SIZES}; do
    for n_tile in ${TILE_SIZES}; do
        total_configs=$((total_configs + 1))
    done
done

echo "Will test ${total_configs} tile configurations across 10 workloads (${total_configs}0 total runs)"
echo ""

current_config=0
for m_tile in ${TILE_SIZES}; do
    for n_tile in ${TILE_SIZES}; do
        current_config=$((current_config + 1))
        echo "[${current_config}/${total_configs}] Testing M_TILE=${m_tile}, N_TILE=${n_tile}"
        run_tile_config ${m_tile} ${n_tile}
    done
done

echo ""
echo "=================================="
echo "Sweep test complete!"
echo "Results saved to: ${RESULTS_FILE}"
echo "=================================="
echo ""
echo "To analyze results, you can use:"
echo "  python3 analyze_tile_sweep.py ${RESULTS_FILE}"
echo ""
echo "Quick peek at best configurations:"
echo ""
echo "Best for Q-Proj 4096:"
grep "Q-Proj-4096" "${RESULTS_FILE}" | sort -t',' -k7 -rn | head -3 | column -t -s','
echo ""
echo "Best for FFN Batch 16:"
grep "FFN-Batch-16" "${RESULTS_FILE}" | sort -t',' -k7 -rn | head -3 | column -t -s','
echo ""
echo "Best for FFN Batch 256:"
grep "FFN-Batch-256" "${RESULTS_FILE}" | sort -t',' -k7 -rn | head -3 | column -t -s','
