#!/bin/bash
# Combined FP32 + BF16 tile sweep - tests both activation paths
# Compares FP32 vs BF16 performance with same tile settings

set -e

BENCHMARK_EXE="./build_release/benchmark_iq4nl_gemm"
RESULTS_DIR="tile_sweep_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FP32_FILE="${RESULTS_DIR}/combined_fp32_${TIMESTAMP}.csv"
BF16_FILE="${RESULTS_DIR}/combined_bf16_${TIMESTAMP}.csv"

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Initialize CSVs with headers
echo "M_TILE,N_TILE,Workload,m,n,k,GFLOPS,Time_ms" > "${FP32_FILE}"
echo "M_TILE,N_TILE,Workload,m,n,k,GFLOPS,Time_ms" > "${BF16_FILE}"

# Quick sweep: test key tile sizes
TILE_CONFIGS="32,32 48,48 64,64 96,96 128,128 32,64 64,32 48,96 96,48"

# Always enable microkernel for these tests
export LLAMINAR_IQ4_GEMM_MICROKERNEL=1

echo "=============================================="
echo "IQ4_NL GEMM Combined FP32/BF16 Tile Sweep"
echo "=============================================="
echo "Microkernel: ENABLED"
echo "Testing both FP32 and BF16 activation paths"
echo "Testing ${TILE_CONFIGS// /,} configurations"
echo "FP32 Results: ${FP32_FILE}"
echo "BF16 Results: ${BF16_FILE}"
echo ""

# Function to extract FP32 metric
extract_fp32_metric() {
    local pattern="$1"
    local metric="$2"
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

# Function to extract BF16 metric
extract_bf16_metric() {
    local pattern="$1"
    local metric="$2"
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
    
    # Set both FP32 and BF16 tiles to same values
    export LLAMINAR_IQ4_M_TILE=${m_tile}
    export LLAMINAR_IQ4_N_TILE=${n_tile}
    export LLAMINAR_IQ4_M_TILE_BF16=${m_tile}
    export LLAMINAR_IQ4_N_TILE_BF16=${n_tile}
    
    # Run benchmark once and capture output
    local output=$(./run_benchmark.sh benchmark_iq4nl_gemm 2>&1)
    
    # Extract FP32 results
    local fp32_q1024=$(echo "$output" | extract_fp32_metric "Prefill (1024 tokens, Q Projection" "Throughput")
    local fp32_q1024_time=$(echo "$output" | extract_fp32_metric "Prefill (1024 tokens, Q Projection" "Time per iter")
    local fp32_q4096=$(echo "$output" | extract_fp32_metric "Prefill (4096 tokens, Q Projection" "Throughput")
    local fp32_q4096_time=$(echo "$output" | extract_fp32_metric "Prefill (4096 tokens, Q Projection" "Time per iter")
    local fp32_ffn16=$(echo "$output" | extract_fp32_metric "Batch FFN Gate (16 tokens" "Throughput")
    local fp32_ffn16_time=$(echo "$output" | extract_fp32_metric "Batch FFN Gate (16 tokens" "Time per iter")
    local fp32_ffn256=$(echo "$output" | extract_fp32_metric "Batch FFN Gate (256 tokens" "Throughput")
    local fp32_ffn256_time=$(echo "$output" | extract_fp32_metric "Batch FFN Gate (256 tokens" "Time per iter")
    
    # Extract BF16 results
    local bf16_q1024=$(echo "$output" | extract_bf16_metric "Prefill (1024 tokens, Q Projection" "Throughput")
    local bf16_q1024_time=$(echo "$output" | extract_bf16_metric "Prefill (1024 tokens, Q Projection" "Time per iter")
    local bf16_q4096=$(echo "$output" | extract_bf16_metric "Prefill (4096 tokens, Q Projection" "Throughput")
    local bf16_q4096_time=$(echo "$output" | extract_bf16_metric "Prefill (4096 tokens, Q Projection" "Time per iter")
    local bf16_ffn16=$(echo "$output" | extract_bf16_metric "Batch FFN Gate (16 tokens" "Throughput")
    local bf16_ffn16_time=$(echo "$output" | extract_bf16_metric "Batch FFN Gate (16 tokens" "Time per iter")
    local bf16_ffn256=$(echo "$output" | extract_bf16_metric "Batch FFN Gate (256 tokens" "Throughput")
    local bf16_ffn256_time=$(echo "$output" | extract_bf16_metric "Batch FFN Gate (256 tokens" "Time per iter")
    
    # Write FP32 results
    echo "${m_tile},${n_tile},Q-Proj-1024,1024,896,896,${fp32_q1024},${fp32_q1024_time}" >> "${FP32_FILE}"
    echo "${m_tile},${n_tile},Q-Proj-4096,4096,896,896,${fp32_q4096},${fp32_q4096_time}" >> "${FP32_FILE}"
    echo "${m_tile},${n_tile},FFN-Batch-16,16,4864,2048,${fp32_ffn16},${fp32_ffn16_time}" >> "${FP32_FILE}"
    echo "${m_tile},${n_tile},FFN-Batch-256,256,4864,2048,${fp32_ffn256},${fp32_ffn256_time}" >> "${FP32_FILE}"
    
    # Write BF16 results
    echo "${m_tile},${n_tile},Q-Proj-1024,1024,896,896,${bf16_q1024},${bf16_q1024_time}" >> "${BF16_FILE}"
    echo "${m_tile},${n_tile},Q-Proj-4096,4096,896,896,${bf16_q4096},${bf16_q4096_time}" >> "${BF16_FILE}"
    echo "${m_tile},${n_tile},FFN-Batch-16,16,4864,2048,${bf16_ffn16},${bf16_ffn16_time}" >> "${BF16_FILE}"
    echo "${m_tile},${n_tile},FFN-Batch-256,256,4864,2048,${bf16_ffn256},${bf16_ffn256_time}" >> "${BF16_FILE}"
    
    echo "  FP32 -> Q-Proj 1024: ${fp32_q1024} GFLOPS, Q-Proj 4096: ${fp32_q4096} GFLOPS"
    echo "  FP32 -> FFN B16: ${fp32_ffn16} GFLOPS, FFN B256: ${fp32_ffn256} GFLOPS"
    echo "  BF16 -> Q-Proj 1024: ${bf16_q1024} GFLOPS, Q-Proj 4096: ${bf16_q4096} GFLOPS"
    echo "  BF16 -> FFN B16: ${bf16_ffn16} GFLOPS, FFN B256: ${bf16_ffn256} GFLOPS"
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

echo "=============================================="
echo "Combined sweep complete!"
echo "=============================================="
echo ""
echo "FP32 Results: ${FP32_FILE}"
echo "BF16 Results: ${BF16_FILE}"
echo ""
echo "Analyze with:"
echo "  python3 analyze_tile_sweep.py ${FP32_FILE}"
echo "  python3 analyze_tile_sweep.py ${BF16_FILE}"
echo ""
echo "Quick comparison (FP32 best configs):"
grep "Q-Proj-4096" "${FP32_FILE}" | sort -t',' -k7 -rn | head -3 | column -t -s','
echo ""
echo "Quick comparison (BF16 best configs):"
grep "Q-Proj-4096" "${BF16_FILE}" | sort -t',' -k7 -rn | head -3 | column -t -s','
