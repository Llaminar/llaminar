#!/bin/bash
# Phase 5+ Performance Benchmarking: Memory Usage and Throughput
# Compares FP32 baseline vs Phase 5 (BF16 activations) vs Phase 5+ (BF16 activations + KV cache)

set -euo pipefail

# Configuration
MODEL="${1:-models/qwen2.5-0.5b-instruct-q8_0.gguf}"
BUILD_DIR="${2:-build_release}"
OUTPUT_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Detect CPU topology (from run_llaminar.sh)
detect_cpu_topology() {
    local physical_ids=$(grep "^physical id" /proc/cpuinfo | awk '{print $NF}' | sort -u | wc -l)
    local total_cores=$(grep "^processor" /proc/cpuinfo | wc -l)
    local unique_cores=$(awk '
        /^processor/ { proc = $NF }
        /^physical id/ { phys_id = $NF }
        /^core id/ { core_id = $NF; print phys_id ":" core_id }
    ' /proc/cpuinfo | sort -u | wc -l)
    
    SOCKETS=$physical_ids
    PHYSICAL_CORES=$unique_cores
    TOTAL_CORES=$total_cores
    CORES_PER_SOCKET=$((PHYSICAL_CORES / SOCKETS))
    OMP_THREADS=$CORES_PER_SOCKET
}

detect_cpu_topology

# Canonical OpenMP settings (from run_llaminar.sh)
export OMP_NUM_THREADS=$OMP_THREADS
export OMP_PLACES=sockets
export OMP_PROC_BIND=close
export OMP_NESTED=false
export OMP_DYNAMIC=false
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0
export MKL_NUM_THREADS=$OMP_THREADS
export MKL_DYNAMIC=false

# Canonical MPI settings (from run_llaminar.sh)
export OMPI_MCA_mpi_leave_pinned=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_btl_openib_allow_ib=1

# OpenBLAS settings
export OPENBLAS_NUM_THREADS=$OMP_THREADS
export GOTO_NUM_THREADS=$OMP_THREADS

# MPI process count (one per socket)
MPI_PROCS=$SOCKETS

# Test configurations - Reduced set for quick validation
# prompt=8: decode 10, 50, 100
# prompt=64: decode 10, 50 only
PROMPT_LENGTHS=(8 64)
DECODE_LENGTHS=(10 50 100)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ Phase 5+ Performance Benchmark Suite                        ║${NC}"
echo -e "${BLUE}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║ Model: $(basename "$MODEL")${NC}"
echo -e "${BLUE}║ Timestamp: ${TIMESTAMP}${NC}"
echo -e "${BLUE}║ System: ${SOCKETS} sockets, ${CORES_PER_SOCKET} cores/socket${NC}"
echo -e "${BLUE}║ MPI: ${MPI_PROCS} processes, OMP: ${OMP_THREADS} threads${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
RESULTS_FILE="$OUTPUT_DIR/phase5plus_benchmark_${TIMESTAMP}.txt"

# Function to extract memory usage from benchmark output
extract_memory() {
    local output="$1"
    echo "$output" | grep -E "Memory Usage:" | sed -E 's/.*Memory Usage:[[:space:]]+([0-9.]+) MB.*/\1/' | head -1
}

# Function to extract throughput
extract_throughput() {
    local output="$1"
    local phase="$2"  # "prefill" or "decode"
    
    if [ "$phase" == "prefill" ]; then
        echo "$output" | grep -A3 "PREFILL PHASE" | grep "Throughput:" | sed -E 's/.*Throughput:[[:space:]]+([0-9.]+) tok\/s.*/\1/'
    else
        echo "$output" | grep -A3 "DECODE PHASE" | grep "Throughput:" | sed -E 's/.*Throughput:[[:space:]]+([0-9.]+) tok\/s.*/\1/'
    fi
}

# Function to run benchmark configuration
run_benchmark() {
    local config_name="$1"
    local env_vars="$2"
    local prompt_len="$3"
    local decode_len="$4"
    
    echo -e "${YELLOW}Testing: ${config_name} (prompt=${prompt_len}, decode=${decode_len})${NC}"
    
    # Generate prompt of desired length
    local prompt=""
    if [ "$prompt_len" -eq 8 ]; then
        prompt="Explain machine learning"
    elif [ "$prompt_len" -eq 64 ]; then
        prompt="Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data without being explicitly programmed"
    elif [ "$prompt_len" -eq 256 ]; then
        prompt="Machine learning is a rapidly evolving field of artificial intelligence that focuses on developing algorithms and statistical models that enable computers to learn and improve from experience without being explicitly programmed. The field encompasses various approaches including supervised learning where models learn from labeled data, unsupervised learning which finds patterns in unlabeled data, and reinforcement learning where agents learn through trial and error by receiving rewards or penalties. Deep learning, a subset of machine learning using neural networks with multiple layers, has achieved remarkable success in areas like computer vision, natural language processing, and speech recognition, revolutionizing industries from healthcare to autonomous vehicles."
    else
        # 512 tokens - generate longer prompt
        prompt="Machine learning is a rapidly evolving field of artificial intelligence that focuses on developing algorithms and statistical models that enable computers to learn and improve from experience without being explicitly programmed. The field encompasses various approaches including supervised learning where models learn from labeled data, unsupervised learning which finds patterns in unlabeled data, and reinforcement learning where agents learn through trial and error by receiving rewards or penalties. Deep learning, a subset of machine learning using neural networks with multiple layers, has achieved remarkable success in areas like computer vision, natural language processing, and speech recognition, revolutionizing industries from healthcare to autonomous vehicles. Modern machine learning systems leverage massive datasets and powerful computational resources including GPUs and TPUs to train increasingly sophisticated models. Transfer learning allows practitioners to adapt pre-trained models to new tasks with limited data, while techniques like data augmentation and regularization help prevent overfitting and improve generalization. The ethical implications of machine learning systems including fairness, transparency, and accountability have become critical considerations as these technologies are deployed in high-stakes domains like criminal justice, healthcare, and financial services. Researchers continue to push the boundaries of what's possible with innovations in areas like few-shot learning, neural architecture search, and explainable AI, making machine learning more accessible, efficient, and trustworthy for real-world applications across diverse industries and scientific disciplines."
    fi
    
    # Run benchmark with proper MPI/OpenMP settings
    local output
    output=$(eval "$env_vars" timeout 300 mpirun -np ${MPI_PROCS} \
        --bind-to socket \
        --map-by socket \
        --mca mpi_leave_pinned 1 \
        --mca btl_vader_single_copy_mechanism none \
        "${BUILD_DIR}/llaminar" \
        --model "$MODEL" \
        --prompt "$prompt" \
        -n "$decode_len" \
        --benchmark 2>&1 || echo "TIMEOUT_OR_ERROR")
    
    # Extract metrics
    local memory=$(extract_memory "$output")
    local prefill_tps=$(extract_throughput "$output" "prefill")
    local decode_tps=$(extract_throughput "$output" "decode")
    
    # Handle errors
    if [ -z "$memory" ]; then memory="ERROR"; fi
    if [ -z "$prefill_tps" ]; then prefill_tps="ERROR"; fi
    if [ -z "$decode_tps" ]; then decode_tps="ERROR"; fi
    
    # Store results
    echo "$config_name,$prompt_len,$decode_len,$memory,$prefill_tps,$decode_tps" >> "$RESULTS_FILE.csv"
    
    # Print results
    if [ "$memory" != "ERROR" ]; then
        echo -e "  Memory: ${GREEN}${memory} MB${NC}"
        echo -e "  Prefill: ${GREEN}${prefill_tps} tok/s${NC}"
        echo -e "  Decode: ${GREEN}${decode_tps} tok/s${NC}"
    else
        echo -e "  ${RED}ERROR: Benchmark failed${NC}"
    fi
    echo ""
}

# Initialize CSV
echo "Configuration,PromptLength,DecodeLength,MemoryMB,PrefillTPS,DecodeTPS" > "$RESULTS_FILE.csv"

# Write header to results file
{
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║ Phase 5+ Performance Benchmark Results                      ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║ Model: $(basename "$MODEL")"
    echo "║ Date: $(date)"
    echo "║ System: $(uname -n)"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
} > "$RESULTS_FILE"

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Test Configuration 1: FP32 Baseline (No BF16)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for decode_len in "${DECODE_LENGTHS[@]}"; do
        # Skip prompt=64 with decode=100
        if [ "$prompt_len" -eq 64 ] && [ "$decode_len" -eq 100 ]; then
            continue
        fi
        run_benchmark "FP32_Baseline" \
            "LLAMINAR_KV_BF16=0 LLAMINAR_QUANT_OUTPUT_BF16=0" \
            "$prompt_len" "$decode_len"
    done
done

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Test Configuration 2: Phase 5 (BF16 Activations Only)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for decode_len in "${DECODE_LENGTHS[@]}"; do
        # Skip prompt=64 with decode=100
        if [ "$prompt_len" -eq 64 ] && [ "$decode_len" -eq 100 ]; then
            continue
        fi
        run_benchmark "Phase5_BF16_Activations" \
            "LLAMINAR_KV_BF16=0 LLAMINAR_QUANT_OUTPUT_BF16=1 LLAMINAR_FORCE_FP32_SOFTMAX=1 LLAMINAR_FORCE_FP32_RMSNORM=1 LLAMINAR_FORCE_FP32_LOGITS=1" \
            "$prompt_len" "$decode_len"
    done
done

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Test Configuration 3: Phase 5+ (BF16 Activations + KV Cache)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

for prompt_len in "${PROMPT_LENGTHS[@]}"; do
    for decode_len in "${DECODE_LENGTHS[@]}"; do
        # Skip prompt=64 with decode=100
        if [ "$prompt_len" -eq 64 ] && [ "$decode_len" -eq 100 ]; then
            continue
        fi
        run_benchmark "Phase5Plus_BF16_Full" \
            "LLAMINAR_KV_BF16=1 LLAMINAR_QUANT_OUTPUT_BF16=1 LLAMINAR_FORCE_FP32_SOFTMAX=1 LLAMINAR_FORCE_FP32_RMSNORM=1 LLAMINAR_FORCE_FP32_LOGITS=1" \
            "$prompt_len" "$decode_len"
    done
done

# Generate summary report
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ Generating Summary Report                                   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Python script to analyze results
python3 << 'EOF' > "$OUTPUT_DIR/phase5plus_analysis_${TIMESTAMP}.txt"
import sys
import pandas as pd
from pathlib import Path

# Read CSV
csv_file = Path("$RESULTS_FILE.csv")
if not csv_file.exists():
    print("ERROR: CSV file not found")
    sys.exit(1)

df = pd.read_csv(csv_file)

# Filter out errors
df = df[df['MemoryMB'] != 'ERROR']
df['MemoryMB'] = pd.to_numeric(df['MemoryMB'])
df['PrefillTPS'] = pd.to_numeric(df['PrefillTPS'])
df['DecodeTPS'] = pd.to_numeric(df['DecodeTPS'])

# Calculate averages by configuration
summary = df.groupby('Configuration').agg({
    'MemoryMB': 'mean',
    'PrefillTPS': 'mean',
    'DecodeTPS': 'mean'
}).round(2)

print("╔══════════════════════════════════════════════════════════════╗")
print("║ SUMMARY: Average Metrics by Configuration                   ║")
print("╠══════════════════════════════════════════════════════════════╣")
print(summary.to_string())
print("╚══════════════════════════════════════════════════════════════╝")
print("")

# Calculate memory savings
baseline_mem = summary.loc['FP32_Baseline', 'MemoryMB']
phase5_mem = summary.loc['Phase5_BF16_Activations', 'MemoryMB']
phase5plus_mem = summary.loc['Phase5Plus_BF16_Full', 'MemoryMB']

phase5_savings = ((baseline_mem - phase5_mem) / baseline_mem) * 100
phase5plus_savings = ((baseline_mem - phase5plus_mem) / baseline_mem) * 100

print("╔══════════════════════════════════════════════════════════════╗")
print("║ MEMORY SAVINGS ANALYSIS                                      ║")
print("╠══════════════════════════════════════════════════════════════╣")
print(f"║ FP32 Baseline:         {baseline_mem:8.2f} MB                       ║")
print(f"║ Phase 5 (BF16 Act):    {phase5_mem:8.2f} MB  ({phase5_savings:5.1f}% savings)      ║")
print(f"║ Phase 5+ (BF16 Full):  {phase5plus_mem:8.2f} MB  ({phase5plus_savings:5.1f}% savings)      ║")
print("╚══════════════════════════════════════════════════════════════╝")
print("")

# Throughput comparison
baseline_prefill = summary.loc['FP32_Baseline', 'PrefillTPS']
baseline_decode = summary.loc['FP32_Baseline', 'DecodeTPS']

phase5plus_prefill = summary.loc['Phase5Plus_BF16_Full', 'PrefillTPS']
phase5plus_decode = summary.loc['Phase5Plus_BF16_Full', 'DecodeTPS']

prefill_ratio = (phase5plus_prefill / baseline_prefill) * 100
decode_ratio = (phase5plus_decode / baseline_decode) * 100

print("╔══════════════════════════════════════════════════════════════╗")
print("║ THROUGHPUT COMPARISON (vs FP32 Baseline)                    ║")
print("╠══════════════════════════════════════════════════════════════╣")
print(f"║ Prefill:  {baseline_prefill:6.2f} tok/s (baseline) → {phase5plus_prefill:6.2f} tok/s ({prefill_ratio:5.1f}%)  ║")
print(f"║ Decode:   {baseline_decode:6.2f} tok/s (baseline) → {phase5plus_decode:6.2f} tok/s ({decode_ratio:5.1f}%)  ║")
print("╚══════════════════════════════════════════════════════════════╝")

EOF

# Display summary
cat "$OUTPUT_DIR/phase5plus_analysis_${TIMESTAMP}.txt"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║ Benchmark Complete!                                         ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║ Results saved to:${NC}"
echo -e "${GREEN}║   ${RESULTS_FILE}${NC}"
echo -e "${GREEN}║   ${RESULTS_FILE}.csv${NC}"
echo -e "${GREEN}║   ${OUTPUT_DIR}/phase5plus_analysis_${TIMESTAMP}.txt${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
