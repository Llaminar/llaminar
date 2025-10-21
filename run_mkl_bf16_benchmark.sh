#!/bin/bash
###############################################################################
# MKL BF16 End-to-End Performance Benchmark
# 
# Compares MKL BF16 backend vs BF16→FP32 fallback on real model inference
# Tests both prefill and decode phases with various sequence lengths
#
# Usage:
#   ./run_mkl_bf16_benchmark.sh [model_path]
#
# Requirements:
#   - llaminar built with USE_MKL=ON (in build_mkl/)
#   - Quantized model (Q4_0, Q6_K, or Q8_0)
#   - 2-rank MPI environment
###############################################################################

set -euo pipefail

# Configuration
MODEL="${1:-models/qwen2.5-0.5b-instruct-q8_0.gguf}"
BUILD_DIR="build_mkl"
RESULTS_DIR="bf16_benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="${RESULTS_DIR}/mkl_vs_fallback_${TIMESTAMP}.md"

# Test configurations (sequence lengths for prefill)
PREFILL_LENGTHS=(8 64 256 512)
DECODE_TOKENS=50

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create results directory
mkdir -p "${RESULTS_DIR}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         MKL BF16 End-to-End Performance Benchmark             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Model: ${MODEL}"
echo "Build: ${BUILD_DIR}"
echo "Results: ${RESULT_FILE}"
echo ""

# Check prerequisites
if [ ! -f "${BUILD_DIR}/llaminar" ]; then
    echo -e "${RED}Error: llaminar not found in ${BUILD_DIR}/${NC}"
    echo "Please build with: cmake -B ${BUILD_DIR} -S . -DUSE_MKL=ON && cmake --build ${BUILD_DIR} --parallel"
    exit 1
fi

if [ ! -f "${MODEL}" ]; then
    echo -e "${RED}Error: Model not found: ${MODEL}${NC}"
    echo "Please provide a valid model path as first argument"
    exit 1
fi

# Verify MKL linkage
echo "Verifying MKL linkage..."
if ldd "${BUILD_DIR}/llaminar" | grep -q libmkl; then
    echo -e "${GREEN}✓ MKL libraries found${NC}"
else
    echo -e "${YELLOW}⚠ Warning: MKL libraries not detected - is USE_MKL=ON?${NC}"
fi

# Initialize results file
cat > "${RESULT_FILE}" << 'EOF'
# MKL BF16 vs FP32 Fallback Performance Comparison

**Date**: $(date +"%Y-%m-%d %H:%M:%S")  
**Model**: $(basename "$MODEL")  
**Hardware**: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)  
**CPU Features**: $(lscpu | grep "Flags" | grep -o "avx[^ ]*" | tr '\n' ' ')

## Test Configuration

- **Build**: ${BUILD_DIR} (MKL enabled)
- **MPI Ranks**: 2 (socket binding)
- **Decode Tokens**: ${DECODE_TOKENS} per test
- **Prefill Sequence Lengths**: ${PREFILL_LENGTHS[@]}
- **Backends Tested**:
  1. **MKL BF16**: `LLAMINAR_QUANT_BF16_PREFER_MKL=1 LLAMINAR_QUANT_BF16_GEMM=1`
  2. **FP32 Fallback**: `LLAMINAR_QUANT_BF16_GEMM=0` (forces BF16→FP32 expansion)

## Results Summary

EOF

# Replace placeholders with actual values
CURRENT_DATE=$(date +"%Y-%m-%d %H:%M:%S")
MODEL_NAME=$(basename "$MODEL")
CPU_MODEL=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
CPU_FLAGS=$(lscpu | grep "Flags" | grep -o "avx[^ ]*" | tr '\n' ' ')

sed -i "s|\\\$(date.*)|${CURRENT_DATE}|g" "${RESULT_FILE}"
sed -i "s|\\\$(basename.*)|${MODEL_NAME}|g" "${RESULT_FILE}"
sed -i "s|\\\$(lscpu.*Model name.*)|${CPU_MODEL}|g" "${RESULT_FILE}"
sed -i "s|\\\$(lscpu.*Flags.*)|${CPU_FLAGS}|g" "${RESULT_FILE}"

# Function to run benchmark
run_benchmark() {
    local backend_name="$1"
    local env_vars="$2"
    local prompt_length="$3"
    local prompt_text="$4"
    
    echo -e "${BLUE}Testing ${backend_name} (prompt length: ${prompt_length})...${NC}"
    
    # Create temporary output file
    local temp_output=$(mktemp)
    
    # Run benchmark with timeout (60 seconds max)
    timeout 60s bash -c "
        export ${env_vars}
        export OMP_NUM_THREADS=28
        export OMP_PLACES=sockets
        export OMP_PROC_BIND=close
        mpirun -np 2 --bind-to socket --map-by socket \
            --mca mpi_leave_pinned 1 \
            --mca btl_vader_single_copy_mechanism none \
            ${BUILD_DIR}/llaminar \
            --benchmark \
            -m ${MODEL} \
            -p '${prompt_text}' \
            -n ${DECODE_TOKENS} 2>&1
    " > "${temp_output}" || {
        echo -e "${RED}✗ Benchmark failed or timed out${NC}"
        rm -f "${temp_output}"
        return 1
    }
    
    # Extract metrics (prefill and decode throughput)
    local prefill_tokens=$(grep "PREFILL PHASE" -A 3 "${temp_output}" | grep "Tokens:" | awk '{print $2}')
    local prefill_time=$(grep "PREFILL PHASE" -A 3 "${temp_output}" | grep "Time:" | awk '{print $2}')
    local prefill_throughput=$(grep "PREFILL PHASE" -A 3 "${temp_output}" | grep "Throughput:" | awk '{print $2}')
    
    local decode_tokens=$(grep "DECODE PHASE" -A 3 "${temp_output}" | grep "Tokens:" | awk '{print $2}')
    local decode_time=$(grep "DECODE PHASE" -A 3 "${temp_output}" | grep "Time:" | awk '{print $2}')
    local decode_throughput=$(grep "DECODE PHASE" -A 3 "${temp_output}" | grep "Throughput:" | awk '{print $2}')
    
    # Print extracted results
    if [ -n "${prefill_throughput}" ] && [ -n "${decode_throughput}" ]; then
        echo -e "${GREEN}✓ Prefill: ${prefill_throughput} tok/s (${prefill_tokens} tokens, ${prefill_time} ms)${NC}"
        echo -e "${GREEN}✓ Decode:  ${decode_throughput} tok/s (${decode_tokens} tokens, ${decode_time} ms)${NC}"
    else
        echo -e "${RED}✗ Failed to extract metrics${NC}"
        rm -f "${temp_output}"
        return 1
    fi
    
    # Store results for aggregation
    echo "${backend_name}|${prompt_length}|${prefill_tokens}|${prefill_time}|${prefill_throughput}|${decode_tokens}|${decode_time}|${decode_throughput}" >> "${RESULTS_DIR}/raw_data_${TIMESTAMP}.txt"
    
    rm -f "${temp_output}"
    return 0
}

# Function to generate prompt of specific length
generate_prompt() {
    local target_length="$1"
    
    if [ "$target_length" -eq 8 ]; then
        echo "Explain quantum computing."
    elif [ "$target_length" -eq 64 ]; then
        echo "Explain the concept of machine learning in detail, including its applications in modern technology and research."
    elif [ "$target_length" -eq 256 ]; then
        echo "Write a comprehensive essay about the history of artificial intelligence, starting from the Turing test in the 1950s, through the AI winters, to the modern deep learning revolution. Include key milestones like expert systems, neural networks, backpropagation, convolutional networks, and transformer architectures. Discuss the impact of large language models on natural language processing."
    elif [ "$target_length" -eq 512 ]; then
        echo "Provide an in-depth analysis of the evolution of computer architecture from the early days of vacuum tubes and punch cards through transistors, integrated circuits, microprocessors, and modern multi-core systems. Discuss the von Neumann architecture, RISC vs CISC debate, pipelining, superscalar execution, out-of-order execution, branch prediction, cache hierarchies, virtual memory, and modern trends like heterogeneous computing, GPUs for general-purpose computing, and specialized accelerators for machine learning workloads. Include examples of specific processor families like Intel x86, ARM, and RISC-V. Explain how Moore's Law has shaped the industry and discuss the challenges of continued scaling as we approach physical limits. Cover emerging technologies like quantum computing and neuromorphic computing that may define the future of computation."
    else
        echo "This is a test prompt."
    fi
}

# Initialize raw data file
echo "Backend|PromptLength|PrefillTokens|PrefillTime|PrefillThroughput|DecodeTokens|DecodeTime|DecodeThroughput" > "${RESULTS_DIR}/raw_data_${TIMESTAMP}.txt"

# Run benchmarks for each sequence length
for length in "${PREFILL_LENGTHS[@]}"; do
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Testing with prompt length: ~${length} tokens${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    prompt=$(generate_prompt "$length")
    
    # Test 1: MKL BF16 Backend
    run_benchmark "MKL" "LLAMINAR_QUANT_BF16_PREFER_MKL=1 LLAMINAR_QUANT_BF16_GEMM=1" "$length" "$prompt"
    
    echo ""
    
    # Test 2: FP32 Fallback (disable BF16 GEMM)
    run_benchmark "FP32_Fallback" "LLAMINAR_QUANT_BF16_GEMM=0" "$length" "$prompt"
    
    echo ""
done

# Aggregate results and create comparison table
echo ""
echo -e "${BLUE}Generating results report...${NC}"

cat >> "${RESULT_FILE}" << 'EOF'

### Prefill Performance

| Prompt Length | Backend | Tokens | Time (ms) | Throughput (tok/s) | Speedup |
|--------------|---------|--------|-----------|-------------------|---------|
EOF

# Process prefill results
while IFS='|' read -r backend length prefill_tok prefill_time prefill_tput decode_tok decode_time decode_tput; do
    if [ "$backend" != "Backend" ]; then
        echo "| ${length} | ${backend} | ${prefill_tok} | ${prefill_time} | ${prefill_tput} | - |" >> "${RESULT_FILE}"
    fi
done < "${RESULTS_DIR}/raw_data_${TIMESTAMP}.txt"

cat >> "${RESULT_FILE}" << 'EOF'

### Decode Performance

| Prompt Length | Backend | Tokens | Time (ms) | Throughput (tok/s) | Speedup |
|--------------|---------|--------|-----------|-------------------|---------|
EOF

# Process decode results
while IFS='|' read -r backend length prefill_tok prefill_time prefill_tput decode_tok decode_time decode_tput; do
    if [ "$backend" != "Backend" ]; then
        echo "| ${length} | ${backend} | ${decode_tok} | ${decode_time} | ${decode_tput} | - |" >> "${RESULT_FILE}"
    fi
done < "${RESULTS_DIR}/raw_data_${TIMESTAMP}.txt"

cat >> "${RESULT_FILE}" << 'EOF'

## Analysis

### Key Findings

1. **Prefill Phase**:
   - MKL BF16 performance vs FP32 fallback across sequence lengths
   - Expected: MKL ~5-10% faster due to avoiding BF16→FP32 expansion

2. **Decode Phase**:
   - Single-token decode performance comparison
   - Expected: Similar performance (decode is typically memory-bound)

3. **Recommendations**:
   - Use MKL BF16 backend for CPUs without AVX512_BF16
   - Provides robustness against OpenBLAS NaN bugs
   - Acceptable performance overhead (~5-10%)

### Raw Data

See `raw_data_${TIMESTAMP}.txt` for complete measurements.

---

**Generated**: ${CURRENT_DATE}
EOF

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  Benchmark Complete!                           ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Results saved to: ${RESULT_FILE}"
echo "Raw data: ${RESULTS_DIR}/raw_data_${TIMESTAMP}.txt"
echo ""
echo "View results:"
echo "  cat ${RESULT_FILE}"
echo ""
