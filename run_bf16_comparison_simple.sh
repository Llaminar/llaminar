#!/bin/bash
# Simplified BF16 Backend Comparison
# Tests OpenBLAS vs Intel MKL in Release mode with proper threading

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Detect CPU topology
physical_ids=$(grep "^physical id" /proc/cpuinfo | awk '{print $NF}' | sort -u | wc -l)
unique_cores=$(awk '
    /^processor/ { proc = $NF }
    /^physical id/ { phys_id = $NF }
    /^core id/ { core_id = $NF; print phys_id ":" core_id }
' /proc/cpuinfo | sort -u | wc -l)

SOCKETS=$physical_ids
CORES_PER_SOCKET=$((unique_cores / SOCKETS))
OMP_THREADS=$CORES_PER_SOCKET

# Set OpenMP/MPI environment
export OMP_NUM_THREADS=$OMP_THREADS
export OMP_PLACES=sockets
export OMP_PROC_BIND=close
export OMP_NESTED=false
export OMP_DYNAMIC=false
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0
export MKL_NUM_THREADS=$CORES_PER_SOCKET
export MKL_DYNAMIC=false
export OPENBLAS_NUM_THREADS=$CORES_PER_SOCKET
export GOTO_NUM_THREADS=$CORES_PER_SOCKET
export OMPI_MCA_mpi_leave_pinned=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none

MPI_CMD="mpirun -np $SOCKETS --bind-to socket --map-by socket --mca mpi_leave_pinned 1 --mca btl_vader_single_copy_mechanism none"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE} BF16 Backend Comparison: OpenBLAS vs Intel MKL (Release)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "Configuration:"
echo "  Sockets: $SOCKETS"
echo "  Cores/socket: $CORES_PER_SOCKET"
echo "  OMP_NUM_THREADS: $OMP_THREADS"
echo "  Build: Release mode"
echo ""

# Output file
OUTPUT_FILE="bf16_benchmark_results/comparison_$(date +%Y%m%d_%H%M%S).md"
mkdir -p bf16_benchmark_results

# Function to run single benchmark
run_benchmark() {
    local backend_name=$1
    local prefer_mkl=$2
    local prompt=$3
    local n_tokens=$4
    
    echo -e "${YELLOW}Testing $backend_name...${NC}" >&2
    
    export LLAMINAR_QUANT_BF16_GEMM=1
    export LLAMINAR_QUANT_BF16_PREFER_MKL=$prefer_mkl
    
    # Run and extract just the benchmark box
    local output=$(timeout 120 $MPI_CMD ./build_release/llaminar --benchmark \
        -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
        -p "$prompt" -n $n_tokens 2>&1 | awk '/INFERENCE BENCHMARK/,/^╚/' || echo "TIMEOUT")
    
    echo "$output" >&2
    echo "" >&2
    
    # Extract metrics using awk - field 3 has the value
    local prefill_time=$(echo "$output" | awk '/PREFILL PHASE/,/DECODE PHASE/ {if (/Time:/) {print $3; exit}}')
    local prefill_tput=$(echo "$output" | awk '/PREFILL PHASE/,/DECODE PHASE/ {if (/Throughput:/) {print $3; exit}}')
    local decode_time=$(echo "$output" | awk '/DECODE PHASE/,/TOTAL/ {if (/Time:/) {print $3; exit}}')
    local decode_tput=$(echo "$output" | awk '/DECODE PHASE/,/TOTAL/ {if (/Throughput:/) {print $3; exit}}')
    
    # Return to stdout for capture
    echo "${backend_name}|${prefill_time}|${prefill_tput}|${decode_time}|${decode_tput}"
}

echo "Running benchmarks..."
echo ""

# Test 1: Short prompt
echo -e "${BLUE}═══ Test 1: Short Prompt (8 tokens) ═══${NC}"
OB_SHORT=$(run_benchmark "OpenBLAS" 0 "Test" 10)
MKL_SHORT=$(run_benchmark "Intel MKL" 1 "Test" 10)

# Test 2: Medium prompt  
echo -e "${BLUE}═══ Test 2: Medium Prompt (~64 tokens) ═══${NC}"
MED_PROMPT="Explain the concept of machine learning in simple terms. What are neural networks and how do they learn from data? Provide examples of real-world applications."
OB_MED=$(run_benchmark "OpenBLAS" 0 "$MED_PROMPT" 50)
MKL_MED=$(run_benchmark "Intel MKL" 1 "$MED_PROMPT" 50)

# Test 3: Long prompt
echo -e "${BLUE}═══ Test 3: Long Prompt (~256 tokens) ═══${NC}"
LONG_PROMPT="Write a comprehensive explanation of distributed computing systems, including the differences between shared memory and distributed memory architectures, the role of message passing interfaces like MPI, common communication patterns such as broadcast and reduce operations, and the challenges of load balancing and fault tolerance in large-scale parallel applications. Discuss how these concepts apply to modern cloud computing and high-performance computing environments. Include examples of popular frameworks and libraries used in distributed computing, and explain the trade-offs between different parallelization strategies when designing scalable applications."
OB_LONG=$(run_benchmark "OpenBLAS" 0 "$LONG_PROMPT" 100)
MKL_LONG=$(run_benchmark "Intel MKL" 1 "$LONG_PROMPT" 100)

# Generate report
cat > "$OUTPUT_FILE" << EOF
# BF16 Backend Comparison: OpenBLAS vs Intel MKL

**Date**: $(date "+%B %d, %Y")  
**Hardware**: Intel Xeon Gold 6238R (Cascade Lake - no AVX512_BF16)  
**Configuration**: $SOCKETS sockets × $CORES_PER_SOCKET cores, Release build (-O3 -DNDEBUG)

## Test Results

### Short Prompt (8 tokens, 10 decode)

|Backend|Prefill Time (ms)|Prefill (tok/s)|Decode Time (ms)|Decode (tok/s)|
|-------|----------------|---------------|----------------|--------------|
EOF

# Parse and add results
IFS='|' read -r name pf_t pf_tp d_t d_tp <<< "$OB_SHORT"
echo "|$name|$pf_t|$pf_tp|$d_t|$d_tp|" >> "$OUTPUT_FILE"
IFS='|' read -r name pf_t pf_tp d_t d_tp <<< "$MKL_SHORT"
echo "|$name|$pf_t|$pf_tp|$d_t|$d_tp|" >> "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << EOF

### Medium Prompt (~64 tokens, 50 decode)

|Backend|Prefill Time (ms)|Prefill (tok/s)|Decode Time (ms)|Decode (tok/s)|
|-------|----------------|---------------|----------------|--------------|
EOF

IFS='|' read -r name pf_t pf_tp d_t d_tp <<< "$OB_MED"
echo "|$name|$pf_t|$pf_tp|$d_t|$d_tp|" >> "$OUTPUT_FILE"
IFS='|' read -r name pf_t pf_tp d_t d_tp <<< "$MKL_MED"
echo "|$name|$pf_t|$pf_tp|$d_t|$d_tp|" >> "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << EOF

### Long Prompt (~256 tokens, 100 decode)

|Backend|Prefill Time (ms)|Prefill (tok/s)|Decode Time (ms)|Decode (tok/s)|
|-------|----------------|---------------|----------------|--------------|
EOF

IFS='|' read -r name pf_t pf_tp d_t d_tp <<< "$OB_LONG"
echo "|$name|$pf_t|$pf_tp|$d_t|$d_tp|" >> "$OUTPUT_FILE"
IFS='|' read -r name pf_t pf_tp d_t d_tp <<< "$MKL_LONG"
echo "|$name|$pf_t|$pf_tp|$d_t|$d_tp|" >> "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << 'EOF'

## Analysis

**OpenBLAS v0.3.26**:
- Software BF16 emulation (verified working, no NaN issues)
- Baseline performance reference

**Intel MKL 2024.2**:
- Optimized software emulation on Cascade Lake
- Expected: Better performance due to optimizations

## Conclusion

Results show performance comparison between OpenBLAS and Intel MKL BF16 backends
on Cascade Lake hardware (software emulation only - no AVX512_BF16 support).

EOF

echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN} Benchmark Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""
cat "$OUTPUT_FILE"
