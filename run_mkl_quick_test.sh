#!/bin/bash
###############################################################################
# Simple MKL BF16 Performance Test
# Quick comparison of MKL vs FP32 fallback on decode operations
###############################################################################

set -euo pipefail

MODEL="${1:-models/qwen2.5-0.5b-instruct-q8_0.gguf}"
BUILD_DIR="${2:-build_release}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   MKL BF16 Quick Performance Test${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "Model: ${MODEL}"
echo ""

# Test prompts
PROMPTS=(
    "Explain AI."  # Short prompt ~3-4 tokens
    "Write a detailed explanation of quantum computing and its applications in cryptography."  # Medium ~20 tokens
)

DECODE_TOKENS=20

for i in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$i]}"
    
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Test $((i+1)): Prompt ${i}${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Test 1: MKL BF16
    echo -e "${BLUE}[1/2] Testing MKL BF16 backend...${NC}"
    LLAMINAR_QUANT_BF16_PREFER_MKL=1 LLAMINAR_QUANT_BF16_GEMM=1 \
        OMP_NUM_THREADS=28 OMP_PLACES=sockets OMP_PROC_BIND=close \
        mpirun -np 2 --bind-to socket --map-by socket \
        --mca mpi_leave_pinned 1 --mca btl_vader_single_copy_mechanism none \
        ${BUILD_DIR}/llaminar --benchmark -m "${MODEL}" -p "${prompt}" -n ${DECODE_TOKENS} 2>&1 | \
        grep -E "(Throughput:|Time:)" | tail -4
    
    echo ""
    
    # Test 2: FP32 Fallback
    echo -e "${BLUE}[2/2] Testing FP32 fallback...${NC}"
    LLAMINAR_QUANT_BF16_GEMM=0 \
        OMP_NUM_THREADS=28 OMP_PLACES=sockets OMP_PROC_BIND=close \
        mpirun -np 2 --bind-to socket --map-by socket \
        --mca mpi_leave_pinned 1 --mca btl_vader_single_copy_mechanism none \
        ${BUILD_DIR}/llaminar --benchmark -m "${MODEL}" -p "${prompt}" -n ${DECODE_TOKENS} 2>&1 | \
        grep -E "(Throughput:|Time:)" | tail -4
    
    echo ""
done

echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   Test Complete${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
