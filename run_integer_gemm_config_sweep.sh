#!/bin/bash
# Integer GEMM Configuration Sweep Script
# Tests 32-byte, 64-byte, and 128-byte processing with various tile sizes

set -euo pipefail

WORKSPACE="/workspaces/llaminar"
BUILD_DIR="${WORKSPACE}/build_v2"
BENCHMARK="${BUILD_DIR}/performance/v2_perf_integer_gemm_qwen_profile"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║ INTEGER GEMM CONFIGURATION SWEEP                              ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║ Testing K-block processing: 32, 64, 128 bytes                 ║"
echo "║ Testing prefetch distances: 0, 1, 2                           ║"
echo "║ Testing tile sizes: template default, 8, 16, 24, 32           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if benchmark exists
if [[ ! -f "$BENCHMARK" ]]; then
    echo -e "${RED}Error: Benchmark not found at $BENCHMARK${NC}"
    echo "Building benchmark..."
    cd "$WORKSPACE"
    cmake --build "$BUILD_DIR" --target v2_perf_integer_gemm_qwen_profile -j$(nproc)
fi

# Results directory
RESULTS_DIR="${WORKSPACE}/results/integer_gemm_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}Results will be saved to: $RESULTS_DIR${NC}"
echo ""

# CSV header
echo "K_BLOCKS,PREFETCH,TILE_M,AVG_GFLOPS,DECODE_GFLOPS,PREFILL32_GFLOPS,PREFILL128_GFLOPS,PREFILL512_GFLOPS" > "$RESULTS_DIR/sweep_results.csv"

# Test configurations
K_BLOCKS_VALUES=(1 2 4)
PREFETCH_VALUES=(0 1 2)
TILE_M_VALUES=(0 8 16 24 32)  # 0 = use template default

total_tests=$((${#K_BLOCKS_VALUES[@]} * ${#PREFETCH_VALUES[@]} * ${#TILE_M_VALUES[@]}))
current_test=0

for k_blocks in "${K_BLOCKS_VALUES[@]}"; do
    for prefetch in "${PREFETCH_VALUES[@]}"; do
        for tile_m in "${TILE_M_VALUES[@]}"; do
            ((current_test++))
            
            echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
            echo -e "${YELLOW}Test $current_test/$total_tests: K_BLOCKS=$k_blocks PREFETCH=$prefetch TILE_M=$tile_m${NC}"
            echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
            
            # Set environment variables
            export LLAMINAR_INT_GEMM_K_BLOCKS_PER_ITER=$k_blocks
            export LLAMINAR_INT_GEMM_PREFETCH_DIST=$prefetch
            export LLAMINAR_INT_GEMM_TILE_M=$tile_m
            export LLAMINAR_INT_GEMM_VERBOSE=0  # Suppress config output
            
            # Run benchmark
            OUTPUT_FILE="$RESULTS_DIR/test_${k_blocks}blocks_${prefetch}prefetch_${tile_m}tilem.log"
            
            if timeout 60 "$BENCHMARK" --simd > "$OUTPUT_FILE" 2>&1; then
                # Extract performance metrics
                avg_gflops=$(grep "AVERAGE" "$OUTPUT_FILE" | awk '{print $4}' || echo "0.00")
                
                # Extract specific operation performance
                decode_gflops=$(grep "Q_proj (decode)" "$OUTPUT_FILE" | head -1 | awk '{print $5}' || echo "0.00")
                prefill32_gflops=$(grep "Q_proj (prefill-32)" "$OUTPUT_FILE" | awk '{print $4}' || echo "0.00")
                prefill128_gflops=$(grep "Q_proj (prefill-128)" "$OUTPUT_FILE" | awk '{print $4}' || echo "0.00")
                prefill512_gflops=$(grep "Q_proj (prefill-512)" "$OUTPUT_FILE" | awk '{print $4}' || echo "0.00")
                
                echo "$k_blocks,$prefetch,$tile_m,$avg_gflops,$decode_gflops,$prefill32_gflops,$prefill128_gflops,$prefill512_gflops" >> "$RESULTS_DIR/sweep_results.csv"
                
                echo -e "${GREEN}✓ AVG: $avg_gflops GFLOPS${NC}"
                echo -e "  Decode: $decode_gflops | Prefill-32: $prefill32_gflops | Prefill-128: $prefill128_gflops | Prefill-512: $prefill512_gflops"
            else
                echo -e "${RED}✗ Test failed or timed out${NC}"
                echo "$k_blocks,$prefetch,$tile_m,FAILED,FAILED,FAILED,FAILED,FAILED" >> "$RESULTS_DIR/sweep_results.csv"
            fi
            
            echo ""
        done
    done
done

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║ SWEEP COMPLETE                                                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Find best configuration
echo -e "${BLUE}Top 5 configurations by average GFLOPS:${NC}"
sort -t',' -k4 -nr "$RESULTS_DIR/sweep_results.csv" | grep -v "K_BLOCKS" | head -5 | while IFS=',' read -r k_blocks prefetch tile_m avg decode p32 p128 p512; do
    echo "  K_BLOCKS=$k_blocks, PREFETCH=$prefetch, TILE_M=$tile_m → $avg GFLOPS"
done

echo ""
echo -e "${BLUE}Results saved to: $RESULTS_DIR${NC}"
echo "View full results: cat $RESULTS_DIR/sweep_results.csv"
echo ""

# Generate summary report
cat > "$RESULTS_DIR/SUMMARY.md" << EOF
# Integer GEMM Configuration Sweep Results

**Date**: $(date)
**Total Tests**: $total_tests
**Results Directory**: $RESULTS_DIR

## Best Configurations

### By Average GFLOPS
\`\`\`
$(sort -t',' -k4 -nr "$RESULTS_DIR/sweep_results.csv" | grep -v "K_BLOCKS" | head -5)
\`\`\`

### By Prefill-512 GFLOPS (Peak)
\`\`\`
$(sort -t',' -k8 -nr "$RESULTS_DIR/sweep_results.csv" | grep -v "K_BLOCKS" | head -5)
\`\`\`

## Key Findings

### K-Block Processing Impact
- 32-byte (1 block): Baseline
- 64-byte (2 blocks): Expected ~5-10% improvement
- 128-byte (4 blocks): Expected ~10-20% improvement (if successful)

### Prefetching Impact
- Distance 0: No prefetching (baseline)
- Distance 1: Minimal latency hiding
- Distance 2: Optimal for AVX512 (typically best)

### Tile Size Impact
- Smaller tiles (8): More outer loop overhead
- Medium tiles (16): Balanced
- Larger tiles (24-32): Reduced overhead but higher register pressure

## Raw Data

See \`sweep_results.csv\` for complete data matrix.

## Environment Variables for Best Config

\`\`\`bash
# Extract best configuration
BEST_LINE=\$(sort -t',' -k4 -nr $RESULTS_DIR/sweep_results.csv | grep -v "K_BLOCKS" | head -1)
BEST_K=\$(echo \$BEST_LINE | cut -d',' -f1)
BEST_PREFETCH=\$(echo \$BEST_LINE | cut -d',' -f2)
BEST_TILE_M=\$(echo \$BEST_LINE | cut -d',' -f3)

export LLAMINAR_INT_GEMM_K_BLOCKS_PER_ITER=\$BEST_K
export LLAMINAR_INT_GEMM_PREFETCH_DIST=\$BEST_PREFETCH
export LLAMINAR_INT_GEMM_TILE_M=\$BEST_TILE_M
\`\`\`
EOF

echo -e "${GREEN}Summary report generated: $RESULTS_DIR/SUMMARY.md${NC}"
