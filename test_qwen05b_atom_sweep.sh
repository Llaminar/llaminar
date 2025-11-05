#!/bin/bash
#
# Test atom parameter expansion on Qwen 0.5B shapes only
#
# Purpose: Validate that adding atom_type and atom_layout_m/n/k to the config space
#          improves performance before running the expensive full benchmark suite.
#
# Qwen 0.5B Workload Shapes:
#   - d_model = 896
#   - ffn_intermediate = 4864
#
# Key shapes:
#   1×896×896   - Single token Q/K/V projection
#   1×4864×896  - Single token FFN gate/up projection
#   1×896×4864  - Single token FFN down projection
#   128×896×896 - Batch Q/K/V projection (batch_size=128)
#
# Expected runtime: 4-8 hours (vs 20-30 hours for full sweep)
#

set -euo pipefail

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Qwen 0.5B Atom Parameter Sweep"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Step 1: Regenerate configs with atom parameters
echo "1. Generating kernel variants with atom parameters..."
cd /workspaces/llaminar
python3 src/v2/kernels/cuda/generate_cuda_gemm_variants.py

# Count generated configs
NUM_CONFIGS=$(find src/v2/kernels/cuda/generated -name "*.cu" -type f | wc -l)
echo "   Generated $NUM_CONFIGS variant files"

# Step 2: Rebuild with new config space
echo ""
echo "2. Rebuilding with expanded config space..."
cmake --build build_v2_release --target profile_cuda_config --parallel

# Step 3: Create benchmark shapes file for Qwen 0.5B
echo ""
echo "3. Creating Qwen 0.5B benchmark shapes..."
cat > /tmp/qwen05b_shapes.txt << 'EOF'
# Qwen 0.5B Benchmark Shapes (4 total)
# Format: batch,m,n,k,label
1,1,896,896,single_token_qkv
1,1,4864,896,single_token_ffn_gate
1,1,896,4864,single_token_ffn_down
1,128,896,896,batch128_qkv
EOF

# Step 4: Run benchmark on limited shapes
echo ""
echo "4. Running benchmark on Qwen 0.5B shapes only..."
echo "   NOTE: This will take 4-8 hours (vs 20-30 hours for full suite)"
echo ""

# Create output directory
mkdir -p cuda_atom_sweep_results
OUTPUT_CSV="cuda_atom_sweep_results/qwen05b_atom_sweep_$(date +%Y%m%d_%H%M%S).csv"

# Write CSV header
echo "batch,m,n,k,tile_m,tile_n,tile_k,threads_m,threads_n,work_m,work_n,prefetch,transpose,vectorize,atom_type,atom_m,atom_n,atom_k,gflops,ms,label" > "$OUTPUT_CSV"

# Benchmark each shape
while IFS=',' read -r batch m n k label || [ -n "$batch" ]; do
    # Skip comments and empty lines
    [[ "$batch" =~ ^#.*$ ]] && continue
    [[ -z "$batch" ]] && continue
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Benchmarking: $label (batch=$batch, M=$m, N=$n, K=$k)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Run profiler for this shape across all configs
    # Output format: tile_m,tile_n,tile_k,threads_m,threads_n,work_m,work_n,prefetch,transpose,vectorize,atom_type,atom_m,atom_n,atom_k,gflops,ms
    ./build_v2_release/profile_cuda_config "$batch" "$m" "$n" "$k" 2>&1 | \
        grep -E "^[0-9]+" | \
        awk -v b="$batch" -v m="$m" -v n="$n" -v k="$k" -v l="$label" \
            '{print b","m","n","k","$0","l}' >> "$OUTPUT_CSV"
    
    # Show top 5 configs for this shape
    echo ""
    echo "Top 5 configs for $label:"
    tail -100 "$OUTPUT_CSV" | sort -t',' -k19 -rn | head -5 | \
        awk -F',' '{printf "  %6.2f GFLOPS - tile=%dx%dx%d threads=%dx%d work=%dx%d atom=%d(%dx%dx%d)\n", 
            $19, $5, $6, $7, $8, $9, $10, $11, $15, $16, $17, $18}'
    
done < /tmp/qwen05b_shapes.txt

# Step 5: Analysis
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. Analyzing results..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Overall best configs
echo "Overall Top 10 Configs:"
tail -n +2 "$OUTPUT_CSV" | sort -t',' -k19 -rn | head -10 | \
    awk -F',' '{printf "  %6.2f GFLOPS - %-25s tile=%dx%dx%d threads=%dx%d atom=%d(%dx%dx%d)\n", 
        $19, $21, $5, $6, $7, $8, $9, $15, $16, $17, $18}'

# Atom type distribution in top configs
echo ""
echo "Atom Type Distribution (top 100 configs):"
tail -n +2 "$OUTPUT_CSV" | sort -t',' -k19 -rn | head -100 | \
    awk -F',' '{print $15}' | sort | uniq -c | \
    awk '{printf "  Atom type %s: %3d configs (%.1f%%)\n", $2, $1, $1/100*100}'

# Atom layout distribution in top configs
echo ""
echo "Atom Layout Distribution (top 100 configs):"
tail -n +2 "$OUTPUT_CSV" | sort -t',' -k19 -rn | head -100 | \
    awk -F',' '{printf "%dx%d\n", $16, $17}' | sort | uniq -c | \
    awk '{printf "  Layout %s: %3d configs (%.1f%%)\n", $2, $1, $1/100*100}'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Results saved to: $OUTPUT_CSV"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  1. Review results to see if atom diversity helps performance"
echo "  2. If >5% improvement on any shape: Run full benchmark suite"
echo "  3. If <2% improvement: Consider reverting atom params"
echo ""
