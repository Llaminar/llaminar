#!/bin/bash
# Run integer GEMM minimal profiling harness
# Usage: ./run_integer_gemm_profiler.sh [test|profile|sweep|help]

set -euo pipefail

HARNESS="./build_v2_release/performance/v2_perf_integer_gemm_minimal"

if [ ! -f "$HARNESS" ]; then
    echo "ERROR: Harness not built. Building now..."
    cmake --build build_v2_release --target v2_perf_integer_gemm_minimal --parallel
fi

case "${1:-test}" in
    test)
        echo "=== Quick Test (MR=1 vs MR=16) ==="
        echo ""
        echo "--- MR=1 (baseline, should be ~52 GFLOPS) ---"
        $HARNESS --mr 1
        echo ""
        echo "--- MR=16 (buggy, should be ~6 GFLOPS) ---"
        $HARNESS --mr 16
        ;;
    
    sweep)
        echo "=== Configuration Sweep ==="
        echo ""
        echo "MR,GFLOPS,Efficiency(%)"
        for mr in 1 2 4 8 16; do
            result=$($HARNESS --mr $mr 2>/dev/null | grep "Throughput:" | awk '{print $2}')
            efficiency=$($HARNESS --mr $mr 2>/dev/null | grep "Efficiency:" | awk '{print $2}')
            echo "$mr,$result,$efficiency"
        done
        ;;
    
    profile)
        echo "=== Profiling with perf ==="
        echo ""
        echo "Running 100 iterations with MR=16..."
        perf record -g --call-graph dwarf $HARNESS --mr 16 --iters 100
        echo ""
        echo "Generating report..."
        perf report --stdio | head -100
        ;;
    
    large)
        echo "=== Large Matrix Test (512×4096×4096) ==="
        echo ""
        $HARNESS --m 512 --n 4096 --k 4096 --iters 10
        ;;
    
    help|*)
        echo "Usage: $0 [MODE]"
        echo ""
        echo "Modes:"
        echo "  test     - Quick test (MR=1 vs MR=16, default)"
        echo "  sweep    - Configuration sweep (MR=1,2,4,8,16)"
        echo "  profile  - Profile with perf (MR=16, 100 iters)"
        echo "  large    - Large matrix test (512×4096×4096)"
        echo "  help     - Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 test               # Quick test"
        echo "  $0 sweep > results.csv  # Save sweep results"
        echo "  $0 profile            # Profile hotspots"
        echo ""
        echo "Custom runs:"
        echo "  $HARNESS --mr 8 --unroll 8 --prefetch 3"
        echo "  $HARNESS --m 256 --n 2048 --k 2048 --iters 100"
        ;;
esac
