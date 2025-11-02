#!/bin/bash
# Monitor validation test progress

echo "=== Neural Network Improvements - Validation Test Monitor ==="
echo "Started: $(date)"
echo ""

# Check if test is running
if pgrep -f "v2_perf_cuda_heuristic_validation" > /dev/null; then
    echo "✓ Validation test is RUNNING"
else
    echo "✗ Validation test is NOT running"
    exit 1
fi

echo ""
echo "=== Data Collection Progress ==="

# Check CSV file
if [ -f build_v2_release/cuda_gemm_benchmark_data.csv ]; then
    LINES=$(wc -l < build_v2_release/cuda_gemm_benchmark_data.csv)
    echo "CSV rows collected: $LINES"
    
    # Check for 1.5B data
    if grep -q "1_5B" build_v2_release/cuda_gemm_benchmark_data.csv 2>/dev/null; then
        LINES_1_5B=$(grep "1_5B" build_v2_release/cuda_gemm_benchmark_data.csv | wc -l)
        echo "1.5B model data: $LINES_1_5B rows ✓"
    else
        echo "1.5B model data: Not yet collected (test still running)"
    fi
else
    echo "CSV file not yet created"
fi

echo ""
echo "=== Recent Test Activity ==="
tail -20 /tmp/validation_run.log 2>/dev/null | grep -E "(Test|Model|METRIC|correlation)" || echo "No log output yet"

echo ""
echo "=== Estimated Completion ==="
echo "Total duration: 45-75 minutes"
echo "Started: $(date)"
echo ""
echo "Run 'tail -f /tmp/validation_run.log' to watch live progress"
