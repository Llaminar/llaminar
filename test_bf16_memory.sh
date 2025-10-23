#!/bin/bash
# Test BF16 Memory Usage - Phase 3 Validation
# Expected: ~2650 MB (vs 6498 MB before fix)

set -e

MODEL="models/qwen2.5-0.5b-instruct-q8_0.gguf"
PROMPT="Write a detailed technical explanation of machine learning, covering neural networks, training processes, optimization algorithms, and practical applications in modern AI systems."

echo "=========================================="
echo "BF16 Memory Test - Phase 3 Validation"
echo "=========================================="
echo ""
echo "Model: $MODEL"
echo "Prompt length: ~40 tokens (long to stress activations)"
echo "Generation: 100 tokens"
echo ""

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    echo "Please ensure the model is available."
    exit 1
fi

echo "=========================================="
echo "Test 1: BASELINE (FP32 activations)"
echo "=========================================="
echo "Expected: ~5171 MB (100% baseline)"
echo ""

export LLAMINAR_QUANT_OUTPUT_BF16=0
export OMP_NUM_THREADS=$(nproc --all)
export OMP_PLACES=sockets
export OMP_PROC_BIND=close

# Run with memory tracking
/usr/bin/time -v mpirun -np 2 --bind-to socket --map-by socket \
    ./build/llaminar -m "$MODEL" -p "$PROMPT" -n 100 2>&1 | \
    grep -E "Maximum resident set size|User time|System time" | head -3

echo ""
echo "=========================================="
echo "Test 2: BF16 ACTIVATIONS (Pull-Through Cache)"
echo "=========================================="
echo "Expected: ~2650 MB (50% + 64MB cache)"
echo "Target: <3000 MB (memory leak eliminated!)"
echo ""

export LLAMINAR_QUANT_OUTPUT_BF16=1
export LLAMINAR_TENSOR_CACHE_STATS=1

# Run with memory tracking
/usr/bin/time -v mpirun -np 2 --bind-to socket --map-by socket \
    ./build/llaminar -m "$MODEL" -p "$PROMPT" -n 100 2>&1 | \
    grep -E "Maximum resident set size|User time|System time|Cache|Hit|Miss" | head -10

echo ""
echo "=========================================="
echo "Memory Analysis Complete"
echo "=========================================="
echo ""
echo "If 'Maximum resident set size' for Test 2 is:"
echo "  - <3000 MB:  ✅ PASS - Memory leak eliminated!"
echo "  - 3000-4000: ⚠️  WARN - Some improvement but not optimal"
echo "  - >6000 MB:  ❌ FAIL - Memory leak still present"
echo ""
