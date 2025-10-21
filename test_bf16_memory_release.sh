#!/bin/bash
# test_bf16_memory_release.sh
# Measure memory savings from BF16 activation storage

set -e
set -x

MODEL="models/qwen2.5-0.5b-instruct-q8_0.gguf"
PROMPT="Write a detailed explanation of how neural networks work, including the mathematics behind backpropagation and gradient descent."
N_TOKENS=10

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         BF16 Memory Savings Test (Release Build)             ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

echo ""
echo "Configuration:"
echo "  Model: qwen2.5-0.5b-instruct-q8_0"
echo "  Prompt: Long technical prompt (~150 tokens)"
echo "  Generate: $N_TOKENS tokens"
echo ""

# Function to extract peak RSS
extract_peak_rss() {
    grep -oP 'peak_rss_mb=\K[0-9]+' | tail -1
}

echo "───────────────────────────────────────────────────────────────"
echo "Test 1: FP32 Baseline (Standard Inference)"
echo "───────────────────────────────────────────────────────────────"

FP32_OUTPUT=$(./run_llaminar.sh -m "$MODEL" -p "$PROMPT" -n $N_TOKENS 2>&1)
FP32_RSS=$(echo "$FP32_OUTPUT" | extract_peak_rss)

echo "$FP32_OUTPUT" | grep -E "(peak_rss|Memory)" || echo "No memory stats found"
echo ""
echo "FP32 Peak RSS: ${FP32_RSS:-unknown} MB"

echo ""
echo "───────────────────────────────────────────────────────────────"
echo "Test 2: BF16 Activations (Pull-Through Cache)"
echo "───────────────────────────────────────────────────────────────"

BF16_OUTPUT=$(LLAMINAR_QUANT_OUTPUT_BF16=1 ./run_llaminar.sh -- -m "$MODEL" -p "$PROMPT" -n $N_TOKENS 2>&1)
BF16_RSS=$(echo "$BF16_OUTPUT" | extract_peak_rss)

echo "$BF16_OUTPUT" | grep -E "(peak_rss|Memory)" || echo "No memory stats found"
echo ""
echo "BF16 Peak RSS: ${BF16_RSS:-unknown} MB"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                         RESULTS                                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

if [ -n "$FP32_RSS" ] && [ -n "$BF16_RSS" ]; then
    SAVINGS=$((FP32_RSS - BF16_RSS))
    PERCENT=$(awk "BEGIN {printf \"%.1f\", ($SAVINGS / $FP32_RSS) * 100}")
    
    echo ""
    echo "  FP32 Baseline:     $FP32_RSS MB"
    echo "  BF16 Activations:  $BF16_RSS MB"
    echo "  ────────────────────────────"
    echo "  Savings:           $SAVINGS MB  ($PERCENT%)"
    echo ""
    
    EXPECTED=1295  # Expected activation savings
    ACTUAL=$SAVINGS
    
    if [ $ACTUAL -ge $((EXPECTED / 2)) ]; then
        echo "  Status: ✅ SUCCESS - Significant memory savings observed!"
    else
        echo "  Status: ⚠️  WARNING - Savings lower than expected"
    fi
else
    echo ""
    echo "  ⚠️  Could not extract memory statistics"
    echo "  FP32 RSS: ${FP32_RSS:-not found}"
    echo "  BF16 RSS: ${BF16_RSS:-not found}"
    echo ""
    echo "  Note: Memory tracking may need to be enabled in the code"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
