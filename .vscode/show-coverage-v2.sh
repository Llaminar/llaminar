#!/bin/bash
# Show gcov coverage metrics for Llaminar V2 project

cd "$(dirname "$0")/.."

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         V2 Test Coverage Summary (gcov)                    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

GCDA_COUNT=$(find build_v2 -name '*.gcda' 2>/dev/null | wc -l)

if [ "$GCDA_COUNT" -eq 0 ]; then
    echo "❌ No coverage data found."
    echo ""
    echo "Make sure you:"
    echo "  1. Built with coverage enabled (Debug build with --coverage)"
    echo "  2. Ran tests to generate .gcda files"
    echo ""
    echo "Example:"
    echo "  ctest --test-dir build_v2 -R '^V2_Unit_'"
    exit 1
fi

# Find all .gcno files for llaminar2_core
GCNO_COUNT=$(find build_v2/src -name "*.gcno" 2>/dev/null | wc -l)

if [ "$GCNO_COUNT" -eq 0 ]; then
    echo "❌ No .gcno files found. Rebuild with coverage enabled."
    echo ""
    echo "Rebuild with:"
    echo "  cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug \\"
    echo "    -DCMAKE_CXX_FLAGS_DEBUG='-g -O0 --coverage -fprofile-abs-path' \\"
    echo "    -DCMAKE_EXE_LINKER_FLAGS_DEBUG='--coverage'"
    echo "  cmake --build build_v2 --parallel"
    exit 1
fi

echo "Found $GCDA_COUNT coverage data files from $GCNO_COUNT instrumented sources"
echo ""

# Change to build directory
cd build_v2

# Key V2 source files to report on
KEY_FILES=(
    "src/v2/pipelines/qwen/Qwen2Pipeline.cpp"
    "src/v2/pipelines/PipelineBase.cpp"
    "src/v2/loaders/ModelLoader.cpp"
    "src/v2/tensors/IQ4_NLTensor.cpp"
    "src/v2/tensors/FP32Tensor.cpp"
    "src/v2/tensors/BF16Tensor.cpp"
    "src/v2/tensors/TensorFactory.cpp"
    "src/v2/kernels/cpu/QuantizedGemm.cpp"
    "src/v2/backends/DeviceOrchestrator.cpp"
    "src/v2/utils/ArgParser.cpp"
)

echo "Coverage Report (Key Files):"
echo "═══════════════════════════════════════════════════════════"

TOTAL_LINES=0
COVERED_LINES=0

for srcfile in "${KEY_FILES[@]}"; do
    # Find corresponding .gcno file
    GCNO_FILE=$(find . -name "$(basename ${srcfile}).gcno" | head -1)
    
    if [ -f "$GCNO_FILE" ]; then
        # Run gcov and capture output
        GCOV_OUTPUT=$(gcov -n "$GCNO_FILE" 2>&1)
        
        # Extract coverage percentage
        COVERAGE=$(echo "$GCOV_OUTPUT" | grep "Lines executed:" | head -1 | sed 's/Lines executed://' | xargs)
        
        if [ -n "$COVERAGE" ]; then
            # Parse percentage
            PERCENT=$(echo "$COVERAGE" | grep -oE '[0-9]+\.[0-9]+%' | head -1)
            LINES_INFO=$(echo "$COVERAGE" | grep -oE '[0-9]+ of [0-9]+' | head -1)
            
            if [ -n "$PERCENT" ]; then
                # Extract numeric values for totals
                COVERED=$(echo "$LINES_INFO" | cut -d' ' -f1)
                TOTAL=$(echo "$LINES_INFO" | cut -d' ' -f3)
                
                TOTAL_LINES=$((TOTAL_LINES + TOTAL))
                COVERED_LINES=$((COVERED_LINES + COVERED))
                
                # Color code based on percentage
                PERCENT_NUM=$(echo "$PERCENT" | sed 's/%//')
                if (( $(echo "$PERCENT_NUM >= 80" | bc -l) )); then
                    COLOR="\033[0;32m" # Green
                elif (( $(echo "$PERCENT_NUM >= 50" | bc -l) )); then
                    COLOR="\033[0;33m" # Yellow
                else
                    COLOR="\033[0;31m" # Red
                fi
                NC="\033[0m" # No color
                
                printf "${COLOR}%-50s %6s  (%s)${NC}\n" "$(basename $srcfile)" "$PERCENT" "$LINES_INFO"
            fi
        fi
    fi
done

echo "═══════════════════════════════════════════════════════════"

if [ "$TOTAL_LINES" -gt 0 ]; then
    OVERALL_PERCENT=$(echo "scale=2; $COVERED_LINES * 100 / $TOTAL_LINES" | bc)
    echo ""
    echo "Overall Coverage (Key Files): ${OVERALL_PERCENT}% ($COVERED_LINES of $TOTAL_LINES lines)"
fi

cd ..

echo ""
echo "✓ Coverage data available for $GCDA_COUNT test runs"
echo ""
echo "For detailed per-line coverage:"
echo "  1. Open a source file in VS Code"
echo "  2. Ctrl+Shift+P → 'Coverage Gutters: Display Coverage'"
echo ""
