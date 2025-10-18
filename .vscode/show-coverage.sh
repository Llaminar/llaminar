#!/bin/bash
# Show gcov coverage metrics for Llaminar project

cd "$(dirname "$0")/.."

echo "=== Gcov Coverage Summary ==="
echo ""
echo "Checking for coverage data..."

GCDA_COUNT=$(find build -name '*.gcda' 2>/dev/null | wc -l)
echo "Found $GCDA_COUNT .gcda files"
echo ""

if [ "$GCDA_COUNT" -eq 0 ]; then
    echo "❌ No coverage data found. Run tests first to generate .gcda files."
    echo ""
    echo "Examples:"
    echo "  ctest --test-dir build -R BatchCorrectnessTest"
    echo "  mpirun -np 2 ./build/test_batch_correctness"
    echo "  ./run_llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p 'Test'"
    exit 1
fi

# Find all .gcno files for llaminar_core
GCNO_COUNT=$(find build/CMakeFiles/llaminar_core.dir/src -name "*.gcno" 2>/dev/null | wc -l)

if [ "$GCNO_COUNT" -eq 0 ]; then
    echo "❌ No .gcno files found. Make sure you built in Debug mode with coverage enabled."
    echo ""
    echo "Rebuild with:"
    echo "  cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug"
    echo "  cmake --build build --parallel"
    exit 1
fi

echo "Generating coverage report..."
echo "Found $GCNO_COUNT instrumented source files"
echo ""

# Change to build directory
cd build

# Run gcov and filter to show only our source files
echo "Coverage Summary (sample of core files):"
echo "=========================================="

# Process a few key files
for file in \
    CMakeFiles/llaminar_core.dir/src/BatchQwenPipeline.cpp.gcno \
    CMakeFiles/llaminar_core.dir/src/QwenPipeline.cpp.gcno \
    CMakeFiles/llaminar_core.dir/src/operators/MPIAttentionBatchOperator.cpp.gcno \
    CMakeFiles/llaminar_core.dir/src/operators/MPIAttentionOperator.cpp.gcno \
    CMakeFiles/llaminar_core.dir/src/operators/MPILinearBatchOperator.cpp.gcno \
    CMakeFiles/llaminar_core.dir/src/operators/MPILinearOperator.cpp.gcno \
    CMakeFiles/llaminar_core.dir/src/operators/MPISwiGLUBatchOperator.cpp.gcno \
    CMakeFiles/llaminar_core.dir/src/operators/MPIEmbeddingOperator.cpp.gcno \
    CMakeFiles/llaminar_core.dir/src/AbstractPipeline.cpp.gcno \
    CMakeFiles/llaminar_core.dir/src/PrefillProviderBaseImpl.cpp.gcno \
    CMakeFiles/llaminar_core.dir/src/BatchPaddingUtils.cpp.gcno \
    CMakeFiles/llaminar_core.dir/src/ModelLoader.cpp.gcno; do
    
    if [ -f "$file" ]; then
        # Get the source filename
        SRCFILE=$(echo "$file" | sed 's|CMakeFiles/llaminar_core.dir/||' | sed 's|\.gcno$||')
        
        # Run gcov and extract coverage percentage
        COVERAGE=$(gcov -n "$file" 2>&1 | grep -A1 "File.*${SRCFILE}'" | grep "Lines executed:" | head -1)
        
        if [ -n "$COVERAGE" ]; then
            printf "%-50s %s\n" "📄 $SRCFILE" "$COVERAGE"
        fi
    fi
done

cd ..

echo ""
echo "✓ Coverage data available for $GCDA_COUNT files"
echo ""
echo "For detailed coverage, open a source file in VS Code and run:"
echo "  Ctrl+Shift+P → 'Gcov Viewer: Show'"
