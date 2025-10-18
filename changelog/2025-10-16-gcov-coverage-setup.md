# Gcov Code Coverage Setup - October 16, 2025

## Summary

Configured Llaminar project for comprehensive code coverage analysis using gcov and the VS Code Gcov Viewer extension.

## Changes Made

### 1. DevContainer Configuration (`.devcontainer/devcontainer.json`)

**Added gcov viewer extension** to persist across new devcontainers:
```json
"extensions": [
    // ... existing extensions ...
    "jacqueslucke.gcov-viewer"
]
```

**Added gcov viewer settings**:
```json
"gcovViewer.buildDirectories": [
    "${workspaceFolder}/build"
]
```

This pre-configures the extension to find coverage data in the build directory.

### 2. CMake Build Configuration (`CMakeLists.txt`)

**Updated Debug build flags** (lines 49-59):

**C++ Debug Flags**:
```cmake
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 --coverage -fprofile-abs-path")
set(CMAKE_C_FLAGS_DEBUG "-g -O0 --coverage -fprofile-abs-path")
```

- `-g`: Debug symbols (already present)
- `-O0`: **No optimizations** (required for accurate line coverage)
- `--coverage`: Gcov instrumentation (generates .gcno during compile, .gcda during runtime)
- `-fprofile-abs-path`: **Absolute paths** in coverage files (helps VS Code extension match files)

**Linker Flags**:
```cmake
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "--coverage")
set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "--coverage")
```

Required to link gcov runtime library for generating .gcda files.

### 3. Documentation (`.vscode/gcov-usage.md`)

Created comprehensive usage guide covering:
- Quick start workflow (build → run → view)
- Advanced usage (selecting build directory, component-specific coverage)
- Troubleshooting common issues
- Performance notes
- CI/CD integration

## Verification

### Build Verification

```bash
# Reconfigured CMake with coverage flags
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS_DEBUG="-g -O0 --coverage -fprofile-abs-path" \
  -DCMAKE_C_FLAGS_DEBUG="-g -O0 --coverage -fprofile-abs-path" \
  -DCMAKE_EXE_LINKER_FLAGS_DEBUG="--coverage"

# Built test with coverage instrumentation
cmake --build build --target test_batch_correctness --parallel
```

✅ **Result**: 692 .gcno files generated (instrumentation successful)

### Runtime Verification

```bash
# Ran test to generate coverage data
mpirun -np 2 ./build/test_batch_correctness \
  --gtest_filter="BatchCorrectnessTest.BatchedAttentionStagesParity"
```

✅ **Result**: .gcda files generated in build directory

**Sample .gcda files**:
```
/workspaces/llaminar/build/CMakeFiles/llaminar_core.dir/src/utils/DebugEnv.cpp.gcda
/workspaces/llaminar/build/CMakeFiles/llaminar_core.dir/src/TopologyManager.cpp.gcda
/workspaces/llaminar/build/CMakeFiles/llaminar_core.dir/src/QwenPipelineAdapter.cpp.gcda
/workspaces/llaminar/build/CMakeFiles/llaminar_core.dir/src/tensors/BatchedKVCache.cpp.gcda
/workspaces/llaminar/build/CMakeFiles/llaminar_core.dir/src/tensors/TensorFactory.cpp.gcda
```

## How to Use

### Quick Start

1. **Build in Debug mode** (coverage automatically enabled):
   ```bash
   cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
   cmake --build build --parallel
   ```

2. **Run tests** to generate coverage data:
   ```bash
   ./run_llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "Hello"
   # Or run specific tests
   ctest --test-dir build -R "BatchCorrectnessTest"
   ```

3. **View coverage in VS Code**:
   - Open a source file (e.g., `src/BatchQwenPipeline.cpp`)
   - Press `Ctrl+Shift+P`
   - Run: **"Gcov Viewer: Show"**
   - See inline coverage highlighting (green = executed, red = not executed)

### Reset Coverage

```bash
# Remove coverage data to start fresh
find build -name "*.gcda" -delete

# Or rebuild completely
rm -rf build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
```

## Key Features

### Automatic for Debug Builds
- ✅ Coverage instrumentation **automatically enabled** for all Debug builds
- ✅ No manual flags needed
- ✅ Just build with `-DCMAKE_BUILD_TYPE=Debug`

### VS Code Integration
- ✅ Extension pre-installed in devcontainer
- ✅ Build directory pre-configured
- ✅ One-command coverage viewing (`Ctrl+Shift+P` → "Gcov Viewer: Show")

### Comprehensive Coverage
- ✅ Line coverage (which lines executed)
- ✅ Function coverage (which functions called)
- ✅ Branch coverage (which conditional paths taken)
- ✅ Execution counts (how many times each line ran)

## Performance Impact

**Debug vs Release builds**:
- Debug (with coverage): 2-5x slower than Release
- `.gcda` files: Can grow large with complex tests
- **Recommendation**: Use Release builds for benchmarking, Debug for testing/debugging

## Files Modified

1. `.devcontainer/devcontainer.json` - Added extension and settings
2. `CMakeLists.txt` - Added coverage flags for Debug builds
3. `.vscode/gcov-usage.md` - Created comprehensive usage guide

## Next Steps

1. **In new devcontainers**: Extension will auto-install
2. **Coverage analysis**: Use `Ctrl+Shift+P` → "Gcov Viewer: Show" on any file
3. **CI/CD**: Consider adding `gcov` reports to CI pipeline
4. **Coverage goals**: Target >80% line coverage for core operators

## Related Resources

- **Gcov Manual**: https://gcc.gnu.org/onlinedocs/gcc/Gcov.html
- **Extension**: `jacqueslucke.gcov-viewer` in VS Code Marketplace
- **Usage Guide**: `.vscode/gcov-usage.md`

## Testing Notes

The first test run validated:
- ✅ 8/8 batch attention stages with exact parity
- ✅ 72.8s test execution time
- ✅ Coverage data successfully generated for 50+ source files
- ✅ Both MPI ranks generated coverage data

Coverage can now be used to identify:
- Untested code paths
- Dead code candidates for removal
- Areas needing additional test coverage
