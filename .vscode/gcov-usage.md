# Using Gcov Code Coverage in Llaminar

This project is configured to support code coverage analysis using gcov and the Gcov Viewer extension in VS Code.

## Quick Start

### 1. Build in Debug Mode with Coverage

Coverage instrumentation is automatically enabled for Debug builds:

```bash
# Configure for Debug (includes --coverage flags)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build build --parallel
```

The Debug build includes:
- `-g`: Debug symbols
- `-O0`: No optimizations (ensures accurate line coverage)
- `--coverage`: Gcov instrumentation (generates .gcno files during compilation)
- `-fprofile-abs-path`: Absolute paths in coverage files (helps extension match files)

### 2. Run Your Program or Tests

Running the program generates `.gcda` files with execution counts:

```bash
# Run main program
./run_llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "Hello"

# Or run specific tests
mpirun -np 2 ./build/test_batch_correctness
ctest --test-dir build -R "BatchCorrectnessTest"
```

**Note**: Each execution creates/updates `.gcda` files in the build directory next to the corresponding `.o` files.

### 3. View Coverage in VS Code

**Option A: Visual inline coverage (recommended)**
1. Open a source file (e.g., `src/BatchQwenPipeline.cpp`)
2. Press `Ctrl+Shift+P` and run: **"Gcov Viewer: Show"**
3. The extension will:
   - Run `gcov` to parse `.gcno` and `.gcda` files
   - Display coverage inline:
     - **Green highlight**: Lines executed
     - **Red highlight**: Lines not executed
     - **Gray**: Non-executable lines
   - Show execution counts in the gutter

**Option B: Coverage summary in terminal**
1. Press `Ctrl+Shift+P` and run: **"Tasks: Run Task"**
2. Select: **"gcov: show coverage metrics"**
3. View coverage percentages for key files in the terminal

Example output:
```
=== Gcov Coverage Summary ===

Found 86 .gcda files
Found 60 instrumented source files

Coverage Summary (sample of core files):
==========================================
📄 src/BatchQwenPipeline.cpp                     Lines executed:65.88% of 340
📄 src/operators/MPIAttentionBatchOperator.cpp   Lines executed:85.13% of 632
📄 src/operators/MPILinearBatchOperator.cpp      Lines executed:65.43% of 188
📄 src/PrefillProviderBaseImpl.cpp               Lines executed:80.88% of 204
...
```

### Reset Coverage

To start fresh:

```bash
# Remove all .gcda files
find build -name "*.gcda" -delete

# Or use VS Code task: Ctrl+Shift+P → Tasks: Run Task → "gcov: reset coverage data"

# Or rebuild from scratch
rm -rf build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
```

## VS Code Tasks

The project includes helpful VS Code tasks for coverage:

1. **`gcov: show coverage metrics`** - Display coverage summary in terminal
   - Shows coverage percentages for key source files
   - Quick overview without opening individual files
   
2. **`gcov: reset coverage data`** - Delete all .gcda files
   - Useful when starting a new coverage analysis
   - Cleans up stale coverage data

Access tasks via:
- `Ctrl+Shift+P` → "Tasks: Run Task"
- Or Terminal menu → "Run Task"

## Advanced Usage

### Selecting Build Directory

If the extension doesn't find coverage data automatically:

1. Press `Ctrl+Shift+P`
2. Run: **"Gcov Viewer: Select Build Directory"**
3. Choose the `build` directory

The build directory is pre-configured in `.devcontainer/devcontainer.json`:
```json
"gcovViewer.buildDirectories": [
    "${workspaceFolder}/build"
]
```

### Coverage for Specific Components

To analyze coverage for specific components:

```bash
# Build everything
cmake --build build --parallel

# Run only specific test
mpirun -np 2 ./build/test_batch_correctness \
  --gtest_filter="BatchCorrectnessTest.BatchedAttentionStagesParity"

# View coverage for BatchQwenPipeline.cpp
# Open file and run "Gcov Viewer: Show"
```

### Combining Multiple Test Runs

Gcov accumulates coverage data across runs. To get comprehensive coverage:

```bash
# Build
cmake --build build --parallel

# Run multiple test suites
ctest --test-dir build -R "BatchCorrectnessTest"
ctest --test-dir build -R "ParityFrameworkTest"
ctest --test-dir build -R "MPIOperatorTests"

# View combined coverage
```

## Troubleshooting

### Coverage data not found

**Problem**: Extension says "No coverage data found"

**Solutions**:
1. Verify you're in Debug mode: `grep "CMAKE_BUILD_TYPE:STRING=Debug" build/CMakeCache.txt`
2. Check `.gcda` files exist: `find build -name "*.gcda" | head`
3. Run the program/tests to generate `.gcda` files
4. Manually select build directory: `Ctrl+Shift+P` → "Gcov Viewer: Select Build Directory"

### Source file not matching coverage data

**Problem**: Coverage shown for wrong file or not shown

**Solution**: The `-fprofile-abs-path` flag ensures absolute paths are used. If issues persist:
1. Rebuild from scratch: `rm -rf build && cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug`
2. Verify paths in `.gcno` files: `strings build/CMakeFiles/llaminar_core.dir/src/BatchQwenPipeline.cpp.gcno | grep BatchQwen`

### Coverage showing 0% for executed code

**Problem**: Code executed but coverage shows 0%

**Solutions**:
1. Ensure linking with `--coverage`: Check `CMAKE_EXE_LINKER_FLAGS_DEBUG` includes `--coverage`
2. Verify `.gcda` files updated: `ls -l build/**/*.gcda` (check timestamps)
3. No optimizations: Ensure `CMAKE_BUILD_TYPE=Debug` (not Release)

## Understanding Coverage Metrics

- **Line Coverage**: Percentage of executable lines that were executed
- **Function Coverage**: Percentage of functions that were called
- **Branch Coverage**: Percentage of conditional branches taken

In the Gcov Viewer:
- **Green numbers**: Execution count (how many times line ran)
- **Red background**: Line never executed
- **Green background**: Line executed at least once
- **#####**: Branch not taken

## Performance Notes

- **Debug builds are slower**: Coverage-instrumented builds run 2-5x slower than Release
- **Use for testing, not benchmarking**: Always use Release builds for performance measurements
- **Storage**: `.gcda` files can grow large with complex tests (clean periodically)

## Integration with CI/CD

For automated coverage reports in CI:

```bash
# In CI script
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure

# Generate coverage report
gcov build/CMakeFiles/llaminar_core.dir/src/*.gcno
# Or use lcov for HTML reports
```

## Related Documentation

- **Gcov Manual**: https://gcc.gnu.org/onlinedocs/gcc/Gcov.html
- **Extension Docs**: Search "Gcov Viewer" in VS Code extensions
- **CMake Coverage**: Our build configured in `CMakeLists.txt` lines 49-59
