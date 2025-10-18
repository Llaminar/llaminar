# Gcov Coverage Metrics Task - October 16, 2025

## Summary

Added a VS Code task to display coverage metrics summary directly in the terminal, complementing the visual inline coverage provided by the Gcov Viewer extension.

## Changes Made

### 1. Created Coverage Metrics Script (`.vscode/show-coverage.sh`)

**Purpose**: Generate and display coverage summary for key source files

**Features**:
- ✅ Validates coverage data exists (.gcda files)
- ✅ Validates build configuration (.gcno files)
- ✅ Shows coverage percentages for 12 core source files
- ✅ Provides helpful error messages if data is missing
- ✅ Clean, formatted output with file paths and percentages

**Sample Output**:
```
=== Gcov Coverage Summary ===

Found 86 .gcda files
Found 60 instrumented source files

Coverage Summary (sample of core files):
==========================================
📄 src/BatchQwenPipeline.cpp                     Lines executed:65.88% of 340
📄 src/QwenPipeline.cpp                          Lines executed:20.08% of 1424
📄 src/operators/MPIAttentionBatchOperator.cpp   Lines executed:85.13% of 632
📄 src/operators/MPIAttentionOperator.cpp        Lines executed:74.78% of 1249
📄 src/operators/MPILinearBatchOperator.cpp      Lines executed:65.43% of 188
📄 src/operators/MPILinearOperator.cpp           Lines executed:69.15% of 201
📄 src/operators/MPISwiGLUBatchOperator.cpp      Lines executed:80.72% of 83
📄 src/operators/MPIEmbeddingOperator.cpp        Lines executed:33.45% of 296
📄 src/AbstractPipeline.cpp                      Lines executed:54.17% of 24
📄 src/PrefillProviderBaseImpl.cpp               Lines executed:80.88% of 204
📄 src/BatchPaddingUtils.cpp                     Lines executed:41.56% of 77
📄 src/ModelLoader.cpp                           Lines executed:38.64% of 1312

✓ Coverage data available for 86 files
```

### 2. Added VS Code Task (`.vscode/tasks.json`)

**Task Name**: `gcov: show coverage metrics`

**Configuration**:
```json
{
    "label": "gcov: show coverage metrics",
    "type": "shell",
    "command": "${workspaceFolder}/.vscode/show-coverage.sh",
    "group": "test",
    "detail": "Display coverage metrics summary from gcov"
}
```

**How to Run**:
1. Press `Ctrl+Shift+P`
2. Type: "Tasks: Run Task"
3. Select: **"gcov: show coverage metrics"**

### 3. Updated Documentation (`.vscode/gcov-usage.md`)

Added two sections:
1. **Option B** under "View Coverage" - Terminal-based coverage summary
2. **VS Code Tasks** section documenting both coverage tasks:
   - `gcov: show coverage metrics`
   - `gcov: reset coverage data`

## Use Cases

### Quick Coverage Check
Instead of opening individual files to see coverage, run the task to get an overview:
- Which files have high coverage (>80%)?
- Which files need more testing (<50%)?
- What's the overall test coverage status?

### Workflow Integration
```bash
# 1. Make code changes
# 2. Build in Debug mode
Task: "cmake: build"

# 3. Run relevant tests
Task: "test: integration (heavy tests)"

# 4. Check coverage impact
Task: "gcov: show coverage metrics"

# 5. View details for specific files
Open file → Ctrl+Shift+P → "Gcov Viewer: Show"
```

### Coverage Monitoring
The task shows coverage for key components:
- **Pipelines**: BatchQwenPipeline, QwenPipeline
- **Operators**: Attention, Linear, SwiGLU, Embedding (both batch and sequential)
- **Core Infrastructure**: AbstractPipeline, PrefillProvider, ModelLoader
- **Utilities**: BatchPaddingUtils

## Key Insights from Current Coverage

Based on the sample run:

**High Coverage** (>80%):
- ✅ MPIAttentionBatchOperator: 85.13%
- ✅ PrefillProviderBaseImpl: 80.88%
- ✅ MPISwiGLUBatchOperator: 80.72%

**Good Coverage** (65-80%):
- 👍 MPIAttentionOperator: 74.78%
- 👍 MPILinearOperator: 69.15%
- 👍 BatchQwenPipeline: 65.88%
- 👍 MPILinearBatchOperator: 65.43%

**Needs Improvement** (<50%):
- ⚠️ BatchPaddingUtils: 41.56%
- ⚠️ ModelLoader: 38.64%
- ⚠️ MPIEmbeddingOperator: 33.45%
- ⚠️ QwenPipeline: 20.08%

## Technical Implementation

### Script Design Decisions

1. **Specific file list** instead of wildcard iteration
   - Ensures consistent output
   - Focuses on most important files
   - Avoids overwhelming output

2. **Error handling** for missing data
   - Checks for .gcda files (runtime coverage data)
   - Checks for .gcno files (build-time instrumentation)
   - Provides actionable error messages

3. **Change to build directory** before running gcov
   - Ensures gcov finds source files correctly
   - Matches gcov's expected working directory

4. **Filter output** to show only main source files
   - Excludes header files
   - Excludes system includes
   - Shows clean file paths

### Quote Escaping Issue

**Original Problem**: Complex bash command with nested quotes in tasks.json
```bash
bash -c 'cd ... && echo '=== Gcov ...' && ...'
# Syntax error: unmatched quotes
```

**Solution**: Separate shell script file
- Cleaner, more maintainable
- No quote escaping issues
- Easier to debug and extend
- Can be run standalone: `./.vscode/show-coverage.sh`

## Files Modified

1. **Created**: `.vscode/show-coverage.sh` (65 lines, executable)
2. **Modified**: `.vscode/tasks.json` - Added "gcov: show coverage metrics" task
3. **Modified**: `.vscode/gcov-usage.md` - Documented new task

## Integration with Existing Workflow

This complements the existing coverage infrastructure:
- **Gcov Viewer extension**: Visual inline coverage (detailed)
- **Coverage metrics task**: Terminal summary (overview)
- **Reset coverage task**: Data cleanup

Together, they provide:
- 📊 Quick overview (terminal task)
- 🔍 Detailed inspection (VS Code extension)
- 🧹 Data management (reset task)

## Next Steps

Potential enhancements:
1. Add threshold warnings (e.g., "<50% coverage" in red)
2. Generate HTML coverage reports for CI/CD
3. Track coverage trends over time
4. Add branch coverage metrics (in addition to line coverage)
5. Create coverage badges for README

## Testing Notes

**Verified**:
- ✅ Task runs successfully from VS Code task runner
- ✅ Shows coverage for 12 key source files
- ✅ Clean, readable output with file paths and percentages
- ✅ Helpful error messages when data is missing
- ✅ Works with existing coverage data (86 .gcda files)

**Dependencies**:
- Requires Debug build with coverage flags
- Requires tests to have run (to generate .gcda files)
- Uses gcov command-line tool (pre-installed in devcontainer)
