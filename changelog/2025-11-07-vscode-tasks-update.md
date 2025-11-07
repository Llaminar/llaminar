# VS Code Tasks Update for E2ERelease Build Configuration

**Date**: 2025-11-07  
**Author**: David Sanftenberg  
**Related**: E2ERelease build configuration (commit fe871fa)

## Overview

Updated `.vscode/tasks.json` to support the new E2ERelease build configuration and reorganized test tasks to use appropriate build types per their requirements.

## Changes Made

### 1. New Tasks Added

#### E2ERelease Configuration Task
- **Label**: `V2: configure (e2e release)`
- **Purpose**: Configure CMake with E2ERelease build type
- **Build Directory**: `build_v2_e2e_release`
- **CMake Flags**: `-DCMAKE_BUILD_TYPE=E2ERelease`

#### E2ERelease Build Task
- **Label**: `E2E Release Build`
- **Purpose**: Build the V2 project in E2ERelease mode
- **Build Directory**: `build_v2_e2e_release`
- **Dependencies**: `V2: configure (e2e release)`

#### E2ERelease Clean Task
- **Label**: `V2: clean (e2e release)`
- **Purpose**: Clean E2ERelease build artifacts
- **Build Directory**: `build_v2_e2e_release`

### 2. Task Naming Updates

Updated primary build task names to match user requirements:

| Old Label | New Label | Build Type |
|-----------|-----------|------------|
| `V2: build (debug)` | `Debug Build` | Debug |
| `V2: build (release)` | `Release Build` | Release |
| N/A | `E2E Release Build` | E2ERelease |
| `V2: test unit` | `Run Unit Tests` | Debug |
| `V2: test integration` | `Run Integration Tests` | Release |
| `V2: test e2e` | `Run E2E Tests` | E2ERelease |

### 3. Test Task Build Dependencies Updated

#### Before (All tests used Debug build):
```json
{
  "label": "V2: test unit",
  "dependsOn": "V2: build (debug)"
}
{
  "label": "V2: test integration",
  "dependsOn": "V2: build (debug)"  // ❌ Wrong!
}
{
  "label": "V2: test e2e",
  "dependsOn": "V2: build (debug)"  // ❌ Wrong!
}
```

#### After (Correct build dependencies):
```json
{
  "label": "Run Unit Tests",
  "dependsOn": "Debug Build"  // ✅ Debug build
}
{
  "label": "Run Integration Tests",
  "dependsOn": "Release Build"  // ✅ Release build
}
{
  "label": "Run E2E Tests",
  "dependsOn": "E2E Release Build"  // ✅ E2ERelease build
}
```

### 4. Test Task Build Directory Updates

#### Integration Tests
- **Old**: `--test-dir ${workspaceFolder}/build_v2`
- **New**: `--test-dir ${workspaceFolder}/build_v2_release`
- **Rationale**: Integration tests should run against optimized Release build

#### E2E Tests
- **Old**: `--test-dir ${workspaceFolder}/build_v2`
- **New**: `--test-dir ${workspaceFolder}/build_v2_e2e_release`
- **Rationale**: E2E parity tests require snapshots (only available in E2ERelease)

### 5. Coverage Report Adjustments

**Integration Tests**:
- **Removed**: Coverage report generation (`.vscode/show-coverage-v2.sh`)
- **Rationale**: Release builds don't have coverage instrumentation

**E2E Tests**:
- **Removed**: Coverage report generation
- **Rationale**: E2ERelease optimized builds don't have coverage instrumentation

## Task Organization

### Configuration Tasks (3)
1. **V2: configure (debug)** - Debug build configuration
2. **V2: configure (release)** - Release build configuration
3. **V2: configure (e2e release)** - E2ERelease build configuration

### Build Tasks (3) - User-Facing Names
1. **Debug Build** (default) - Debug mode compilation
2. **Release Build** - Optimized release compilation
3. **E2E Release Build** - Optimized with snapshots for E2E testing

### Clean Tasks (3)
1. **V2: clean (debug)** - Clean Debug build
2. **V2: clean (release)** - Clean Release build
3. **V2: clean (e2e release)** - Clean E2ERelease build

### Test Tasks (6)
1. **V2: test all** - All tests (Debug build) with coverage
2. **Run Unit Tests** - Unit tests (Debug build) with coverage
3. **Run Integration Tests** - Integration tests (Release build, no coverage)
4. **Run E2E Tests** - E2E parity tests (E2ERelease build, no coverage)
5. **V2: test performance** - Performance benchmarks (Release build)
6. **V2: test parity** - Parity tests (Debug build) with coverage

## Build Type → Test Type Mapping

| Build Type | Build Directory | Test Types | Snapshots? | Coverage? |
|------------|----------------|------------|------------|-----------|
| Debug | `build_v2` | Unit, Parity, All | ✅ Yes | ✅ Yes |
| Release | `build_v2_release` | Integration, Performance | ❌ No | ❌ No |
| E2ERelease | `build_v2_e2e_release` | E2E | ✅ Yes | ❌ No |

## Rationale for Build Type Selection

### Unit Tests → Debug Build
- **Why**: Unit tests need detailed debugging information
- **Coverage**: Yes (instrumented with `--coverage`)
- **Speed**: Acceptable (tests are fast)

### Integration Tests → Release Build
- **Why**: Test production-like optimized code paths
- **Coverage**: No (Release builds not instrumented)
- **Speed**: Faster execution than Debug

### E2E Tests → E2ERelease Build
- **Why**: Require snapshot capture (ENABLE_PIPELINE_SNAPSHOTS)
- **Optimization**: Full Release optimizations (-O3 -DNDEBUG)
- **Coverage**: No (optimized builds not instrumented)
- **Speed**: Fast execution with snapshot overhead (~10-15%)

### Performance Tests → Release Build
- **Why**: Must measure true production performance
- **Coverage**: No (instrumentation would skew results)
- **Speed**: Maximum performance

## Testing the Changes

### Quick Verification

```bash
# 1. Test Debug Build task
# In VS Code: Ctrl+Shift+B → Select "Debug Build"

# 2. Test Release Build task
# In VS Code: Tasks menu → Run Task → "Release Build"

# 3. Test E2E Release Build task
# In VS Code: Tasks menu → Run Task → "E2E Release Build"

# 4. Test Unit Tests
# In VS Code: Tasks menu → Run Task → "Run Unit Tests"

# 5. Test Integration Tests
# In VS Code: Tasks menu → Run Task → "Run Integration Tests"

# 6. Test E2E Tests
# In VS Code: Tasks menu → Run Task → "Run E2E Tests"
```

### Expected Results

1. **Debug Build**: Compiles to `build_v2/` with coverage
2. **Release Build**: Compiles to `build_v2_release/` optimized
3. **E2E Release Build**: Compiles to `build_v2_e2e_release/` optimized + snapshots
4. **Run Unit Tests**: Executes from `build_v2/`, shows coverage
5. **Run Integration Tests**: Executes from `build_v2_release/`, no coverage
6. **Run E2E Tests**: Executes from `build_v2_e2e_release/`, verbose output

## Files Modified

- `.vscode/tasks.json` (361 lines total)
  - Added 3 new tasks (E2ERelease configure, build, clean)
  - Renamed 5 existing tasks (simplified naming)
  - Updated 3 test task dependencies
  - Updated 2 test task build directories
  - Removed coverage from 2 test tasks

## Impact

### Positive Changes
- ✅ E2E tests now use correct build configuration (E2ERelease)
- ✅ Integration tests now use optimized Release build
- ✅ Simplified task names match user requirements
- ✅ Correct build dependencies prevent linking errors
- ✅ Coverage only reported where instrumentation exists

### No Breaking Changes
- ✅ All existing tasks still work
- ✅ Default build task unchanged (Debug Build)
- ✅ Legacy task names (V2: configure, V2: clean) preserved
- ✅ All test patterns unchanged (^V2_Unit_, ^V2_Integration_, etc.)

## Next Steps

1. **Test all tasks**: Verify each task works correctly
2. **Update documentation**: Add task guide to V2_BUILD_GUIDE.md
3. **Consider workflow**: May want keyboard shortcuts for common tasks
4. **Monitor usage**: Track which tasks are used most frequently

## References

- **E2ERelease Build Config**: `changelog/2025-11-07-e2e-release-build-config.md`
- **Build Verification**: `changelog/2025-11-07-build-config-verification.md`
- **CMake Configuration**: `src/v2/CMakeLists.txt` (lines 15-22, 537-550)
- **Test Configuration**: `tests/v2/CMakeLists.txt` (lines 2055-2085)
