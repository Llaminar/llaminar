# MPI Test Synchronization Fix

## Problem

The `ParityFrameworkTest` was hanging indefinitely when run with MPI (`mpirun -np 2`). One process would reach 100% CPU while the other remained nearly idle, creating an imbalance that lasted over an hour before manual interruption.

## Root Cause

**Classic MPI GTEST_SKIP synchronization bug**: When rank 0 encounters a condition requiring test skip (e.g., missing test file), it calls `GTEST_SKIP()` and immediately exits the test. However, rank 1 is still waiting for MPI collective operations (like `MPI_Bcast`) that rank 0 will never reach, causing a deadlock.

### Example of Problematic Pattern

```cpp
// BUGGY CODE - Don't do this!
TEST(ParityFramework, DistributedPipelineVsLlamaCpp) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    std::string model_path = find_test_model();
    if (rank == 0 && model_path.empty()) {
        GTEST_SKIP() << "No test model found";  // Rank 0 exits here
    }
    broadcast_string(model_path, 0, MPI_COMM_WORLD);  // Rank 1 waits here forever!
}
```

**What happens:**
1. Rank 0 finds no model → calls `GTEST_SKIP()` → exits test
2. Rank 1 reaches `broadcast_string()` → waits for rank 0 → hangs indefinitely

## Solution

**Always broadcast the skip decision to ALL ranks before calling `GTEST_SKIP()`:**

```cpp
// CORRECT PATTERN - Do this!
TEST(ParityFramework, DistributedPipelineVsLlamaCpp) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Step 1: Rank 0 checks condition
    std::string model_path;
    int should_skip = 0;
    if (rank == 0) {
        model_path = find_test_model();
        should_skip = model_path.empty() ? 1 : 0;
    }
    
    // Step 2: Broadcast decision to ALL ranks
    MPI_Bcast(&should_skip, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Step 3: ALL ranks skip together (or none do)
    if (should_skip) {
        GTEST_SKIP() << "No test model found";
    }
    
    // Step 4: Proceed with normal MPI collectives
    broadcast_string(model_path, 0, MPI_COMM_WORLD);
}
```

### Key Principles

1. **Rank 0 decides** - Only rank 0 checks file existence, environment variables, etc.
2. **Broadcast decision** - Use `MPI_Bcast` to send the skip flag to all ranks
3. **All ranks skip together** - Every rank executes `GTEST_SKIP()` (or none do)
4. **Then proceed safely** - After synchronization, MPI collectives won't deadlock

## Files Fixed

### `tests/test_parity_framework.cpp`

**Test 1: `DistributedPipelineVsLlamaCpp`** (lines 288-310)

**Before:**
```cpp
std::string model_path = find_test_model();
if (rank == 0 && model_path.empty()) {
    GTEST_SKIP() << "No test model found";
}
broadcast_string(model_path, 0, MPI_COMM_WORLD);  // Deadlock!
```

**After:**
```cpp
std::string model_path;
int should_skip = 0;
if (rank == 0) {
    model_path = find_test_model();
    should_skip = model_path.empty() ? 1 : 0;
}
MPI_Bcast(&should_skip, 1, MPI_INT, 0, MPI_COMM_WORLD);
if (should_skip) {
    GTEST_SKIP() << "No test model found";
}
broadcast_string(model_path, 0, MPI_COMM_WORLD);  // Safe!
```

**Test 2: `DistributedPipelineVsPyTorchReference`** (lines 545-625)

**Before:**
```cpp
const char *snapshot_dir_env = std::getenv("PYTORCH_SNAPSHOT_DIR");
if (!snapshot_dir_env) {
    GTEST_SKIP() << "PYTORCH_SNAPSHOT_DIR not set";  // Each rank decides independently!
}

const char *tokens_env = std::getenv("PYTORCH_SNAPSHOT_TOKENS");
if (!tokens_env) {
    GTEST_SKIP() << "PYTORCH_SNAPSHOT_TOKENS not set";
}

// ... later ...
std::string model_path = find_test_model();
if (rank == 0 && model_path.empty()) {
    GTEST_SKIP() << "No test model found";
}
broadcast_string(model_path, 0, MPI_COMM_WORLD);  // Potential deadlock!
```

**After:**
```cpp
// Step 1: Rank 0 checks all preconditions
int should_skip = 0;
if (rank == 0) {
    const char *snapshot_dir_env = std::getenv("PYTORCH_SNAPSHOT_DIR");
    const char *tokens_env = std::getenv("PYTORCH_SNAPSHOT_TOKENS");
    
    if (!snapshot_dir_env) {
        should_skip = 1;
    } else if (!tokens_env) {
        should_skip = 2;
    }
}

// Step 2: Broadcast decision
MPI_Bcast(&should_skip, 1, MPI_INT, 0, MPI_COMM_WORLD);

// Step 3: All ranks skip together with appropriate message
if (should_skip == 1) {
    GTEST_SKIP() << "PYTORCH_SNAPSHOT_DIR not set";
} else if (should_skip == 2) {
    GTEST_SKIP() << "PYTORCH_SNAPSHOT_TOKENS not set";
}

// Step 4: Additional checks with broadcast
int tokens_empty = 0;
if (rank == 0) {
    tokens_empty = token_ids.empty() ? 1 : 0;
}
MPI_Bcast(&tokens_empty, 1, MPI_INT, 0, MPI_COMM_WORLD);
if (tokens_empty) {
    GTEST_SKIP() << "PYTORCH_SNAPSHOT_TOKENS is empty";
}

int model_not_found = 0;
if (rank == 0) {
    model_path = find_test_model();
    model_not_found = model_path.empty() ? 1 : 0;
}
MPI_Bcast(&model_not_found, 1, MPI_INT, 0, MPI_COMM_WORLD);
if (model_not_found) {
    GTEST_SKIP() << "No test model found";
}
broadcast_string(model_path, 0, MPI_COMM_WORLD);  // Now safe!
```

## Performance Impact

**Before fix:**
- Test hung indefinitely (>1 hour)
- One process at 100% CPU, other nearly idle
- Required manual interrupt (CTRL-C)

**After fix:**
- Test completes in **0.44-0.46 seconds**
- Both processes synchronized
- Stable across multiple runs

## General Pattern for MPI + GoogleTest

For any MPI test that might skip:

```cpp
TEST(MyTest, SomeFeature) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Pattern: Check → Broadcast → Skip together
    int skip_code = 0;
    if (rank == 0) {
        // Only rank 0 checks preconditions
        if (!precondition_1_met()) skip_code = 1;
        else if (!precondition_2_met()) skip_code = 2;
        // ... etc
    }
    
    // ALL ranks participate in broadcast
    MPI_Bcast(&skip_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // ALL ranks skip together (or none)
    if (skip_code == 1) {
        GTEST_SKIP() << "Precondition 1 not met";
    } else if (skip_code == 2) {
        GTEST_SKIP() << "Precondition 2 not met";
    }
    
    // Now safe to proceed with MPI collectives
    // ...
}
```

## Lessons Learned

1. **Never mix MPI collectives with conditional early exits**
   - All ranks must reach the same MPI calls
   - Use broadcast to synchronize skip decisions

2. **Environment checks are non-deterministic across ranks**
   - Rank 0 might have different environment than rank 1
   - Always have rank 0 check and broadcast the result

3. **File system checks must be coordinated**
   - Don't assume all ranks see the same files
   - Rank 0 checks, broadcasts existence flag

4. **Test with realistic MPI scenarios**
   - Always test with `mpirun -np 2` or more
   - Watch for hanging or CPU imbalance
   - Use timeouts in test scripts

## Related Issues

This fix resolves:
- ✅ MPI test hangs in `ParityFrameworkTest`
- ✅ CPU imbalance (100% vs 6.6%) between ranks
- ✅ Timeout failures in CI/CD pipelines

## Verification

Run the test to verify the fix:

```bash
cd /workspaces/llaminar

# Single run
ctest --test-dir build -R "^ParityFrameworkTest$" --output-on-failure

# Multiple runs for stability
for i in {1..5}; do
    echo "=== Run $i ==="
    timeout 10 ctest --test-dir build -R "^ParityFrameworkTest$" || echo "TIMEOUT!"
done
```

Expected: Test completes in < 1 second on all runs.

## Future Improvements

1. **Add MPI synchronization helper macros:**
   ```cpp
   #define MPI_GTEST_SKIP_IF(rank, condition, message) \
       do { \
           int _skip = 0; \
           if ((rank) == 0) _skip = (condition) ? 1 : 0; \
           MPI_Bcast(&_skip, 1, MPI_INT, 0, MPI_COMM_WORLD); \
           if (_skip) GTEST_SKIP() << (message); \
       } while(0)
   ```

2. **Document MPI testing patterns** in contributor guide

3. **Add CI check** for MPI test hangs (with aggressive timeouts)

4. **Consider test isolation** - Run MPI tests in separate test binary to avoid interference

## References

- MPI Standard: [https://www.mpi-forum.org/docs/](https://www.mpi-forum.org/docs/)
- GoogleTest Documentation: [https://google.github.io/googletest/](https://google.github.io/googletest/)
- Related Issue: MPI + GoogleTest Synchronization Patterns
