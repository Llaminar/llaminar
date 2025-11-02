# Hash-Based Automatic ML Heuristic Retraining

**Date**: November 2, 2025  
**Feature**: Automatic hash-based retraining for CUDA GEMM ML heuristic  
**Component**: Build system automation

## Summary

Implemented **hash-based automatic retraining** that only regenerates ML heuristic weights when benchmark CSV data actually changes. This provides a middle ground between:
- ❌ **Manual retraining**: Requires developer to remember to retrain after running benchmarks
- ❌ **Always retrain**: Wastes 30-60 min on every performance test run

✅ **Hash-based auto-retrain**: Automatically detects CSV changes and retrains only when needed

## How It Works

### Hash-Based Caching

```bash
# 1. Benchmark runs and generates CSV
./build_v2/performance/v2_perf_cuda_heuristic_validation
# Output: build_v2/cuda_gemm_benchmark_data.csv (~132,192 rows)

# 2. Compute SHA256 hash of CSV
sha256sum cuda_gemm_benchmark_data.csv
# Example: a3f5e9c1b2d4... (64 hex chars)

# 3. Compare with stored hash
cat build_v2/.cuda_gemm_benchmark_data.csv.sha256
# If different: retrain ML model
# If same: skip training, use existing weights

# 4. After successful training, update hash file
echo "a3f5e9c1b2d4..." > build_v2/.cuda_gemm_benchmark_data.csv.sha256
```

### Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ Run Benchmark                                               │
│   ./build_v2/performance/v2_perf_cuda_heuristic_validation │
│   └─> cuda_gemm_benchmark_data.csv                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Compute Hash                                                │
│   NEW_HASH=$(sha256sum csv)                                 │
│   OLD_HASH=$(cat .csv.sha256)                               │
└─────────────────────────────────────────────────────────────┘
                          ↓
                  ┌───────────────┐
                  │ Hash Changed? │
                  └───────────────┘
                    ↙           ↘
              YES (retrain)   NO (skip)
                    ↓               ↓
    ┌───────────────────────┐   ┌──────────────────┐
    │ Train ML Model        │   │ Use Existing     │
    │ (30 sec)              │   │ Weights          │
    │ └─> weights.h         │   │ ✓ Up-to-date     │
    └───────────────────────┘   └──────────────────┘
                    ↓
    ┌───────────────────────┐
    │ Update Hash File      │
    │ echo $NEW_HASH > .sha │
    └───────────────────────┘
```

## Implementation

### Files Modified

**1. `scripts/train_cuda_heuristic.sh`** (Enhanced)
- Added hash computation (SHA256)
- Added hash comparison logic
- Skips training if hash unchanged
- Updates hash file after successful training
- Improved status messages (🆕 🔄 ⏭️ ✅)

**2. `tests/v2/CMakeLists.txt`** (New CMake target)
- Added `train_cuda_heuristic_auto` target
- Calls check script + training script
- Only retrains if check script returns 0

**3. `scripts/check_cuda_heuristic_needs_retrain.sh`** (New utility)
- Standalone script to check retraining status
- Returns 0 if retrain needed, 1 if up-to-date
- Useful for CI/CD pipelines

### New CMake Target

```cmake
# Hash-based automatic retraining (only retrains if CSV changed)
add_custom_target(train_cuda_heuristic_auto
    COMMAND ${CMAKE_COMMAND} -E echo "Checking if ML retraining needed (hash-based)..."
    COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/check_cuda_heuristic_needs_retrain.sh && 
            bash ${PROJECT_SOURCE_DIR}/scripts/train_cuda_heuristic.sh || 
            ${CMAKE_COMMAND} -E echo "Heuristic weights already up-to-date"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Auto-train CUDA GEMM heuristic (only if CSV hash changed)"
    VERBATIM
)
```

## Usage

### Method 1: CMake Target (Recommended)

```bash
# Build benchmark
cd /workspaces/llaminar
cmake --build build_v2 --target v2_perf_cuda_heuristic_validation

# Run benchmark (generates CSV)
./build_v2/performance/v2_perf_cuda_heuristic_validation

# Auto-retrain (only if CSV changed)
cmake --build build_v2 --target train_cuda_heuristic_auto

# Output if unchanged:
# ⏭️  CSV data unchanged - skipping ML training
# ✓ Using existing cuda_heuristic_weights.h

# Output if changed:
# 🔄 CSV data changed - retraining required
# [Training ML model...]
# ✓ ML model trained successfully
```

### Method 2: Shell Script (Direct)

```bash
# Enhanced script now includes hash checking
./scripts/train_cuda_heuristic.sh

# Automatically skips if CSV unchanged:
# [3/5] Checking if ML model retraining needed...
#       New data hash: a3f5e9c1b2d4...
#       Old data hash: a3f5e9c1b2d4...
#       ⏭️  CSV data unchanged - skipping ML training
#       ✓ Using existing cuda_heuristic_weights.h
```

### Method 3: Manual Check (CI/CD)

```bash
# Check retraining status (exit code indicates need)
./scripts/check_cuda_heuristic_needs_retrain.sh

# Exit codes:
# 0 = retraining needed (new hash)
# 1 = up-to-date (hash unchanged)
# 2 = error (CSV missing)

# Example CI/CD usage:
if ./scripts/check_cuda_heuristic_needs_retrain.sh; then
    echo "Retraining required, running full pipeline..."
    cmake --build . --target train_cuda_heuristic_auto
else
    echo "Heuristic weights up-to-date, skipping training"
fi
```

## When Training Happens

### Scenarios that Trigger Retraining

1. **New Test Cases Added**
   - Added Qwen3-MoE 235B tests → CSV changes → hash differs → retrain
   - Added DeepSeek V3 tests → CSV changes → hash differs → retrain

2. **Configuration Space Expanded**
   - Expanded tile sizes from 648 → 3,888 configs → CSV changes → retrain
   - Added new atom layouts → CSV changes → retrain

3. **Hardware Changes**
   - Running on different GPU (RTX 4090 → A100) → GFLOPS values change → hash differs → retrain

4. **Performance Improvements**
   - Kernel optimizations change GEMM performance → CSV changes → retrain

### Scenarios that Skip Training

1. **Rebuild Without Benchmark**
   ```bash
   # Just rebuilding code, not running benchmarks
   cmake --build build_v2 --target cuda_backend
   # CSV unchanged → skip training
   ```

2. **Multiple Builds**
   ```bash
   # First run: benchmark + retrain
   ./build_v2/performance/v2_perf_cuda_heuristic_validation
   cmake --build build_v2 --target train_cuda_heuristic_auto
   # Training happens
   
   # Second run: just check
   cmake --build build_v2 --target train_cuda_heuristic_auto
   # ⏭️ Skipped (hash unchanged)
   ```

3. **CI/CD Caching**
   ```bash
   # Cache both CSV and hash file
   # Future CI runs check hash before expensive training
   ```

## Performance Impact

### Before (Manual Workflow)
```
Developer workflow:
1. Run benchmark (45-75 min) ✓
2. Remember to retrain (often forgotten!) ❌
3. Manually run training script (30 sec)
4. Rebuild (2-3 min)

Issues:
- Easy to forget retraining
- Wastes time training when unnecessary
- No automatic tracking
```

### After (Hash-Based Auto-Retrain)
```
Developer workflow:
1. Run benchmark (45-75 min) ✓
2. Run train_cuda_heuristic_auto (automatic check!) ✓
3. Rebuild (2-3 min)

Benefits:
- Automatically detects when retraining needed
- Skips training if CSV unchanged (saves 30 sec)
- Clear status messages
- Works seamlessly in CI/CD
```

### Time Savings

| Scenario | Before | After | Savings |
|----------|--------|-------|---------|
| Benchmark run, CSV unchanged | 45 min + 30 sec training | 45 min + instant skip | 30 sec |
| Multiple builds, no benchmark | Manual check needed | Automatic skip | Mental overhead |
| CI/CD rebuild | Always retrain (wasteful) | Hash-based skip | 30 sec × builds |

## Hash File Details

### Location
```bash
build_v2/.cuda_gemm_benchmark_data.csv.sha256
```

### Format
```
a3f5e9c1b2d4f8a7c6b5e4d3f2a1b0c9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3
```
- Single line
- 64 hex characters (SHA256 hash)
- Computed from entire CSV file (all ~132,192 rows)

### Why SHA256?
- **Fast**: Compute hash in <100ms even for large CSV
- **Reliable**: Cryptographic hash, no collisions
- **Standard**: Available on all Linux systems
- **Portable**: Same hash across different machines

## Edge Cases Handled

### 1. Benchmark Failure
```bash
# If benchmark fails during CSV generation
./build_v2/performance/v2_perf_cuda_heuristic_validation
# ERROR: Benchmark failed

# Old CSV preserved (backed up to .old)
# Hash file unchanged
# Training skipped (old weights still valid)
```

### 2. Missing CSV
```bash
./scripts/check_cuda_heuristic_needs_retrain.sh
# ❌ No benchmark data found at build_v2/cuda_gemm_benchmark_data.csv
#    Run: ./build_v2/performance/v2_perf_cuda_heuristic_validation
# Exit code: 2 (error)
```

### 3. First Run (No Hash File)
```bash
# No .sha256 file exists yet
./scripts/train_cuda_heuristic.sh
# [3/5] Checking if ML model retraining needed...
#       New data hash: a3f5e9c1...
#       🆕 No previous hash found - training required
# [Training proceeds...]
# [Hash file created]
```

### 4. Corrupted Hash File
```bash
# Empty or malformed hash file
cat build_v2/.csv.sha256
# (empty)

# Script treats as "no hash" → trains
# Overwrites with valid hash after training
```

## Integration with Existing Workflow

### No Breaking Changes
```bash
# Old manual workflow still works
cmake --build . --target train_cuda_heuristic
# Always retrains (ignores hash)

# New automatic workflow
cmake --build . --target train_cuda_heuristic_auto
# Hash-based (skips if unchanged)

# Shell script enhanced but backward compatible
./scripts/train_cuda_heuristic.sh
# Now includes hash checking automatically
```

### Migration Path
1. **Phase 1**: Use `train_cuda_heuristic_auto` manually
2. **Phase 2**: Update documentation to recommend auto target
3. **Phase 3**: Consider making auto the default (optional)

## Monitoring and Debugging

### Check Current Status
```bash
# Manual check
./scripts/check_cuda_heuristic_needs_retrain.sh

# Example outputs:
✅ Heuristic weights up-to-date
   CSV hash: a3f5e9c1b2d4...
   No retraining needed

🔄 Benchmark data changed - retraining required
   Old hash: a3f5e9c1...
   New hash: b4d6f8a2...
   Run: ./scripts/train_cuda_heuristic.sh
```

### Force Retraining
```bash
# Option 1: Delete hash file
rm build_v2/.cuda_gemm_benchmark_data.csv.sha256
./scripts/train_cuda_heuristic.sh
# Treats as new data → retrains

# Option 2: Use non-auto target
cmake --build . --target train_cuda_heuristic
# Always retrains (ignores hash)

# Option 3: Delete weights file
rm src/v2/kernels/cuda/cuda_heuristic_weights.h
./scripts/check_cuda_heuristic_needs_retrain.sh
# Reports: 🆕 No weights file found - training required
```

### Debugging Hash Mismatches
```bash
# Compare hashes manually
sha256sum build_v2/cuda_gemm_benchmark_data.csv
# a3f5e9c1b2d4...

cat build_v2/.cuda_gemm_benchmark_data.csv.sha256
# a3f5e9c1b2d4...

# If different: CSV changed since last training
# Likely causes:
#   - New test cases added
#   - Configuration space expanded
#   - Hardware changed (different GPU)
#   - Benchmark code modified
```

## Future Enhancements

### Potential Improvements

1. **Multi-Hash Tracking**
   - Track hash of source code (test file) separately
   - Detect when test structure changes (not just data)

2. **Incremental Training**
   - Detect which tests changed
   - Retrain only on new data (append to model)

3. **Parallel Hash Checking**
   - Check hash during benchmark run
   - Start training immediately when benchmark finishes

4. **Hash in Generated Header**
   - Embed CSV hash in `cuda_heuristic_weights.h`
   - Runtime verification: "These weights trained on hash X"

5. **CI/CD Integration**
   ```yaml
   # .github/workflows/ml-heuristic.yml
   - name: Check if retraining needed
     id: check
     run: ./scripts/check_cuda_heuristic_needs_retrain.sh
     
   - name: Train ML model
     if: steps.check.outcome == 'success'
     run: cmake --build . --target train_cuda_heuristic_auto
   ```

## Related Documentation

- `docs/ML_HEURISTIC_TRAINING.md` - Comprehensive training workflow
- `CUDA_ML_HEURISTIC_SUMMARY.md` - Quick reference
- `changelog/2025-11-02-ml-heuristic-documentation-and-extension.md` - Initial 32B/72B extension
- `changelog/2025-11-02-deepseek-v3-671b-coverage.md` - DeepSeek V3 extension
- `changelog/2025-11-02-qwen3moe-235b-coverage.md` - Qwen3-MoE extension

## Conclusion

Hash-based automatic retraining provides the best of both worlds:
- ✅ **Automatic**: Detects when retraining needed without manual tracking
- ✅ **Efficient**: Skips training when CSV unchanged (saves 30 sec)
- ✅ **Transparent**: Clear status messages show what's happening
- ✅ **CI/CD Ready**: Exit codes and scripts integrate with automation
- ✅ **Non-Breaking**: Existing manual workflow still works

**Recommended Usage**:
```bash
# After running benchmarks, always use:
cmake --build build_v2 --target train_cuda_heuristic_auto

# Script automatically:
# - Checks CSV hash
# - Skips if unchanged (⏭️)
# - Retrains if changed (🔄)
# - Updates hash file (✓)
```

**Time Investment**: ~30 min implementation, saves 30 sec per unnecessary retrain × ∞ future builds = significant long-term savings + eliminates mental overhead!
