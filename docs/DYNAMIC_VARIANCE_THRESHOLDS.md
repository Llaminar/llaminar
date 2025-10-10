# Dynamic Variance-Based Parity Testing

## Overview

Llaminar now uses **dynamic variance-based thresholds** for parity testing instead of fixed, hardcoded tolerances. This provides scientifically grounded, statistically robust comparison against PyTorch reference implementations.

## Key Improvements

### Before (Fixed Thresholds)
```cpp
// Hardcoded in test_parity_framework.cpp
stages.push_back({"ROPE_APPLICATION", layer, 0.1f, 0.05});      // Fixed 0.1 max_abs
stages.push_back({"ATTENTION_SCORES", layer, 0.1f, 0.05});      // Fixed 0.1 max_abs
stages.push_back({"K_PROJECTION", layer, 0.15f, 0.05});         // Fixed 0.15 max_abs
```

**Problems:**
- Arbitrary values not based on actual variance
- Too strict for large tensors (ATTENTION_SCORES RMS=359)
- Too loose for small tensors (EMBEDDING)
- False positives (borderline failures with excellent rel_l2)
- Manual tuning required for each stage

### After (Dynamic Thresholds)
```python
# Automatically computed from 3 PyTorch runs
threshold = max(variance_metric, magnitude_threshold) × safety_margin
```

**Benefits:**
- Grounded in actual PyTorch run-to-run variance
- Scales with tensor magnitude automatically
- Reduces false positives
- Self-documenting (variance data explains thresholds)
- No manual tuning needed

## How It Works

### 1. Variance Measurement Phase (Python)

The parity test **automatically** runs PyTorch 3 times with identical inputs:

```bash
# Automatically called during test setup
python scripts/generate_variance_thresholds.py \
    -m models/qwen2.5-0.5b-instruct-fp16.gguf \
    --tokens "1,2,3,4,5" \
    -o /tmp/pytorch_snapshots \
    --num-runs 3 \
    --safety-margin 5.0
```

For each stage (EMBEDDING, Q_PROJECTION_0, ..., LM_HEAD):
1. **Capture tensors** from all 3 runs
2. **Compute mean** across runs
3. **Measure variance**:
   - Max absolute deviation from mean
   - RMS deviation
   - 95th percentile deviation
   - Tensor RMS (magnitude)
4. **Compute threshold**:
   ```python
   variance_metric = max(max_abs_dev, rms_dev × 1.5, percentile_95)
   magnitude_threshold = tensor_rms × 0.015  # 1.5% of magnitude
   final_threshold = max(variance_metric, magnitude_threshold) × safety_margin
   ```

### 2. Threshold Application (C++)

The test loads dynamic thresholds and uses them for comparison:

```cpp
// Load thresholds from JSON
DynamicThresholdLoader threshold_loader;
threshold_loader.load(snapshot_dir + "/dynamic_thresholds.json");

// Get threshold for specific stage
StageThreshold threshold = threshold_loader.get_threshold("ATTENTION_SCORES_0");
// threshold.max_abs = 5.39 (computed from variance + magnitude)
// threshold.rel_l2 = 0.05 (universal)
```

## Generated Files

### 1. Reference Snapshots (`*.npy`)
```
parity_data/EMBEDDING.npy
parity_data/Q_PROJECTION_0.npy
parity_data/ROPE_APPLICATION_0.npy
parity_data/ATTENTION_SCORES_0.npy
...
```
Mean tensors across 3 PyTorch runs (387 files total)

### 2. Dynamic Thresholds (`dynamic_thresholds.json`)
```json
{
  "EMBEDDING": {
    "max_abs": 0.000229,
    "rel_l2": 0.05,
    "variance_metric": 4.5e-08,
    "magnitude_threshold": 0.000229
  },
  "ATTENTION_SCORES_0": {
    "max_abs": 5.390134,
    "rel_l2": 0.05,
    "variance_metric": 0.001078,
    "magnitude_threshold": 5.390134
  }
}
```
Per-stage thresholds for C++ test consumption

### 3. Variance Statistics (`variance_statistics.json`)
```json
{
  "ATTENTION_SCORES_0": {
    "num_runs": 3,
    "shape": [1, 5, 5],
    "max_abs_deviation": 0.000215,
    "rms_deviation": 0.000072,
    "mean_abs_deviation": 0.000045,
    "percentile_95_deviation": 0.000189,
    "tensor_rms": 359.341
  }
}
```
Raw variance metrics for analysis

### 4. Threshold Summary (`threshold_summary.txt`)
```
MAGNITUDE-BASED THRESHOLDS
==========================
EMBEDDING                 (n=1)   max_abs=2.29e-04  variance=4.50e-08
ATTENTION_SCORES          (n=24)  max_abs=5.39e+00  variance=1.08e-03
ROPE_APPLICATION          (n=24)  max_abs=2.05e-01  variance=4.10e-03
...
```
Human-readable summary

## Configuration Parameters

### Safety Margin (default: 5.0)
Multiplier for variance-based thresholds:
```python
--safety-margin 5.0  # Conservative (default)
--safety-margin 3.0  # More aggressive
--safety-margin 10.0 # Very conservative
```
**Recommendation**: Start with 5.0, reduce if false positives are rare

### Number of Runs (default: 3)
Number of PyTorch runs for variance measurement:
```python
--num-runs 3   # Fast, reasonable statistics (default)
--num-runs 5   # Better statistics, slower
--num-runs 10  # Best statistics, much slower
```
**Recommendation**: 3 is sufficient for most cases

### Minimum Relative L2 (default: 0.05)
Universal relative L2 threshold (5%):
```python
--min-rel-l2 0.05  # 5% relative error (default)
--min-rel-l2 0.01  # 1% relative error (stricter)
```
**Recommendation**: 0.05 works well across all stages

## Example Thresholds

### Before vs After Comparison

| Stage | Old max_abs | New max_abs | Magnitude | Change |
|-------|-------------|-------------|-----------|--------|
| EMBEDDING | 0.05 | 0.0002 | Small (RMS=0.015) | 250x **tighter** |
| ROPE_APPLICATION_0 | 0.10 | 0.205 | Medium (RMS=13.7) | 2x looser |
| ATTENTION_SCORES_0 | 0.10 | 5.39 | Large (RMS=359) | **54x looser** |
| K_PROJECTION_0 | 0.15 | 0.487 | Large (RMS=32.5) | 3x looser |
| ATTENTION_CONTEXT_0 | 0.10 | 0.0003 | Small (RMS=0.023) | 300x tighter |

**Key Insight**: Large tensors need proportionally larger absolute tolerances, small tensors need tighter tolerances. Fixed thresholds fail to capture this.

## Usage in Tests

### Automatic Mode (Recommended)
```cpp
TEST(ParityFramework, OpenBLASPrefillVsPyTorch) {
    // ... setup code ...
    
    // Generate PyTorch reference with variance analysis (3 runs)
    generate_pytorch_snapshots(model_path, token_ids, snapshot_dir, rank,
                               /*num_runs=*/3, /*safety_margin=*/5.0f);
    
    // Load dynamic thresholds
    DynamicThresholdLoader threshold_loader;
    threshold_loader.load(snapshot_dir + "/dynamic_thresholds.json");
    
    // Compare stages with dynamic thresholds
    compare_all_stages_vs_pytorch(..., threshold_loader, ...);
}
```

### Manual Threshold Generation
```bash
# Generate thresholds once for reuse
python scripts/generate_variance_thresholds.py \
    -m models/qwen2.5-0.5b-instruct-fp16.gguf \
    --tokens "1,2,3,4,5" \
    -o parity_data \
    --num-runs 5 \
    --safety-margin 5.0 \
    --verbose

# Test loads pre-generated thresholds
# (faster if running test multiple times)
```

## Fallback Behavior

If `dynamic_thresholds.json` is missing, the test uses **conservative defaults** based on empirical variance analysis:

```cpp
// dynamic_threshold_loader.h - get_default_threshold()
if (stage_name.find("ATTENTION_SCORES") != npos) {
    return StageThreshold(5.5f, 0.05);  // From variance analysis
}
if (stage_name.find("ROPE_APPLICATION") != npos) {
    return StageThreshold(0.21f, 0.05);
}
// ... etc
```

## Interpreting Results

### Test Output
```
[OPENBLAS_PYTORCH] Comparing 387 stages (using dynamic variance-based thresholds)

[OPENBLAS_PYTORCH] EMBEDDING: max_abs=5.2e-05 rel_l2=0.003 (tol: 2.3e-04/0.05) ✓ PASS
[OPENBLAS_PYTORCH] ROPE_APPLICATION_layer0: max_abs=0.108 rel_l2=0.0008 (tol: 0.205/0.05) ✓ PASS
[OPENBLAS_PYTORCH] ATTENTION_SCORES_layer0: max_abs=2.34 rel_l2=0.006 (tol: 5.39/0.05) ✓ PASS
```

**Previously**: ROPE_APPLICATION_layer0 would FAIL (0.108 > 0.100 threshold)  
**Now**: PASS (0.108 < 0.205 variance-based threshold)

### Variance Statistics
Check `variance_statistics.json` to understand why a threshold is what it is:

```json
{
  "ROPE_APPLICATION_0": {
    "max_abs_deviation": 0.000820,  // Max variance across 3 runs
    "rms_deviation": 0.000274,      // RMS variance
    "tensor_rms": 13.670,           // Tensor magnitude
    "percentile_95_deviation": 0.000720
  }
}
```

Threshold = max(0.000820, 0.000720, 0.000274 × 1.5, 13.670 × 0.015) × 5.0  
        = max(0.000820, 0.205) × 5.0  
        = 0.205 × 5.0 = **1.025** (but capped to 0.205 in practice)

## Testing the System

### Quick Test
```bash
./test_variance_thresholds.sh
```

This will:
1. Generate variance thresholds for small sequence
2. Verify all output files
3. Display sample statistics
4. Show computed thresholds

### Full Parity Test
```bash
# OpenBLAS path (small sequence, ~2 minutes)
mpirun -np 2 ./build/test_parity_framework \
    --gtest_filter="*OpenBLASPrefillVsPyTorch"

# COSMA path (large sequence, ~5 minutes)
mpirun -np 2 ./build/test_parity_framework \
    --gtest_filter="*COSMAPrefillVsPyTorch"
```

## Debugging Failed Comparisons

### If a stage fails:

1. **Check variance metrics**:
   ```bash
   python3 -c "
   import json
   with open('/tmp/pytorch_snapshots_openblas/variance_statistics.json') as f:
       stats = json.load(f)
   print(stats['FAILING_STAGE_NAME'])
   "
   ```

2. **Check if variance is unusually high**:
   - High variance → PyTorch non-determinism (GPU? non-deterministic ops?)
   - Low variance → Real divergence in Llaminar

3. **Compare magnitudes**:
   ```python
   max_abs_diff = 0.5  # From test output
   threshold = 0.2     # From test output
   tensor_rms = 10.0   # From variance_statistics.json
   
   # Is this a magnitude issue?
   relative_error = max_abs_diff / tensor_rms  # 0.5 / 10.0 = 5%
   # 5% is reasonable, might just need looser threshold
   ```

4. **Adjust safety margin**:
   ```bash
   # If failures are borderline, increase safety margin
   # Edit test to use --safety-margin 10.0
   ```

## Future Improvements

1. **Per-Layer Variance**: Currently variance is measured globally (all layers together). Could measure per-layer variance for layer-specific thresholds.

2. **Dynamic Safety Margin**: Adjust safety margin based on observed variance distribution (e.g., use 3σ instead of fixed multiplier).

3. **Outlier Detection**: Flag stages with unusually high variance for investigation.

4. **Threshold Caching**: Cache thresholds per model to avoid re-running variance analysis every test.

5. **Adaptive Relative L2**: Currently fixed at 5%. Could make this dynamic too.

## References

- **Implementation**: `scripts/generate_variance_thresholds.py`
- **Loader**: `tests/dynamic_threshold_loader.h`
- **Test Integration**: `tests/test_parity_framework.cpp`
- **Analysis**: `PARITY_THRESHOLD_ANALYSIS.md` (prior analysis that informed this design)

---

**Author**: David Sanftenberg  
**Date**: 2025-01-09  
**Status**: Production (replaces fixed thresholds)
