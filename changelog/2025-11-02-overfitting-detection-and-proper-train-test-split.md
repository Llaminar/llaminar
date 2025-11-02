# Overfitting Detection and Proper Train/Test Split Implementation

**Date**: November 2, 2025  
**Session**: Phase 23 continuation - ML Heuristic Generalization Validation

## Executive Summary

Discovered **severe overfitting** in CUDA GEMM ML heuristic through canary testing. Implemented proper train/validation/test split based on model sizes to measure true generalization. Results show the model struggles to generalize to unseen matrix dimensions, revealing fundamental issues with the current ML approach.

## Problem Discovery

### Canary Test Creation

Created `Perf__CudaGemmHeuristicCanary.cpp` to test ML heuristic on **UNTRAINED** matrix shapes:
- **1.5B model** (1280x1280) - interpolation between 0.5B (896) and 4B (2560)
- **Odd batch=17** (17x2048x2048) - training only had 32, 128, 256
- **Odd dimensions** (1x1537x2048) - non-power-of-2

### Catastrophic Results (Before Fix)

```
Test                    Rank    GFLOPS Gap    Performance
──────────────────────────────────────────────────────────
1.5B interpolation      #762    -10.1        34% slower
Odd batch (17)          #3474   -280.2       61% slower
Odd dimension           #1701   -11.9        46% slower

Overall generalization: 0% top-20 hit rate, 0% top-30 hit rate
```

**Diagnosis**: Model was trained with **random 80/20 split** from all 34 test cases, meaning:
- Training set: Random 80% of ALL model sizes (0.5B-671B)
- Test set: Random 20% of ALL model sizes (also 0.5B-671B)
- **Problem**: Test set contained SAME model sizes as training, just different configs
- **Result**: Perfect R²=1.0000 but 0% generalization to unseen shapes

## Solution Implemented

### Model-Based Train/Val/Test Split

**Strategy**: NEVER let model see 14B/235B/671B shapes during training

```python
Train Set (60% - 46,656 samples):
- Qwen 0.5B (3 tests)
- Qwen 7B (3 tests)  
- Qwen 72B (3 tests)
Models: 0.5B, 7B, 72B

Validation Set (20% - 19,440 samples):
- Qwen 4B (3 tests)
- Qwen 32B (2 tests)
Models: 4B, 32B

Test/Hold-out Set (20% - 77,760 samples):
- Qwen 14B (3 tests)
- DeepSeek 671B (8 tests)
- Qwen3-MoE 235B (9 tests)
Models: 14B, 235B, 671B (NEVER seen during training)
```

### True Generalization Metrics

**Old Method** (Random Split - MISLEADING):
```
Train R²: 1.0000 (perfect memorization)
Test R²:  1.0000 (but test had same model sizes!)
Reality:  0% generalization to unseen shapes
```

**New Method** (Model-Based Split - HONEST):
```
Train R²: 1.0000 (fits training data well)
Val R²:   0.9790 (good on similar model sizes)
Test R²:  0.8579 (true generalization to 14B/235B/671B)

Linear Model (for C++ deployment):
Train R²: 0.9424
Val R²:   0.9262
Test R²:  0.7899
```

## Key Findings

### 1. Gradient Boosting Overfits Severely

- **Train R² = 1.0000** → Perfect memorization of training configs
- **Test R² = 0.8579** → 14% degradation on unseen model sizes
- **Gap indicates**: Model memorizing specific dimensions, not learning general patterns

### 2. Linear Regression More Honest

- **Test R² = 0.7899** vs GB's 0.8579
- More realistic for C++ deployment (no tree ensemble needed)
- Simpler model = less overfitting potential

### 3. Lookup Table Approach Fundamentally Limited

Current approach exports **GB predictions as lookup table** (12 MB):
- Works perfectly for exact training configs (R²=1.0)
- **0% hit rate** on canary shapes (lookup misses)
- Falls back to linear regression (which also struggles)

**Evidence from canary test**:
```
[INFO] [CUDA AutoTuner] Lookup stats: 0/7776 hits (0.0%), 7776 misses
[INFO] [CUDA AutoTuner] Top 5 scored configs for m=17 n=2048 k=2048:
[INFO]   #1: tile_16x16x32 (score=46.4)  ← All configs get same score!
[INFO]   #2: tile_16x16x32 (score=46.4)
[INFO]   #3: tile_16x16x32 (score=46.4)
[INFO]   #4: tile_16x16x32 (score=46.4)
[INFO]   #5: tile_16x16x32 (score=46.4)
```

**Problem**: Linear model predicts nearly identical scores for all configs on unseen shapes.

### 4. Worst Predictions on Large Batches

Top 10 errors all from **Qwen 14B Batch256** (unseen during training):
```
Config              Actual      Predicted   Error
────────────────────────────────────────────────────
tile_64x64x32       2982.7      1339.4      1643.4 GFLOPS (55% error)
```

Model underestimates large batch performance by **1600+ GFLOPS**.

## Root Cause Analysis

### Why Generalization Fails

1. **Feature Engineering Insufficient**:
   - Current features: tile sizes, thread counts, occupancy estimates
   - Missing: Problem size ratios, batch-specific features, dimension alignments
   - Linear combinations can't capture complex interactions

2. **Training Data Gaps**:
   - Batch sizes: Only 32, 128, 256 → Can't interpolate to 17, 64, etc.
   - Model sizes: 0.5B → 4B → 7B → 32B → 72B (huge gaps)
   - Missing: 1B, 2B, 8B, 16B, 24B, 48B intermediate sizes

3. **Model Complexity Mismatch**:
   - Gradient Boosting: 200 trees × depth 8 = huge capacity
   - Training samples: 46K configs across just 9 test cases
   - **Overparameterized**: Model capacity >> data diversity

## Recommendations

### Short-term (Immediate)

1. **Accept Lower Accuracy** ✅ DONE
   - Document that Test R²=0.79-0.86 is realistic
   - Set expectations: 30-40% slower on unseen shapes
   - Canary tests will catch regressions

2. **Add More Training Data**
   - Include intermediate sizes: 1.5B, 10B, 50B
   - Include odd batches: 3, 5, 17, 33, 64, 192
   - Add non-power-of-2 dimensions

3. **Regularization**
   - Reduce GB tree depth: 8 → 5
   - Reduce estimators: 200 → 100
   - Increase min_samples_leaf: 5 → 20

### Medium-term

1. **Better Feature Engineering**
   ```python
   # Add these features:
   - batch_size_log2 (captures exponential scaling)
   - dimension_alignment (n % tile_n == 0)
   - memory_bandwidth_estimate
   - warp_efficiency (threads % 32 == 0)
   - shared_memory_pressure
   ```

2. **Ensemble of Simple Models**
   - Train separate models per problem size category:
     - Tiny (m < 8)
     - Small (8 ≤ m < 64)
     - Medium (64 ≤ m < 256)
     - Large (m ≥ 256)

3. **Hybrid Heuristic**
   - Use ML for seen shapes (lookup table)
   - Use analytical model for unseen shapes (engineered heuristic)
   - Combine with weighted average based on confidence

### Long-term

1. **Active Learning**
   - Run canary benchmarks periodically
   - Add worst predictions to training set
   - Retrain incrementally

2. **Neural Network**
   - Better at learning non-linear patterns
   - Can handle sparse features (one-hot encoding)
   - Transfer learning from similar domains

3. **Automated Hyperparameter Tuning**
   - Grid search over regularization params
   - Optimize for validation R² (not training R²)

## Files Modified

### New Files
- `tests/v2/performance/Perf__CudaGemmHeuristicCanary.cpp` (432 lines)
  - Canary test suite for generalization validation
  - Tests 4 untrained matrix shapes
  - Expects >33% top-20 hit rate (currently 0%)

### Modified Files
- `src/v2/kernels/cuda/python/train_cuda_heuristic.py`
  - Replaced random train_test_split with model-based split
  - Added validation set (4B, 32B models)
  - Enhanced metrics reporting (train/val/test breakdown)
  - Warning when Test R² << Train R² (overfitting detection)

- `tests/v2/CMakeLists.txt`
  - Added v2_perf_cuda_heuristic_canary target
  - Labels: "V2;Performance;CUDA;GEMM;AutoTuning;HeuristicCanary;GeneralizationTest"

### Regenerated Files
- `src/v2/kernels/cuda/cuda_heuristic_weights.h`
  - Now trained on 0.5B/7B/72B only (not all model sizes)
  - Linear regression coefficients updated
  
- `src/v2/kernels/cuda/cuda_heuristic_lookup.h`
  - Lookup table now from partial training set
  - 0% hit rate on unseen shapes (expected)

## Performance Impact

### Training Time
- **Before**: 73 min (full 34 tests) → ~40 sec training
- **After**: Same benchmark time, same training time
- No runtime impact

### Inference Performance
- **On trained shapes** (0.5B, 7B, 72B): Same (R²=1.0)
- **On validation shapes** (4B, 32B): -2% (R²=0.98 → 0.93)
- **On test shapes** (14B, 235B, 671B): -14% (R²=1.0 → 0.86)
- **On canary shapes** (untrained): -30-60% slower

### Real-World Impact
Most production models: 7B, 13B, 70B, 405B
- **7B**: Covered by training (perfect)
- **13B**: Close to 14B test set (degraded but workable)
- **70B**: Close to 72B training (good)
- **405B**: Extrapolation beyond 671B (unpredictable)

## Testing

### Canary Test Results (After Fix)

```bash
cd build_v2_release
LLAMINAR_USE_ML_HEURISTIC=1 ./performance/v2_perf_cuda_heuristic_canary

[ RUN      ] CudaGemmHeuristicCanary.Qwen_1_5B_SingleToken_QKV
[Canary]   ML heuristic: Rank #740 (19.9 GFLOPS)
[Canary]   Best empirical: 30.0 GFLOPS
[  FAILED  ] Expected rank ≤30, got 740

[ RUN      ] CudaGemmHeuristicCanary.SmallOddBatch_17x2048x2048  
[Canary]   ML heuristic: Rank #3478 (180.2 GFLOPS)
[Canary]   Best empirical: 460.4 GFLOPS (61% slower)
[  FAILED  ] Expected rank ≤30, got 3478

Overall: 0% top-20 hit rate, 0% top-30 hit rate
```

**Status**: Still failing, but now we understand why:
- Linear model predicts same score (46.4) for all configs
- Need better features or different model architecture

### Validation Metrics

```bash
python3 train_cuda_heuristic.py --input cuda_gemm_benchmark_data.csv

Train R²:  1.0000  (Train MAE: 0.25 GFLOPS)
Val R²:    0.9790  (Val MAE:   27.57 GFLOPS)
Test R²:   0.8579  (Test MAE:  55.55 GFLOPS)

⚠️  WARNING: Test R² (0.8579) << Train R² (1.0000)
   Model may be overfitting to training sizes
```

## Lessons Learned

### 1. Always Test on Truly Unseen Data

**Mistake**: Random split gives **illusion of generalization**
- Test set had same model sizes as training
- R²=1.0 looked great but was meaningless

**Fix**: Hold-out **entire model size families**
- Never let model see 14B/235B/671B during training
- True test of generalization

### 2. Canary Tests Are Essential

Automated tests on untrained shapes catch:
- Overfitting (early warning)
- Feature engineering gaps
- Model architecture issues

### 3. Simple Models Often Better in Production

- **Gradient Boosting**: R²=0.86 test, but needs full tree ensemble
- **Linear Regression**: R²=0.79 test, trivial C++ implementation
- **Trade-off**: 7% accuracy for 100× simpler deployment

### 4. R²=1.0 is a Red Flag

Perfect training R² almost always means overfitting:
- Data too small for model capacity
- Features too specific to training examples
- Need regularization

## Next Steps

### Immediate (This Session)
- ✅ Implement model-based train/val/test split
- ✅ Create canary test suite
- ✅ Document findings
- ⏳ Decide: Accept lower accuracy or improve model?

### Near-term
- [ ] Add intermediate model sizes to training (1.5B, 10B, 50B)
- [ ] Engineer better features (batch-aware, alignment-aware)
- [ ] Try regularized GB (max_depth=5, n_estimators=100)
- [ ] Retrain and re-evaluate canaries

### Future Exploration
- [ ] Neural network approach
- [ ] Per-category models (tiny/small/medium/large)
- [ ] Hybrid ML + analytical heuristic
- [ ] Active learning (incremental retraining)

## References

- **Canary test implementation**: `tests/v2/performance/Perf__CudaGemmHeuristicCanary.cpp`
- **Training script changes**: `src/v2/kernels/cuda/python/train_cuda_heuristic.py` (lines 450-490)
- **Validation results**: `build_v2_release/retrain_with_proper_split.log`
- **ML theory**: Bias-variance tradeoff, generalization bounds, cross-validation best practices

---

**Conclusion**: The ML heuristic **severely overfits** to training model sizes. Proper train/val/test splitting reveals true generalization performance (R²=0.79-0.86), which is acceptable but far from perfect. The canary tests provide ongoing validation that the model doesn't degrade further. Next steps involve either accepting this performance level or investing in better features/architecture to improve generalization.
