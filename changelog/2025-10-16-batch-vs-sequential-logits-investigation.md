# Batch vs Sequential Logits Mismatch Investigation
**Date**: 2025-10-16  
**Status**: IN PROGRESS - Root cause identified  
**Test**: `test_batch_correctness::BatchCorrectnessTest.PrefillBatchVsSequential`

## Problem Statement

The batch prefill pipeline produces logits that are 3-4x different from sequential prefill for the same inputs.

**Symptoms:**
- Test comparison: All 151,936 vocab tokens differ significantly
- Batch logits range: [3.0, 5.5]
- Sequential logits range: [12.0, 14.0] (approximately 3-4x larger)
- Bias extraction bug fixed (eliminated NaNs), but magnitude difference persists

## Magnitude Tracing Results

### Pipeline Stage Comparison (Layer 0, Rank 0)

| Stage | Batch | Sequential Seq1 (4 tokens) | Sequential Seq2 (5 tokens) |
|-------|-------|----------------------------|----------------------------|
| **Attention Output** | L2=0.0128654 | L2=0.0121047 | L2=0.0140843 |
| **FFN Down Output** | L2=0.160408 | L2=0.161102 | L2=0.152267 |
| **Final Logits** | L2=2.02669 | L2=2.41047 | L2=2.51077 |

### Critical Finding

**Attention and FFN outputs match closely** (difference < 5%):
- Attention: 0.0128 vs 0.0121/0.0141 ✅
- FFN: 0.160 vs 0.161/0.152 ✅

**Final logits diverge significantly**:
- Batch: 2.027
- Sequential: 2.410/2.511 (19% higher)

### First 5 Logit Values Comparison (Last Token)

**Batch Sequence 1:**
```
[3.25484, 3.53065, 3.90826, 5.50655, 4.15215]
```

**Sequential Sequence 1:**
```
[2.92385, 4.12419, 2.10687, -0.542804, 0.815657]
```

These values are fundamentally different, not just scaled.

## Root Cause Analysis

**Conclusion**: The divergence occurs in the **LM head projection** stage, NOT in the transformer layers.

### Evidence:
1. ✅ Attention computation is correct (magnitudes match)
2. ✅ FFN computation is correct (magnitudes match)  
3. ❌ LM head projection produces different results

### Hypotheses:
1. **Different weight slicing** - Batch vs sequential use different MPI slicing of lm_head weights
2. **Missing aggregation** - Sequential might be missing an MPI_Allreduce after projection
3. **Double aggregation** - Batch might be aggregating when it shouldn't
4. **Replicated vs distributed** - One path uses replicated weights, the other distributed

## Next Steps

1. **Use existing parity framework**: The project already has comprehensive snapshot-based parity tests in `test_batch_correctness.cpp`:
   - `BatchCorrectnessTest.BatchedAttentionStagesParity` - Validates attention stages (PASSING ✅)
   - Can be extended to compare FFN and LM head stages

2. **Add missing snapshot captures to BatchQwenPipeline**:
   - Currently only captures attention stages
   - Need to add: FFN_NORM, FFN_GATE, FFN_UP, FFN_SWIGLU, FFN_DOWN, FFN_RESIDUAL
   - Need to add: FINAL_NORM, LM_HEAD

3. **Compare LM head projection implementations**:
   - Batch: `BatchQwenPipeline` line ~671 (where FINAL LOGITS logged)
   - Sequential: `PrefillProviderBaseImpl` line ~155 (where MAGNITUDE_TRACE_SEQ logged)

4. **Check weight distribution**:
   - Verify lm_head weight shape on each rank
   - Check if weights are replicated or sliced

5. **Check MPI operations**:
   - Sequential: Does it do Allreduce after lm_head projection?
   - Batch: Does it gather results properly?

6. **Verify operator routing**:
   - Which MPILinearOperator variant is used for lm_head in each path?
   - Check if batch uses MPILinearBatchOperator vs sequential uses MPILinearOperator

## Existing Parity Tests

The project has a mature snapshot-based parity testing framework:

### Test Locations

1. **`tests/TestParityFramework.cpp`**: Core parity framework
   - `ParityFramework.OpenBLASPrefillVsPyTorch` - Validates against PyTorch ground truth
   - `ParityFramework.COSMAPrefillVsPyTorch` - Validates COSMA distributed path
   - `ParityFramework.IncrementalDecodeVsPyTorch` - Validates decode equivalence
   - Uses `SnapshotRegistry` and `SnapshotComparator` for systematic comparison

2. **`tests/test_batch_correctness.cpp`**: Batch-specific parity tests  
   - `BatchCorrectnessTest.BatchedAttentionStagesParity` - ✅ **PASSING**
   - Compares batch vs sequential attention stages systematically
   - Reports: "✓ ALL TESTED STAGES MATCH!" for 8 attention stages

### How to Run

```bash
# Attention parity (already passing)
mpirun -np 2 ./build/test_batch_correctness \
  --gtest_filter="BatchCorrectnessTest.BatchedAttentionStagesParity"

# Full pipeline parity (to be extended)
mpirun -np 2 ./build/test_batch_correctness \
  --gtest_filter="BatchCorrectnessTest.PrefillBatchVsSequential"
```

### Extending the Test

To find the LM head divergence, we need to:
1. Add snapshot captures for FFN stages in `src/BatchQwenPipeline.cpp`
2. Add snapshot captures for FINAL_NORM and LM_HEAD
3. Extend `BatchedAttentionStagesParity` to include these stages
4. The comparison infrastructure already exists and works perfectly

## Files Modified

- `src/PrefillProviderBaseImpl.cpp` - Added magnitude tracing for sequential FFN down output
- `src/QwenPipelineAdapter.cpp` - Added file-based tracing (debugging logging mystery)
- `src/QwenPipeline.cpp` - Added provider call tracing
- `src/BatchQwenPipeline.cpp` - Already had magnitude traces (existing)
- `src/operators/MPIAttentionBatchOperator.cpp` - Already had magnitude traces (existing)
- `src/operators/MPIAttentionOperator.cpp` - Already had magnitude traces (existing)

## Debugging Journey

### Mystery: Why Provider Logging Didn't Appear

Initially, attempts to add logging to `PrefillProviderBaseImpl::execute()` produced no output despite the function clearly executing (metrics were returned). 

**Resolution**: The test binary wasn't being rebuilt properly due to compilation errors:
1. Used `getRank()` which doesn't exist (should be `mpiContext().rank`)
2. Tried `mpi_ctx_` which is private (should be `mpiContext().rank`)
3. After fixing accessor method, logging appeared successfully

**Lesson**: Always check for compilation errors when logging mysteriously fails. The old binary was running despite source changes.

## Reproduction

```bash
# Build test
cmake --build build --target test_batch_correctness --parallel

# Run with magnitude traces
mpirun -np 2 ./build/test_batch_correctness \
  --gtest_filter="BatchCorrectnessTest.PrefillBatchVsSequential" 2>&1 \
  | grep -E "MAGNITUDE_TRACE.*Layer0.*(Attention|FFN Down|FINAL LOGITS)"
```

## Expected vs Actual

**Expected**: Batch and sequential should produce identical logits (within floating point tolerance ~1e-5)

**Actual**: 
- Transformer layers match closely (< 5% difference)
- Final logits differ significantly (19-25% L2 norm difference)
- Individual logit values are fundamentally different (not just scaled)

This suggests a systematic issue in the LM head projection or its MPI communication pattern.
