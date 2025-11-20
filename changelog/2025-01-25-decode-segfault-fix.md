# V2 Decode Segfault Fix - Null Pointer Dereference

**Date:** 2025-01-25  
**Status:** ✅ FIXED - Decode now works  
**Milestone:** First successful V2 decode phase with token generation

## Problem

After successfully implementing V2 first prefill, the decode phase was segfaulting with:
```
Signal: Segmentation fault (11)
Signal code: Address not mapped (1)
Failing at address: (nil)
```

### Root Cause

**Null pointer dereference in `PipelineBase::logits()`:**

1. `Qwen2Pipeline` has a `getLogits(int seq_idx)` method that returns logits from `logits_buffer_`
2. `Main.cpp` calls `pipeline->logits()` which is defined in `PipelineBase`
3. `PipelineBase::logits()` tries to access `logits_` member (base class field)
4. **`logits_` was never set by Qwen2Pipeline** (uses `logits_buffer_` instead)
5. Segfault when dereferencing `nullptr`

### Debug Process

Added comprehensive LOG_DEBUG statements to Main.cpp decode loop:
```cpp
LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Starting decode iteration " << i);
LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Getting logits...");
const float *logits = pipeline->logits();  // <-- SEGFAULT HERE
LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Logits retrieved");
```

Output showed:
```
[18:17:03.493] [DEBUG] [Main.cpp:510] [Rank 0] Starting decode iteration 0
[18:17:03.493] [DEBUG] [Main.cpp:513] [Rank 0] Getting logits...
[61201af66063:2208577] *** Process received signal ***
[61201af66063:2208577] Signal: Segmentation fault (11)
```

This revealed the crash was in `pipeline->logits()` call.

## Solution

**Override `logits()` in Qwen2Pipeline to delegate to `getLogits(0)`:**

### File: `src/v2/pipelines/qwen/Qwen2Pipeline.h`

```cpp
// PipelineBase interface
bool forward(const int *tokens, int seq_len) override;
const char *architecture() const override { return "qwen2"; }
const float *logits() const override { return getLogits(0); }  // ← NEW: Override to use Qwen2's logits_buffer_

/**
 * @brief Get output logits for E2E testing/validation
 *
 * @param seq_idx Sequence index in batch (default=0)
 * @return Logits tensor [padded_seq_len, vocab_size], or nullptr if forward() not called
 */
const float *getLogits(int seq_idx = 0) const;
```

### Why This Works

- **Before:** `PipelineBase::logits()` → dereferences `logits_` (nullptr) → segfault
- **After:** `Qwen2Pipeline::logits()` → calls `getLogits(0)` → returns `logits_buffer_->data() + offset` → valid pointer

## Test Results

✅ **Decode now completes successfully:**
```bash
$ ./run_llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "Hi" -n 5
Hi IORæī§æķĻĠindoorĠâģĠØŃØ§ÙĦ
```

✅ **No segfault** - process runs to completion  
✅ **Tokens generated** - 5 tokens successfully sampled and output  
⚠️ **Token quality issue** - output is garbled (separate issue, likely tokenizer decoding)

## Performance Notes

Decode is **very slow** (~6-7 seconds per token in Release build):
- Expected: ~1-2 tok/s for small models on CPU
- Actual: ~0.15 tok/s
- Likely causes:
  - No KV cache optimization
  - Suboptimal attention implementation
  - Missing kernel optimizations
  - Potential MPI overhead in tensor-parallel mode

## Related Work

**Unit Tests for Device Index Bug:**
- Created `tests/v2/unit/Test__DeviceIndexCPUKernels.cpp` (11 tests, all passing)
- Prevents regression of device_idx=0 CPU kernel bug
- Commit: `5203fd2`

**Debug Logging:**
- Added comprehensive LOG_DEBUG throughout Main.cpp decode loop
- Logging revealed exact segfault location (logits() call)
- Can be kept for future debugging or removed if verbose

## Next Steps

### Immediate Priorities

1. **Fix Token Decoding** - Investigate why tokens are garbled
   - Check tokenizer decode logic
   - Verify vocab_size matches model
   - Test with simpler prompts

2. **Performance Optimization** - Improve decode speed
   - Profile with perf/vtune to identify bottlenecks
   - Optimize attention for single-token decode
   - Consider KV cache implementation

3. **Clean Up Debug Logging** - Remove or gate behind flag
   - Main.cpp has 9 LOG_DEBUG statements
   - Consider LLAMINAR_DEBUG_DECODE flag

### Medium-Term

4. **E2E Testing** - Validate decode correctness
   - Compare per-layer activations against PyTorch reference
   - Use snapshot framework for decode phase
   - Create decode-specific parity tests

5. **Batched Decode** - Support multiple sequences
   - Test with batch_size > 1
   - Verify attention masking works correctly
   - Profile batch scaling performance

## Files Modified

- `src/v2/pipelines/qwen/Qwen2Pipeline.h` - Added `logits()` override
- `src/v2/Main.cpp` - Added debug logging (can be removed)

## Commands

```bash
# Rebuild with fix
cmake --build build_v2_release --target llaminar2 --parallel

# Test decode (works but tokens garbled)
./run_llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "Hi" -n 5

# Test with longer generation
./run_llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "The capital of France is" -n 10
```

## Lessons Learned

1. **Virtual Override Pattern:**
   - When subclass uses different member names (logits_buffer_ vs logits_), must override interface method
   - Base class can't magically know about subclass-specific fields

2. **Debug Logging Strategy:**
   - Granular logging at each step of loop reveals exact failure point
   - MPI-aware logging (with rank) helps identify sync vs async issues
   - Start with ERROR level, add DEBUG when needed

3. **Null Pointer Debugging:**
   - Segfault at (nil) = null pointer dereference
   - Check all pointer returns from virtual methods
   - Ensure subclass properly implements/overrides interface

## Architecture Note

This fix highlights a design pattern in V2:
- **PipelineBase** provides minimal interface (`logits()`)
- **Qwen2Pipeline** has richer API (`getLogits(int seq_idx)` for batching)
- Must bridge the gap with override that delegates to subclass-specific method

Alternative approaches considered:
1. Change Main.cpp to cast to Qwen2Pipeline* and use getLogits() - **too specific**
2. Move logits_buffer_ to base class - **breaks per-pipeline flexibility**
3. Make logits() pure virtual - **forces all pipelines to implement, even if not needed**

Current solution (override with delegation) is cleanest and maintains flexibility.
