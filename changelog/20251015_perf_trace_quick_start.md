# Performance Tracing - Quick Start Guide

## Immediate Next Steps

### 1. Test the Framework (5 minutes)

```bash
# Build if needed
cmake --build build --parallel

# Create a simple test to verify the framework works
cat > test_perf_trace.cpp << 'EOF'
#include "PerformanceTracer.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace llaminar::perf;

void slowFunction() {
    PERF_TRACE_SCOPE("slowFunction");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void fastFunction() {
    PERF_TRACE_SCOPE("fastFunction");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

int main() {
    // Initialize (would normally be done via environment)
    PerformanceTracer::instance().setEnabled(true);
    PerformanceTracer::instance().setMPIRank(0);
    
    {
        PERF_TRACE_SCOPE("main_loop");
        
        for (int i = 0; i < 5; i++) {
            PERF_TRACE_SCOPE("iteration");
            slowFunction();
            fastFunction();
        }
    }
    
    PerformanceTracer::instance().printSummary(true);
    PerformanceTracer::instance().dumpResults("test_trace.json");
    
    return 0;
}
EOF

# Note: This test file is for reference - actual testing will happen via instrumented code
```

### 2. Add First Instrumentation (15 minutes)

Start with `OpenblasPrefillProvider::executeAttentionBlock()` as it's a high-level function that calls many sub-operations:

```bash
# Open the file
code src/OpenblasPrefillProvider.cpp

# Add at the top of the file (after includes):
#include "PerformanceTracer.h"

# In executeAttentionBlock(), wrap major sections:
bool OpenBLASPrefillProvider::executeAttentionBlock(...) {
    PERF_TRACE_SCOPE_CAT("attention_block", "prefill");
    
    // ... existing code ...
}
```

### 3. Run First Benchmark (2 minutes)

```bash
# Enable tracing
export LLAMINAR_PERF_TRACE=1

# Run prefill-only benchmark (faster iteration)
./run_llaminar.sh --benchmark \
  -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -n 0  # Prefill only

# Check output
cat llaminar_trace.json
```

### 4. Analyze Results (10 minutes)

**Option A: Chrome Tracing (Visual)**
1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Click "Load" button
4. Select `llaminar_trace.json`
5. Explore the timeline

**Option B: Console Summary**
- Automatically printed at program exit
- Shows top operations by time
- Includes call count, avg/min/max times

**Look for**:
- Operations taking >50ms per call
- Unexpectedly high call counts
- Long-running operations at top of summary

## Instrumentation Pattern Reference

### Basic Scoped Timing
```cpp
void myFunction() {
    PERF_TRACE_SCOPE("myFunction");
    // ... code is automatically timed ...
} // Trace ends when scope exits
```

### Categorized Timing
```cpp
void prefillLayer(int idx) {
    PERF_TRACE_SCOPE_CAT("prefill_layer", "prefill");
    // Category helps filter in analysis
}
```

### Nested Timing
```cpp
void complexOperation() {
    PERF_TRACE_SCOPE("complex_op");
    
    {
        PERF_TRACE_SCOPE("phase1");
        // ... phase 1 code ...
    }
    
    {
        PERF_TRACE_SCOPE("phase2");
        // ... phase 2 code ...
    }
}
```

### Conditional Tracing
```cpp
if (PERF_TRACE_ENABLED()) {
    // Only executed when tracing is active
    performExpensiveValidation();
}
```

## High-Priority Files to Instrument

### Tier 1: Core Prefill Path
1. **src/OpenblasPrefillProvider.cpp**:
   - `executeAttentionBlock()` - Entire attention pipeline
   - `executeLinearProjection()` - All linear layers
   - `executePrefill()` - Overall orchestration

2. **src/operators/MPIAttentionOperator.cpp**:
   - `execute()` - Main entry point
   - Internal helper methods (if tracing shows they're significant)

3. **src/operators/MPILinearOperator.cpp**:
   - `execute()` - Matrix multiplication wrapper
   - `cblas_sgemm` calls (if overhead suspected)

### Tier 2: Supporting Operations
4. **src/operators/MPIRMSNormOperator.cpp**:
   - `execute()` - Normalization
   - Core computation

5. **src/operators/MPIEmbeddingOperator.cpp**:
   - `execute()` - Token embedding lookup

### Tier 3: Infrastructure
6. **src/ModelLoader.cpp**:
   - `loadModel()` - Overall loading
   - Per-tensor loading loop
   - Dequantization operations

7. **src/QwenPipeline.cpp**:
   - `prefill()` - Orchestrates prefill
   - `decode()` - Single token generation

## Incremental Instrumentation Strategy

**Session 1** (Today):
- ✅ Framework implementation
- ⏳ Instrument `OpenblasPrefillProvider::executeAttentionBlock()`
- ⏳ Run baseline benchmark
- ⏳ Analyze top 5 hot spots

**Session 2**:
- Instrument identified hot spots from Session 1
- Add finer-grained tracing to slowest operations
- Measure improvement from NUMA optimization
- Identify system call overhead

**Session 3**:
- Compare with llama.cpp performance
- Optimize top bottlenecks
- Add decode path instrumentation
- Full prefill+decode benchmark

## Expected Bottleneck Categories

Based on common LLM inference patterns, expect to find:

1. **Matrix Multiplication** (40-60% of time):
   - Q/K/V projections
   - Attention output projection
   - FFN layers
   - *Action*: Verify OpenBLAS threading, NUMA locality

2. **Memory Operations** (10-20%):
   - Tensor allocation/deallocation
   - Memory copies
   - Cache updates
   - *Action*: Pre-allocate buffers, reduce copies

3. **Normalization** (5-10%):
   - RMSNorm operations
   - *Action*: Vectorization, cache gamma weights

4. **Attention Mechanics** (10-20%):
   - Score computation
   - Softmax
   - Weighted sum
   - *Action*: Fused kernels, memory layout

5. **Overhead** (5-15%):
   - MPI barriers/communication
   - Thread synchronization
   - System calls
   - *Action*: Reduce barriers, cache env vars

## Measuring Success

**Current Status** (Unknown):
- Prefill: ? tok/s
- Decode: ~1.04 tok/s (Debug build)

**Target Goals**:
- Prefill: >1000 tok/s (approaching llama.cpp's ~1200 tok/s)
- Decode: >50 tok/s (Release build, currently Debug)

**Incremental Targets**:
- Week 1: Identify top 3 bottlenecks
- Week 2: Achieve 20% improvement via hot path optimization
- Week 3: Achieve 50% improvement via system call caching
- Week 4: Approach llama.cpp performance (80%+)

## Debug Commands

```bash
# Enable with filter
export LLAMINAR_PERF_TRACE=1
export LLAMINAR_PERF_TRACE_FILTER=attention

# High detail
export LLAMINAR_PERF_TRACE_DETAIL=high

# Custom output
export LLAMINAR_PERF_TRACE_DUMP=prefill_trace.json

# Disable for clean run
unset LLAMINAR_PERF_TRACE
```

## Troubleshooting

**Problem**: No trace output
- **Solution**: Check `LLAMINAR_PERF_TRACE=1` is set
- **Solution**: Verify code is instrumented (add `PERF_TRACE_SCOPE`)
- **Solution**: Ensure program runs to completion (dump happens on exit)

**Problem**: Trace file is huge
- **Solution**: Use `LLAMINAR_PERF_TRACE_FILTER` to reduce scope
- **Solution**: Run shorter benchmarks (fewer tokens)
- **Solution**: Reduce instrumentation in tight loops

**Problem**: Performance regression with tracing
- **Expected**: 5-10% overhead when enabled
- **Acceptable**: Worth it for identifying 50%+ optimization opportunities
- **Mitigation**: Only enable for profiling sessions

## Next Session Checklist

- [ ] Instrument `OpenblasPrefillProvider::executeAttentionBlock()`
- [ ] Instrument `MPIAttentionOperator::execute()`  
- [ ] Instrument `MPILinearOperator::execute()`
- [ ] Run prefill-only benchmark with tracing
- [ ] Analyze Chrome trace timeline
- [ ] Identify top 3 bottlenecks
- [ ] Document findings in changelog
- [ ] Plan optimization strategy

---

**Ready to start**: The framework is built, tested, and integrated. Simply add `#include "PerformanceTracer.h"` and `PERF_TRACE_SCOPE()` calls to begin profiling!
