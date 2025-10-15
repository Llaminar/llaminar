# Performance Tracing Framework

**Date**: 2025-10-15  
**Author**: David Sanftenberg  
**Status**: Initial implementation

## Overview

Introduced a hierarchical, zero-overhead performance tracing framework to identify bottlenecks in prefill and decode hot paths. llama.cpp achieves ~1200 tok/s on Q8 models while Llaminar shows significantly lower performance, suggesting system call overhead or other inefficiencies in critical paths.

## Design Goals

1. **Zero Overhead When Disabled**: Complete compile-time elimination via macros
2. **Hierarchical Tracing**: Track nested operations with proper call stacks
3. **Thread-Safe**: Handle OpenMP parallel regions correctly
4. **MPI-Aware**: Aggregate statistics across ranks
5. **Selective Activation**: Filter specific operations via environment variables
6. **Chrome Trace Format**: Export for visualization in chrome://tracing

## Architecture

### Core Components

**PerformanceTracer.h**:
- `PerformanceTracer` singleton: Main coordinating class
- `TraceEvent`: Individual timing event with metadata
- `ThreadTraceStack`: Per-thread hierarchical stack
- `TraceScope`: RAII-based automatic timing

**PerformanceTracer.cpp**:
- Event collection and aggregation
- JSON export (Chrome trace format)
- Summary statistics reporting
- Thread-local storage management

**Integration with DebugEnv**:
- `PerfTraceEnv` struct in `src/utils/DebugEnv.h`
- Environment variable parsing in `src/utils/DebugEnv.cpp`
- Centralized configuration access via `debugEnv().perf`

### Usage Patterns

**Scoped Tracing (Recommended)**:
```cpp
void myFunction() {
    PERF_TRACE_SCOPE("myFunction");
    // ... code to trace ...
} // Automatically ends trace
```

**Categorized Tracing**:
```cpp
void prefillLayer(int layer_idx) {
    PERF_TRACE_SCOPE_CAT("prefill_layer", "prefill");
    // ... prefill logic ...
}
```

**Manual Begin/End** (for complex control flow):
```cpp
if (condition) {
    PERF_TRACE_BEGIN("conditional_path");
}
// ... code ...
if (condition) {
    PERF_TRACE_END("conditional_path");
}
```

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `LLAMINAR_PERF_TRACE=1` | Enable tracing | Required to activate |
| `LLAMINAR_PERF_TRACE_DETAIL=high` | Detail level | `low`, `medium`, `high` |
| `LLAMINAR_PERF_TRACE_FILTER=prefill` | Filter operations | Only trace matching names |
| `LLAMINAR_PERF_TRACE_DUMP=trace.json` | Output file | Chrome trace format |
| `LLAMINAR_PERF_TRACE_AUTO_DUMP=0` | Disable auto dump on exit | Default: enabled |

## Implementation Status

### Completed
- ✅ Core framework (`PerformanceTracer.h/cpp`)
- ✅ DebugEnv integration (`PerfTraceEnv` struct)
- ✅ Environment variable parsing
- ✅ Chrome trace format export
- ✅ Hierarchical timing support
- ✅ Thread-local stacks
- ✅ Summary statistics

### Pending
- ⏳ Instrumentation of hot paths (next step)
- ⏳ CMake integration (add to build system)
- ⏳ MPI aggregation (currently per-rank traces)
- ⏳ Benchmark runner integration
- ⏳ Testing and validation

## Next Steps

### 1. Add to CMake Build System
```cmake
# Add to CMakeLists.txt
target_sources(llaminar_core PRIVATE
    src/PerformanceTracer.cpp
)
```

### 2. Instrument Hot Paths

**Priority 1: Prefill Path**
- `OpenblasPrefillProvider::executeAttentionBlock()`
- `OpenblasPrefillProvider::executeLinearProjection()`
- `MPIAttentionOperator::execute()` (prefill branch)
- `MPILinearOperator::execute()`
- `MPIRMSNormOperator::execute()`

**Priority 2: Decode Path**
- `QwenPipeline::decode()`
- `MPIAttentionOperator::execute()` (decode branch)
- K/V cache operations
- Token sampling

**Priority 3: Infrastructure**
- Model loading (`ModelLoader::loadModel()`)
- Tokenization
- MPI communication (Allreduce, Allgatherv)

### 3. Example Instrumentation

**Before**:
```cpp
bool OpenBLASPrefillProvider::executeAttentionBlock(...) {
    // Attention logic
    return true;
}
```

**After**:
```cpp
bool OpenBLASPrefillProvider::executeAttentionBlock(...) {
    PERF_TRACE_SCOPE_CAT("attention_block", "prefill");
    
    {
        PERF_TRACE_SCOPE("rmsnorm");
        // RMSNorm execution
    }
    
    {
        PERF_TRACE_SCOPE("qkv_projection");
        // Q/K/V projections
    }
    
    {
        PERF_TRACE_SCOPE("attention_scores");
        // Score computation
    }
    
    {
        PERF_TRACE_SCOPE("output_projection");
        // Output projection
    }
    
    return true;
}
```

## Expected Benefits

### Performance Insights
- Identify specific functions consuming most time
- Compare prefill vs decode bottlenecks
- Detect unexpected synchronization points
- Find system call overhead (getenv, malloc, etc.)

### Comparison with llama.cpp
- Target: Achieve >1000 tok/s prefill performance
- Current baseline: To be measured after instrumentation
- Focus areas: Matrix operations, memory allocation, MPI overhead

### Optimization Opportunities
- Cache expensive system calls
- Reduce memory allocations in hot paths
- Optimize thread synchronization
- Minimize MPI barrier overhead

## Output Format Examples

### Console Summary
```
╔══════════════════════════════════════════════════════════════════════════╗
║                    PERFORMANCE TRACE SUMMARY                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║ Operation                      Calls  Total (ms)   Avg (ms)  Min (μs)  Max (μs) ║
╠══════════════════════════════════════════════════════════════════════════╣
║ prefill_layer                     24     8432.12     351.34    342156    358924 ║
║ attention_block                   24     6234.45     259.77    251234    267890 ║
║ qkv_projection                    24     3123.23     130.13    125678    134567 ║
║ attention_scores                  24     2456.78     102.37     98765    105432 ║
║ output_projection                 24      654.44      27.27     25123     29876 ║
║ rmsnorm                           48      564.32      11.76     10234     13456 ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### Chrome Trace Format
```json
{
  "traceEvents": [
    {
      "name": "prefill_layer",
      "cat": "prefill",
      "ph": "X",
      "ts": 1729012345678901,
      "dur": 351340,
      "tid": 12345,
      "pid": 0,
      "args": { "depth": 0 }
    },
    ...
  ]
}
```

View in Chrome: `chrome://tracing` → Load JSON file

## Compile-Time Control

The framework is disabled by default. To enable:

**Option 1: CMake Flag**
```bash
cmake -B build -S . -DLLAMINAR_ENABLE_PERF_TRACE=ON
```

**Option 2: Manual Definition**
```cpp
// In CMakeLists.txt or compiler flags
add_compile_definitions(LLAMINAR_ENABLE_PERF_TRACE)
```

When disabled, all `PERF_TRACE_*` macros expand to no-ops and are completely eliminated by the compiler (zero runtime cost).

## Testing Strategy

1. **Smoke Test**: Verify framework activation
   ```bash
   export LLAMINAR_PERF_TRACE=1
   ./build/llaminar --benchmark -m model.gguf -n 10
   # Should produce trace.json
   ```

2. **Nested Operations**: Verify hierarchical tracking
   - Check depth values in trace events
   - Validate parent-child relationships
   
3. **Thread Safety**: Run with OpenMP parallelism
   - Verify per-thread stacks don't interfere
   - Check aggregated statistics are correct

4. **MPI Awareness**: Test multi-rank execution
   - Verify per-rank trace files
   - Validate MPI rank attribution

## Known Limitations

1. **No Dynamic Enable/Disable**: Must be compile-time enabled
2. **No Real-Time Monitoring**: Results only available at program exit or explicit dump
3. **Memory Overhead**: Stores all events in memory (may be significant for long runs)
4. **Single-Machine MPI**: Per-rank traces, not aggregated to rank 0 yet

## Future Enhancements

1. **Ring Buffer Mode**: Limit memory usage for long-running processes
2. **Live Monitoring**: HTTP endpoint for real-time trace viewing
3. **Statistical Sampling**: Sample 1-in-N calls to reduce overhead
4. **NVTX Integration**: For NVIDIA Nsight profiling
5. **Flame Graph Export**: Direct export to flamegraph.pl format

## References

- Chrome Trace Event Format: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/
- NVIDIA NVTX: https://nvidia.github.io/NVTX/doxygen-cpp/index.html
- Perfetto: https://perfetto.dev/

## Changelog

### 2025-10-15
- Initial framework implementation
- DebugEnv integration
- Zero-overhead macro design
- Chrome trace format export
- Hierarchical timing support
