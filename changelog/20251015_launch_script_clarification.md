# Launch Script Clarification and Documentation Update

**Date:** October 15, 2025  
**Author:** David Sanftenberg  
**Type:** Documentation Improvement

## Summary

Clarified the distinct purposes of `run_llaminar.sh` and `run_performance_bench.sh` in documentation to prevent confusion. Both scripts serve different audiences and use cases, and should be maintained separately.

## Motivation

During benchmark mode implementation, there was discussion about consolidating the two launch scripts. Investigation revealed they serve fundamentally different purposes:

1. **`run_llaminar.sh`** - Production/user-facing script
2. **`run_performance_bench.sh`** - Development/profiling script

Consolidating them would mix concerns and complicate the canonical launcher.

## Script Comparison

### run_llaminar.sh (Canonical Launcher)

**Purpose:** Production inference and benchmarking with real model execution

**Executable:** `./build/llaminar` (main binary)

**Use Cases:**
- ✅ Production inference workflows
- ✅ `--benchmark` mode for end-to-end performance measurement
- ✅ Model comparison and validation
- ✅ User-facing benchmarks

**Key Features:**
- Automatic CPU topology detection
- OpenMP thread configuration (28 threads/socket on dev system)
- MPI process binding (1 per socket)
- OpenBLAS threading policy (match_omp/hybrid/single)
- Binary argument pass-through via `--` separator
- Comprehensive execution logging

**Example Usage:**
```bash
# Production inference
./run_llaminar.sh -m models/qwen2.5-0.5b-instruct-q8_0.gguf -v

# Benchmark mode
./run_llaminar.sh --benchmark -m models/qwen2.5-0.5b-instruct-q8_0.gguf -p "Test prompt" -n 50

# System info
./run_llaminar.sh -v --print-topology
```

### run_performance_bench.sh (Development Profiling)

**Purpose:** Detailed parallelization efficiency analysis for developers

**Executable:** `./build/test_prefill_performance_bench` (GTest test suite)

**Use Cases:**
- 🔬 Analyzing parallelization efficiency (strong/weak scaling)
- 🔬 Tuning threading strategies
- 🔬 Backend selection optimization research
- 🔬 Profiling COSMA vs OpenBLAS on various matrix shapes

**Key Features:**
- GTest filter support (`--filter` flag)
- Same OpenMP/MPI configuration as canonical launcher
- Efficiency metrics (>90% excellent, <50% poor)
- Focused on prefill performance analysis
- Developer-oriented output with detailed timing breakdowns

**Example Usage:**
```bash
# Run all prefill benchmarks
./run_performance_bench.sh

# Run specific test suites
./run_performance_bench.sh --filter "OpenBLAS_StrongScaling*"
./run_performance_bench.sh --filter "COSMA_ModelShapes*"
```

**Output Format:**
```
[==========] Running 12 tests from 3 test suites.
[ RUN      ] OpenBLAS_StrongScaling.Threads_1
[       OK ] OpenBLAS_StrongScaling.Threads_1 (2345 ms)
...
Efficiency: 92.5% (Excellent)
```

## Decision: Keep Both Scripts

**Recommendation:** Maintain both scripts with clear documentation of their distinct purposes.

**Rationale:**
1. **Different executables** - Main binary vs GTest suite
2. **Different interfaces** - Binary args vs GTest filters
3. **Different audiences** - Users/benchmarking vs developers/profiling
4. **Minimal duplication** - Both need OpenMP/MPI config (unavoidable)
5. **Clear separation of concerns** - Production vs development tooling

## Documentation Updates

### README.md

Added new section "Development Profiling (Advanced)" after the Benchmark Mode section:

```markdown
## Development Profiling (Advanced)

For developers analyzing parallelization efficiency and conducting detailed 
prefill performance studies, Llaminar provides a separate GTest-based 
profiling suite:

# [usage examples]

**Note:** This is distinct from `--benchmark` mode:
- **`run_llaminar.sh --benchmark`**: Production inference benchmarking
- **`run_performance_bench.sh`**: Development profiling with efficiency analysis
```

### .github/copilot-instructions.md

Added section "Development Profiling (Advanced)" with clear distinction:

```markdown
**Important Distinction:**
- **`run_llaminar.sh --benchmark`**: Production inference benchmarking
  - Use for: End-to-end measurement, model comparisons, user benchmarks
  - Outputs: Clean tok/s metrics for prefill/decode
  
- **`run_performance_bench.sh`**: Development profiling with GTest suite
  - Use for: Parallelization analysis, threading tuning, optimization
  - Outputs: Detailed GTest metrics with efficiency percentages
```

## Alternative Considered: Script Consolidation

**Option:** Enhance `run_llaminar.sh` to auto-detect test binaries

```bash
# Pseudo-code for consolidated approach
if [[ "$1" == *"test_"* ]]; then
    # Test binary mode - support --gtest_filter
else
    # Normal llaminar binary
fi
```

**Rejected Because:**
- Mixes production and development concerns
- Complicates the canonical launcher documentation
- Test scripts are for developers, not end users
- Maintenance burden of unified interface
- Would require significant refactoring of both scripts

## Future Considerations

### Potential Enhancements

1. **Unified configuration module** - Extract OpenMP/MPI config to shared library
2. **Task integration** - Add VS Code tasks for performance profiling
3. **Benchmark suite expansion** - More GTest-based profiling tests
4. **Documentation cross-links** - Better navigation between script docs

### When to Reconsider Consolidation

Only if:
- User-facing need emerges for GTest-style filtering in production
- Scripts diverge significantly in configuration (currently aligned)
- Maintenance burden of two scripts becomes problematic

## Testing

**Verified:**
- ✅ `run_llaminar.sh --benchmark` works correctly (26.86 tok/s prefill)
- ✅ `run_performance_bench.sh` exists and has GTest executable
- ✅ Both scripts have similar OpenMP/MPI configuration
- ✅ Documentation updates render correctly
- ✅ No references to deprecated/consolidated scripts

## Impact

**User Impact:** Minimal - clarifies existing tooling  
**Developer Impact:** Positive - clear guidance on which script to use  
**Documentation:** Comprehensive updates to README and copilot-instructions  
**Code Changes:** None - documentation only  

## Related Files

- `README.md` - Added Development Profiling section
- `.github/copilot-instructions.md` - Added script distinction section
- `run_llaminar.sh` - No changes (canonical launcher)
- `run_performance_bench.sh` - No changes (development profiling)

## Conclusion

Keeping both scripts with clear documentation provides the best user experience:
- Users know to always use `run_llaminar.sh` for production work
- Developers know to use `run_performance_bench.sh` for profiling
- Clear separation prevents confusion and maintains clean architecture
- Minimal duplication is acceptable for clear separation of concerns

The documentation now clearly explains when and why to use each script, eliminating potential confusion without adding complexity to either tool.
