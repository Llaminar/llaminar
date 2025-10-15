# Decode-Only and Prefill-Only Benchmark Modes

**Date:** 2025-10-15  
**Author:** David Sanftenberg  
**Category:** Feature Enhancement  
**Status:** Implemented

## Summary

Added support for decode-only and prefill-only benchmarks by allowing users to set either phase to 0 tokens, enabling isolated performance measurement of each inference phase.

## Motivation

When optimizing inference performance, it's valuable to measure prefill and decode phases independently:

- **Prefill-only:** Test large batch/sequence processing performance without decode overhead
- **Decode-only:** Measure single-token generation latency without prefill initialization
- **Isolation:** Identify which phase is the bottleneck for specific workloads
- **A/B Testing:** Compare optimization impact on individual phases

## Usage

### Decode-Only Benchmark

Skip prefill by providing an empty prompt (or use `-p ""`):

```bash
# Decode-only: 0 prefill tokens, 128 decode tokens
./run_llaminar.sh -- --benchmark -m model.gguf -p "" -n 128
```

**Use Case:** Measure pure decode performance (single-token generation latency)

### Prefill-Only Benchmark

Skip decode by setting `-n 0`:

```bash
# Prefill-only: ~512 prefill tokens, 0 decode tokens
./run_llaminar.sh -- --benchmark -m model.gguf -n 0

# Custom prefill size
./run_llaminar.sh -- --benchmark -m model.gguf \
  -p "Your custom prompt here" -n 0
```

**Use Case:** Measure pure prefill performance (batch processing throughput)

### Standard Benchmark (Both Phases)

Default behavior unchanged:

```bash
# Both phases: ~512 prefill, 128 decode
./run_llaminar.sh -- --benchmark -m model.gguf
```

## Implementation Details

### Changes to BenchmarkRunner

**File:** `src/BenchmarkRunner.cpp`

#### 1. Conditional Prefill Execution

```cpp
// Skip prefill if token_count == 0
if (token_count > 0)
{
    auto prefill_start = std::chrono::high_resolution_clock::now();
    if (!pipeline.prefill(tokens, weights, stage_ctx))
    {
        // Handle error...
    }
    auto prefill_end = std::chrono::high_resolution_clock::now();
    // Calculate metrics...
}
else
{
    std::cout << " skipped (0 tokens)\n";
}
```

#### 2. Conditional Decode Execution

```cpp
// Skip decode if n_predict == 0
if (max_new_tokens == 0)
{
    std::cout << " skipped (0 tokens requested)\n";
    
    // Finalize metrics for prefill-only
    metrics.decode_tokens = 0;
    metrics.decode_time_ms = 0.0;
    metrics.decode_tokens_per_sec = 0.0;
    metrics.total_time_ms = metrics.prefill_time_ms;
    metrics.total_tokens_per_sec = metrics.prefill_tokens_per_sec;
    
    return metrics;
}
```

#### 3. Conditional Metrics Display

```cpp
// Only show prefill section if tokens > 0
if (prefill_tokens > 0)
{
    std::cout << "╠══════════════╣\n";
    std::cout << "║ PREFILL PHASE ║\n";
    // ... metrics ...
}
else
{
    std::cout << "║ PREFILL PHASE (SKIPPED) ║\n";
}

// Only show decode section if tokens > 0
if (decode_tokens > 0)
{
    std::cout << "╠══════════════╣\n";
    std::cout << "║ DECODE PHASE  ║\n";
    // ... metrics ...
}
else
{
    std::cout << "║ DECODE PHASE (SKIPPED) ║\n";
}

// Only show total if both phases ran
if (prefill_tokens > 0 && decode_tokens > 0)
{
    std::cout << "║ TOTAL ║\n";
    // ... combined metrics ...
}
```

## Example Outputs

### Prefill-Only Benchmark

```bash
$ ./run_llaminar.sh -- --benchmark -m models/qwen2.5-0.5b-instruct-q8_0.gguf -n 0

Tokenizing prompt... done (487 tokens)
Tokens: [791, 4996, 17354, 39935, 35308, 916, 279, 26016, 5679, 11, ...]
Running prefill... done (2143.67 ms, 227.18 tok/s)
Running decode... skipped (0 tokens requested)

╔══════════════════════════════════════════════════════════════╗
║                    INFERENCE BENCHMARK                       ║
╠══════════════════════════════════════════════════════════════╣
║ Model: models/qwen2.5-0.5b-instruct-q8_0.gguf               ║
║ Backend: OpenBLAS                                            ║
╠══════════════════════════════════════════════════════════════╣
║ PREFILL PHASE                                                ║
║   Tokens:            487 tokens                              ║
║   Time:          2143.67 ms                                 ║
║   Throughput:     227.18 tok/s                             ║
╠══════════════════════════════════════════════════════════════╣
║ DECODE PHASE                                    (SKIPPED)    ║
╚══════════════════════════════════════════════════════════════╝
```

### Decode-Only Benchmark

```bash
$ ./run_llaminar.sh -- --benchmark -m models/qwen2.5-0.5b-instruct-q8_0.gguf -p "" -n 50

Tokenizing prompt... skipped (0 tokens)
Running decode... done (5234.21 ms, 9.55 tok/s)

╔══════════════════════════════════════════════════════════════╗
║                    INFERENCE BENCHMARK                       ║
╠══════════════════════════════════════════════════════════════╣
║ Model: models/qwen2.5-0.5b-instruct-q8_0.gguf               ║
║ Backend: OpenBLAS                                            ║
╠══════════════════════════════════════════════════════════════╣
║ PREFILL PHASE                                   (SKIPPED)    ║
╠══════════════════════════════════════════════════════════════╣
║ DECODE PHASE                                                 ║
║   Tokens:             50 tokens                              ║
║   Time:          5234.21 ms                                 ║
║   Throughput:       9.55 tok/s                             ║
╚══════════════════════════════════════════════════════════════╝
```

### Standard Benchmark (Both Phases)

```bash
$ ./run_llaminar.sh -- --benchmark -m models/qwen2.5-0.5b-instruct-q8_0.gguf

Tokenizing prompt... done (487 tokens)
Running prefill... done (2143.67 ms, 227.18 tok/s)
Running decode... done (12845.32 ms, 9.97 tok/s)

╔══════════════════════════════════════════════════════════════╗
║                    INFERENCE BENCHMARK                       ║
╠══════════════════════════════════════════════════════════════╣
║ Model: models/qwen2.5-0.5b-instruct-q8_0.gguf               ║
║ Backend: OpenBLAS                                            ║
╠══════════════════════════════════════════════════════════════╣
║ PREFILL PHASE                                                ║
║   Tokens:            487 tokens                              ║
║   Time:          2143.67 ms                                 ║
║   Throughput:     227.18 tok/s                             ║
╠══════════════════════════════════════════════════════════════╣
║ DECODE PHASE                                                 ║
║   Tokens:            128 tokens                              ║
║   Time:         12845.32 ms                                 ║
║   Throughput:       9.97 tok/s                             ║
╠══════════════════════════════════════════════════════════════╣
║ TOTAL                                                        ║
║   Tokens:            615 tokens                              ║
║   Time:         14989.00 ms                                 ║
║   Throughput:      41.03 tok/s                             ║
╚══════════════════════════════════════════════════════════════╝
```

## Use Cases

### 1. Prefill Performance Testing

Useful for:
- Testing COSMA vs OpenBLAS prefill performance
- Measuring NUMA optimization impact on large batches
- Validating tensor parallelism scaling
- Long-context performance benchmarking

```bash
# Test prefill with different sizes
./run_llaminar.sh -- --benchmark -m model.gguf \
  -p "$(python -c 'print(\"test \" * 128)')" -n 0  # 128 tokens

./run_llaminar.sh -- --benchmark -m model.gguf \
  -p "$(python -c 'print(\"test \" * 512)')" -n 0  # 512 tokens

./run_llaminar.sh -- --benchmark -m model.gguf \
  -p "$(python -c 'print(\"test \" * 2048)')" -n 0 # 2048 tokens
```

### 2. Decode Performance Testing

Useful for:
- Measuring single-token generation latency
- Testing KV cache performance
- Validating incremental decode optimizations
- Comparing decode backends

```bash
# Test decode with different generation lengths
./run_llaminar.sh -- --benchmark -m model.gguf -p "" -n 10   # Short
./run_llaminar.sh -- --benchmark -m model.gguf -p "" -n 50   # Medium
./run_llaminar.sh -- --benchmark -m model.gguf -p "" -n 200  # Long
```

### 3. A/B Testing Optimizations

```bash
# Baseline prefill (NUMA disabled)
export LLAMINAR_NUMA_FIRST_TOUCH=0
./run_llaminar.sh -- --benchmark -m model.gguf -n 0

# Optimized prefill (NUMA enabled)
unset LLAMINAR_NUMA_FIRST_TOUCH
./run_llaminar.sh -- --benchmark -m model.gguf -n 0

# Compare improvement
```

## Technical Notes

### Prefill-Only Limitations

When running prefill-only (`-n 0`), the benchmark:
1. Runs the prefill phase normally
2. Skips the decode loop entirely
3. Returns early with prefill metrics only
4. Does not generate any tokens

**Note:** Some pipeline implementations may require at least one decode step for proper initialization. If you encounter errors, try `-n 1` instead.

### Decode-Only Limitations

When running decode-only (`-p ""`), the benchmark:
1. Skips tokenization (0 tokens)
2. Skips the prefill phase
3. Runs decode starting from an empty KV cache

**Important:** Decode-only mode may not work correctly on all pipelines, as some expect prefill to initialize the KV cache. This mode is primarily useful for testing decode performance in isolation when the pipeline supports it.

### MPI Compatibility

Both modes work correctly with MPI:
- Token counts and skip flags are broadcast to all ranks
- All ranks execute (or skip) phases synchronously
- Metrics are collected from rank 0 only

## Backward Compatibility

✅ **Fully backward compatible** - existing benchmark usage unchanged:

```bash
# These all still work as before
./run_llaminar.sh -- --benchmark -m model.gguf
./run_llaminar.sh -- --benchmark -m model.gguf -p "prompt" -n 128
```

## Future Enhancements

Possible improvements:
1. **Warmup phase:** Skip first run for cache/JIT warmup
2. **Multiple iterations:** Average over N runs for stability
3. **Statistical analysis:** Report mean/stddev/percentiles
4. **Phase-specific backends:** Allow COSMA for prefill, OpenBLAS for decode

## Related Work

- **20251015_benchmark_mode_implementation.md** - Original benchmark infrastructure
- **20251015_benchmark_default_parameters.md** - Default parameter settings
- **20251015_simpletensor_numa_optimization.md** - NUMA optimization (testable with prefill-only)

## Conclusion

The benchmark mode now supports isolated testing of prefill and decode phases by setting either to 0 tokens. This enables:

- **Targeted optimization:** Measure individual phase improvements
- **Performance analysis:** Identify bottlenecks
- **Flexible testing:** Custom workload combinations
- **Clean output:** Skipped phases clearly marked

Use `-n 0` for prefill-only or `-p ""` for decode-only benchmarks.

---

**Status:** ✅ Implemented and ready for use
