# Benchmark Mode Implementation

**Date**: 2025-10-15  
**Status**: ✅ Complete and Functional  
**Feature**: Clean inference benchmarking mode with performance metrics

## Overview

Implemented a comprehensive `--benchmark` mode for Llaminar that provides clean, production-quality performance measurement of LLM inference with separate prefill and decode phase timing.

## Implementation Details

### New Files Created

1. **src/BenchmarkRunner.h** (68 lines)
   - `BenchmarkMetrics` struct with prefill/decode/total metrics
   - `runInferenceBenchmark()` function declaration
   - Professional box-drawing formatted output

2. **src/BenchmarkRunner.cpp** (~230 lines)
   - Full benchmark implementation with MPI-aware execution
   - Separate timing for prefill vs decode phases
   - Greedy sampling with MPI broadcast for deterministic results
   - Clean console output with tokens/sec metrics

### Modified Files

1. **src/ArgumentParser.h**
   - Added `bool benchmark_mode` flag

2. **src/ArgumentParser.cpp**
   - Added `--benchmark` argument parsing
   - Sets both `benchmark_mode` and `inference_mode` flags
   - Added help text section

3. **src/Main.cpp**
   - Added benchmark mode execution (section 6)
   - Sets log level to ERROR for clean output
   - Handles MPI rank differences (tokenizer only on rank 0)
   - Uses dummy tokenizer on non-rank-0 processes

4. **CMakeLists.txt**
   - Added `src/BenchmarkRunner.cpp` to source list

## Key Technical Challenges Solved

### 1. Logits Access Pattern
**Problem**: StageContext doesn't contain logits  
**Solution**: Use `pipeline.logits(latest_logits)` to fetch logits tensor after prefill/decode, then extract last row for sampling

### 2. MPI Token Synchronization
**Problem**: Non-rank-0 processes had empty token vectors from dummy tokenizer  
**Solution**: Broadcast token count and tokens from rank 0 to all ranks before prefill (similar to standard inference path)

### 3. Tokenizer Interface Compliance
**Problem**: TokenizerInterface has 8 pure virtual methods  
**Solution**: Implemented complete DummyTokenizer struct for non-rank-0 processes with all required methods

### 4. API Mismatches Fixed
- `TokenizerInterface::tokenize()` not `encode()`
- `TokenizerInterface::detokenize()` not `decode()`
- `getSpecialToken(name)` not `eos_token_id()`
- `AbstractPipeline::prefill/decode` return `bool` and require `StageContext&`
- Logits fetched via `pipeline.logits()` not from StageContext

## Usage

```bash
# Basic benchmark
mpirun -np 2 --bind-to socket --map-by socket ./build/llaminar \
  --benchmark \
  -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "Your prompt here" \
  -n 50

# Longer prefill test
./build/llaminar --benchmark \
  -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "Explain machine learning in simple terms." \
  -n 100
```

## Sample Output

```
Tokenizing prompt... done (8 tokens)
Tokens: [840, 20772, 5662, 6832, 304, 4285, 3793, 13]
Running prefill... done (1216.49 ms, 6.58 tok/s)
Running decode... done (48095.52 ms, 1.04 tok/s)

Generated text:
Machine learning is a type of artificial intelligence that allows computers...

╔══════════════════════════════════════════════════════════════╗
║                    INFERENCE BENCHMARK                       ║
╠══════════════════════════════════════════════════════════════╣
║ Model: models/qwen2.5-0.5b-instruct-q8_0.gguf              ║
║ Backend: OpenBLAS                                          ║
╠══════════════════════════════════════════════════════════════╣
║ PREFILL PHASE                                                ║
║   Tokens:              8 tokens                              ║
║   Time:          1216.49 ms                                 ║
║   Throughput:       6.58 tok/s                             ║
╠══════════════════════════════════════════════════════════════╣
║ DECODE PHASE                                                 ║
║   Tokens:             50 tokens                              ║
║   Time:         48095.52 ms                                 ║
║   Throughput:       1.04 tok/s                             ║
╠══════════════════════════════════════════════════════════════╣
║ TOTAL                                                        ║
║   Tokens:             58 tokens                              ║
║   Time:         49312.01 ms                                 ║
║   Throughput:       1.18 tok/s                             ║
╚══════════════════════════════════════════════════════════════╝
```

## Performance Notes

- Debug build performance: ~1-7 tok/s (varies by phase and prompt length)
- Prefill scales better with more tokens (8 tokens → 6.58 tok/s)
- Decode is more consistent (~1.04 tok/s)
- Release build expected to be significantly faster (needs Fortran compiler resolution for COSMA)

## Future Enhancements

- [ ] Fix Release build configuration (Fortran compiler issue)
- [ ] Add CSV/JSON export option for metrics
- [ ] Support multiple prompts from file
- [ ] Add warmup runs option
- [ ] Include memory usage metrics
- [ ] Add COSMA vs OpenBLAS backend comparison mode

## Testing

Verified working with:
- Model: `qwen2.5-0.5b-instruct-q8_0.gguf` (645MB Q8_0 quantized)
- MPI ranks: 2 (socket binding)
- Prompts: Various lengths from 2 to 8 tokens
- Decode lengths: 20, 50, 100 tokens

## Lessons Learned

1. **MPI debugging is critical**: Segfaults from null tokenizer references on non-rank-0 processes
2. **API exploration matters**: Needed to study ResponseGenerator implementation to understand logits access pattern
3. **Token broadcasting essential**: All ranks must have identical token sequences for distributed inference
4. **Debug assertions help**: The "null data" error pointed us to the empty token vector issue
5. **Small increments work**: Built up from crashes → errors → success through systematic debugging

## Related Documentation

- See `.github/copilot-instructions.md` for canonical runtime configuration
- See `src/chat/ResponseGenerator.cpp` for reference implementation patterns
- See `src/AbstractPipeline.h` for pipeline interface documentation
