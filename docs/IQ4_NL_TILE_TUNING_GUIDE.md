# IQ4_NL GEMM Tile Size Tuning Guide

## Overview

The IQ4_NL fused GEMM implementation uses adaptive cache tiling to optimize performance across different workload characteristics. Tile sizes can be manually overridden for experimentation and tuning using environment variables.

## Quick Start

```bash
# Run with custom tile sizes (microkernel enabled)
LLAMINAR_IQ4_GEMM_MICROKERNEL=1 \
LLAMINAR_IQ4_M_TILE=64 \
LLAMINAR_IQ4_N_TILE=64 \
./run_benchmark.sh benchmark_iq4nl_gemm

# Quick sweep across key tile sizes (~10 configs, 4 workloads each)
./benchmark_tile_sweep_quick.sh

# Full sweep across all tile combinations (~81 configs, 10 workloads each)
# WARNING: Takes ~1-2 hours!
./benchmark_tile_sweep.sh
```

## Environment Variables

### Tile Size Overrides

#### FP32 Activation Path
- **`LLAMINAR_IQ4_M_TILE`**: Override M dimension tile size for FP32 path (rows per tile)
  - Default: Adaptive (varies by workload: 32-128)
  - Valid range: 16-256
  - When set: Overrides adaptive selection for FP32 multiply()

- **`LLAMINAR_IQ4_N_TILE`**: Override N dimension tile size for FP32 path (columns per tile)
  - Default: Adaptive (varies by workload: 24-128)
  - Valid range: 16-256
  - When set: Overrides adaptive selection for FP32 multiply()

**Important**: Both M_TILE and N_TILE must be set together for FP32 override to activate.

#### BF16 Activation Path
- **`LLAMINAR_IQ4_M_TILE_BF16`**: Override M dimension tile size for BF16 path (rows per tile)
  - Default: 64 (fixed)
  - Valid range: 16-256
  - When set: Overrides default for BF16 multiply_bf16()

- **`LLAMINAR_IQ4_N_TILE_BF16`**: Override N dimension tile size for BF16 path (columns per tile)
  - Default: 64 (fixed)
  - Valid range: 16-256
  - When set: Overrides default for BF16 multiply_bf16()

**Important**: Both M_TILE_BF16 and N_TILE_BF16 must be set together for BF16 override to activate.

### Microkernel Toggle

- **`LLAMINAR_IQ4_GEMM_MICROKERNEL`**: Enable multi-column vectorized decode
  - Default: 0 (disabled)
  - Set to 1: Decode 4 columns at once (reduces loop overhead)
  - Performance impact: +35% on FFN Batch 16, +7-9% on Q-proj 1024/2048

## Adaptive Tiling Strategy (Default)

The implementation uses aspect ratio analysis to select optimal tile sizes:

### Square Matrices (aspect ratio 0.5-2.0)
**Example**: Q-Projection (896×896)
**Strategy**: Balanced tiling for L2 cache fit

| m (batch size) | M_TILE | N_TILE | Total Working Set |
|----------------|--------|--------|-------------------|
| ≥4096          | 32     | 32     | ~114 KB           |
| ≥2048          | 48     | 48     | ~172 KB           |
| ≥1024          | 64     | 64     | ~229 KB           |
| ≥512           | 96     | 96     | ~344 KB           |
| <512           | 128    | 128    | ~458 KB           |

### Wide Matrices (aspect ratio >2.0)
**Example**: FFN (896→4864)
**Strategy**: Small N_TILE (reduce decode buffer), large M_TILE (maximize reuse)

| m (batch size) | M_TILE | N_TILE | Decode Buffer |
|----------------|--------|--------|---------------|
| ≥4096          | 32     | 24     | ~85 KB        |
| ≥2048          | 64     | 24     | ~85 KB        |
| ≥1024          | 96     | 32     | ~114 KB       |
| <1024          | 128    | 48     | ~172 KB       |

### Tall Matrices (aspect ratio <0.5)
**Strategy**: Very small N_TILE, maximum M_TILE

| m (batch size) | M_TILE | N_TILE |
|----------------|--------|--------|
| ≥4096          | 64     | 24     |
| ≥2048          | 96     | 32     |
| <2048          | 128    | 48     |

## Benchmarking Scripts

### FP32 Quick Sweep (Recommended for Initial Tuning)
```bash
./benchmark_tile_sweep_quick.sh
```
- **Duration**: ~15-20 minutes
- **Path**: FP32 activation only
- **Configurations**: 9 tile settings
- **Workloads**: 4 key shapes (Q-Proj 1024/4096, FFN Batch 16/256)
- **Output**: CSV file in `tile_sweep_results/quick_sweep_TIMESTAMP.csv`

### BF16 Quick Sweep
```bash
./benchmark_tile_sweep_bf16.sh
```
- **Duration**: ~15-20 minutes
- **Path**: BF16 activation only
- **Configurations**: 9 tile settings
- **Workloads**: 4 key shapes (Q-Proj 1024/4096, FFN Batch 16/256)
- **Output**: CSV file in `tile_sweep_results/bf16_sweep_TIMESTAMP.csv`

### Combined FP32 + BF16 Sweep (Recommended)
```bash
./benchmark_tile_sweep_combined.sh
```
- **Duration**: ~15-20 minutes (same workload, extracts both paths)
- **Path**: Both FP32 and BF16 measured simultaneously
- **Configurations**: 9 tile settings
- **Workloads**: 4 key shapes
- **Output**: Two CSV files:
  - `tile_sweep_results/combined_fp32_TIMESTAMP.csv`
  - `tile_sweep_results/combined_bf16_TIMESTAMP.csv`
- **Advantage**: Directly compare FP32 vs BF16 with identical tile settings

### Full Sweep (Comprehensive Analysis)
```bash
./benchmark_tile_sweep.sh  # FP32 only
```
- **Duration**: 1-2 hours
- **Configurations**: 81 tile combinations (9×9 grid: 16, 24, 32, 48, 64, 96, 128, 192, 256)
- **Workloads**: 10 shapes (Q-Proj 512/1024/2048/4096/8192, Single Token, FFN Batch 16/64/128/256)
- **Output**: CSV file in `tile_sweep_results/tile_sweep_TIMESTAMP.csv`

### Analyzing Results
```bash
python3 analyze_tile_sweep.py tile_sweep_results/quick_sweep_*.csv
```

**Output includes**:
- Best configurations per workload category (square, wide, single-token)
- Top 3 tile settings for each workload
- Universal configuration (single setting optimal across all workloads)
- Performance range and sensitivity analysis

## Example Usage

### Find Best Settings for FP32 Path
```bash
# 1. Run FP32 quick sweep with microkernel enabled
./benchmark_tile_sweep_quick.sh

# 2. Analyze results
python3 analyze_tile_sweep.py tile_sweep_results/quick_sweep_*.csv

# 3. Test recommended settings
LLAMINAR_IQ4_GEMM_MICROKERNEL=1 \
LLAMINAR_IQ4_M_TILE=64 \
LLAMINAR_IQ4_N_TILE=32 \
./run_benchmark.sh benchmark_iq4nl_gemm
```

### Find Best Settings for BF16 Path
```bash
# 1. Run BF16 quick sweep
./benchmark_tile_sweep_bf16.sh

# 2. Analyze results
python3 analyze_tile_sweep.py tile_sweep_results/bf16_sweep_*.csv

# 3. Test recommended settings
LLAMINAR_IQ4_GEMM_MICROKERNEL=1 \
LLAMINAR_IQ4_M_TILE_BF16=48 \
LLAMINAR_IQ4_N_TILE_BF16=96 \
./run_benchmark.sh benchmark_iq4nl_gemm
```

### Compare FP32 vs BF16 with Same Tile Settings
```bash
# 1. Run combined sweep (measures both paths)
./benchmark_tile_sweep_combined.sh

# 2. Analyze both paths
python3 analyze_tile_sweep.py tile_sweep_results/combined_fp32_*.csv
python3 analyze_tile_sweep.py tile_sweep_results/combined_bf16_*.csv

# 3. Compare best configs from each path
# Example: If FP32 optimal is 64×32 and BF16 optimal is 48×96
LLAMINAR_IQ4_GEMM_MICROKERNEL=1 \
LLAMINAR_IQ4_M_TILE=64 \
LLAMINAR_IQ4_N_TILE=32 \
LLAMINAR_IQ4_M_TILE_BF16=48 \
LLAMINAR_IQ4_N_TILE_BF16=96 \
./run_benchmark.sh benchmark_iq4nl_gemm
```

### Compare Against Adaptive Default
```bash
# Baseline: Adaptive tiling (no overrides)
LLAMINAR_IQ4_GEMM_MICROKERNEL=1 \
./run_benchmark.sh benchmark_iq4nl_gemm > baseline.txt

# Test fixed tiles
LLAMINAR_IQ4_GEMM_MICROKERNEL=1 \
LLAMINAR_IQ4_M_TILE=96 \
LLAMINAR_IQ4_N_TILE=96 \
./run_benchmark.sh benchmark_iq4nl_gemm > override.txt

# Compare
diff -y baseline.txt override.txt
```

## Performance Tuning Tips

### When to Override Adaptive Tiling

**Consider manual tuning if**:
- Your workload is highly specialized (e.g., always 1024 tokens)
- You're targeting specific hardware (different L2/L3 cache sizes)
- Profiling shows cache miss patterns not addressed by adaptive strategy

**Stick with adaptive defaults if**:
- Workloads vary widely in size
- Code needs to run on diverse hardware
- Performance is already within 5% of optimal

### Tile Size Guidelines

**Smaller tiles (M/N_TILE = 32-48)**:
- **Pros**: Better L1/L2 cache locality, less memory bandwidth
- **Cons**: Higher loop overhead, reduced compute intensity
- **Best for**: Very large batches (m ≥ 4096), memory-bound ops

**Larger tiles (M/N_TILE = 96-128)**:
- **Pros**: Better compute reuse, lower overhead
- **Cons**: Higher cache pressure, potential evictions
- **Best for**: Medium batches (m ≤ 1024), compute-bound ops

**Balanced tiles (M/N_TILE = 64)**:
- **Universal sweet spot**: Works well across most workloads
- Recommended starting point for manual tuning

### Microkernel Considerations

**Enable microkernel when**:
- Batch size m ∈ [1024, 4096] (sweet spot: +7-9% gain)
- Medium FFN batches (batch=16: +35%, batch=64: +2.5%)
- Acceptable slight regression on very large ops (<4%)

**Disable microkernel when**:
- Targeting peak performance on single-token decode
- Very large batches (m ≥ 8192) where overhead reduction less critical
- Code simplicity preferred over marginal gains

## CSV Output Format

```csv
M_TILE,N_TILE,Workload,m,n,k,GFLOPS,Time_ms
64,64,Q-Proj-1024,1024,896,896,267.26,6.37
64,64,FFN-Batch-16,16,4864,2048,241.43,6.24
...
```

## Implementation Details

### Code Locations

- **Tile override parsing**: `src/utils/DebugEnv.{h,cpp}`
  - `env.dequant.iq4_override_m_tile`
  - `env.dequant.iq4_override_n_tile`

- **Adaptive selection logic**: `src/tensors/IQ4_NLTensor.h` (lines ~850-935)
  - Aspect ratio analysis
  - Default tile size tables
  - Override check at beginning of tiling logic

- **Microkernel implementation**: `src/tensors/IQ4_NLTensor.h` (lines ~945-1020)
  - Multi-column decode (4 columns at once)
  - K-blocks outer loop, columns inner loop

### Memory Footprint

**Decode buffer size** (per thread):
```
Size = N_TILE × k × sizeof(float)
     = N_TILE × 896 × 4 bytes
```

Example: N_TILE=64 → ~229 KB per thread

**Total working set** (rough estimate):
```
Total ≈ (M_TILE + N_TILE) × k × 4 bytes
```

Example: M=64, N=64, k=896 → ~458 KB

Target: Keep working set ≤ L2 cache size (typically 256-512 KB per core)

## Troubleshooting

### Performance Regression with Overrides

**Symptoms**: Manual tiles perform worse than adaptive defaults
**Causes**:
- Tiles too large: L2 cache thrashing
- Tiles too small: Loop overhead dominates
- Mismatch with workload characteristics

**Solutions**:
1. Run quick sweep to find actual optimal settings
2. Compare results against adaptive default
3. Check if microkernel toggle affects outcome

### Sweep Takes Too Long

**Issue**: Full sweep (81 configs × 10 workloads) can take 1-2 hours
**Solutions**:
- Use `benchmark_tile_sweep_quick.sh` instead (9 configs × 4 workloads, ~15 min)
- Manually test specific tile ranges based on workload type
- Profile once, then hard-code optimal settings for production

### CSV Parsing Errors in Analysis

**Symptoms**: `analyze_tile_sweep.py` fails with KeyError or ValueError
**Causes**:
- Benchmark output format changed
- Incomplete benchmark run (interrupted)
- Extraction regex mismatch

**Solutions**:
- Check CSV file has complete data (no empty GFLOPS values)
- Verify benchmark output format matches extraction patterns
- Re-run sweep if file corrupted

## Future Work

- [ ] Extend tile overrides to BF16 path (currently FP32 only)
- [ ] Auto-tuning based on hardware detection (L2/L3 cache sizes)
- [ ] Profile-guided tile selection (measure cache miss rates)
- [ ] Adaptive unroll factor for microkernel (currently fixed at 4)
- [ ] Per-operator tile caching (avoid re-computing aspect ratio)

## References

- Microkernel design: Inspired by BLIS microkernel architecture
- Cache tiling: Based on classical blocked matrix multiplication
- Aspect ratio heuristic: Custom analysis for fused dequant+GEMM pattern
