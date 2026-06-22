# Flash Attention 2 Prefill — Wavefront-Cooperative Redesign

**Date**: 2026-02-25
**Target**: AMD MI50 (gfx906), Qwen2.5-7B (head_dim=128, n_heads=28, n_kv_heads=4)
**Baseline**: 764.10 tok/s prefill, kernel avg 15.67ms (112 calls/inference)

---

## 1. Problem Statement

### Why Incremental Wins Are Capped

For head_dim=128 (Qwen2.5-7B), the runtime tile config is **tile_q=64 × tile_kv=32** (not 64×64 — that needs 97.5KB LDS, which exceeds 64KB). Double-buffered K/V + score matrix consumes **58,368 bytes** (57.0KB), allowing only **1 block per CU → 4 wavefronts → 10% occupancy**.

With only 4 wavefronts, the SIMD has almost no ability to hide memory latency (LDS: ~40 cycles, HBM: ~400 cycles). This is why incremental optimizations (dot2, LDS padding, softmax fast-path) yielded only -7.4% at the kernel level — the kernel is *execution-latency-bound*, not compute-bound or bandwidth-bound.

| Resource | Current Value | Bottleneck? |
|----------|--------------|-------------|
| LDS/block | 58,368 B (89% of 64KB) | **YES** — limits to 1 block/CU |
| VGPRs/thread | 36 | No — allows 7 waves/SIMD |
| HBM bandwidth | ~43 GB/s utilized (~4% of 1024 GB/s) | No |
| Compute (dot2) | 512 dot2/thread/KV-tile | No |
| **Occupancy** | **4 waves/CU (10%)** | **ROOT CAUSE** |

### LDS Budget Breakdown (Current Design)

```
Q tile     (64×130 FP16):          16,640 B
K tiles    (2×32×130 FP16):        16,640 B  ← double-buffered
V tiles    (2×32×130 FP16):        16,640 B  ← double-buffered
Scores     (64×33 FP32):            8,448 B
─────────────────────────────────────────────
TOTAL                              58,368 B (57.0 KB)
```

---

## 2. Strategy Comparison

### Strategy 1: Eliminate Double-Buffering (REJECTED)

| Sub-option | Tile Config | LDS | Blocks/CU | Waves/CU | Occupancy | KV Iters |
|------------|------------|-----|-----------|----------|-----------|----------|
| A: 64×32 single | Same tiles, no dbl-buf | 41,728 B | 1 | 4 | 10% | 19 |
| B: 32×32 single | Smaller tiles | 29,184 B | 2 | 8 | 20% | 19 |
| C: 64×64 no-pad | Larger KV tile, zero pad | 65,536 B | 1 | 4 | 10% | 10 |

**Rejection rationale**:
- Sub-option A saves LDS but doesn't cross the 1→2 blocks/CU threshold. Same occupancy, but worse because we lose compute/load overlap.
- Sub-option B achieves 2 blocks/CU → 8 wavefronts → 2× occupancy. But we lose double-buffering AND halve the tile area, meaning more KV loop iterations with more `__syncthreads()` barriers.
- Sub-option C exactly fills 64KB with zero padding, causing severe bank conflicts + still only 4 wavefronts.

At most 2× occupancy at the cost of losing double-buffering and shrinking tiles. Unfavorable tradeoff.

### Strategy 2: Wavefront-Cooperative Prefill (SELECTED)

Adapt the decode kernel's architecture where **lanes own output dims, not score elements**. Eliminates K/V tiles and score matrix from LDS entirely.

---

## 3. Cooperative Design: Architecture

### Core Concept

| Aspect | Current Design | Cooperative Design |
|--------|---------------|-------------------|
| **Thread role** | Each thread computes score elements | Each lane owns 2 output dims |
| **Q storage** | FP16 in LDS (shared) | FP16 in LDS (shared) — same |
| **K/V access** | Tiled into LDS from HBM | Loaded per-position from HBM (coalesced) |
| **Score matrix** | Materialized in LDS (8.4KB) | Never materialized — fused inline |
| **P@V accumulation** | Sequential per V row from LDS | Per-lane FMA (each lane: `O += p * V[dim]`) |
| **Cross-lane comm** | None during Q@K^T | 6 shuffles per Q-row per KV-pos (wavefrontReduceSum) |
| **LDS footprint** | 58.4KB | **10.4KB** (tile_q=32) |

### Key Architectural Choices

1. **Q rows distributed across wavefronts** (not KV-split like decode): Each wavefront independently processes its own Q rows against ALL KV positions. No inter-wavefront merge needed (unlike decode's split-K). Wavefront 0 handles Q rows 0-7, wavefront 1 handles 8-15, etc.

2. **K/V from HBM with L2 cache reuse**: All 4 wavefronts process the same KV position simultaneously, creating perfect L2 cache hits. Effective HBM traffic = `kv_len × head_dim × 4 × 2 = 596KB` per block — identical to current design.

3. **FP16 Q in LDS for occupancy**: Q stored as FP16 (2× smaller than FP32), converted inline to FP32 for the dot product. 2 extra `v_cvt_f32_f16` per lane per Q-row per KV-pos, but this is fully hidden by the ~5× higher occupancy.

4. **Online softmax fused with dot product**: After `wavefrontReduceSum`, the score is uniform across all 64 lanes. Softmax update and `O += p * V` happen immediately — no LDS write/read cycle.

### Inner Loop Pseudocode

```c
// Each wavefront handles q_per_wave Q rows (8 for tile_q=32)
float O_lane[q_per_wave][2];  // Output accumulators (2 dims per lane)
float m_i[q_per_wave];        // Online softmax max
float l_i[q_per_wave];        // Online softmax sum

for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
    // Cooperative K load: 64 lanes load contiguous K elements (coalesced!)
    // Lane i loads K[kv_pos, i] and K[kv_pos, i+64]
    float k0 = K[kv_pos * kv_stride + lane_id];
    float k1 = K[kv_pos * kv_stride + lane_id + 64];

    for (int qi = 0; qi < q_per_wave; qi++) {
        // Dot product: Q (FP16 from LDS) × K (FP32 from HBM)
        float q0 = __half2float(Q_lds[qi * q_stride + lane_id]);
        float q1 = __half2float(Q_lds[qi * q_stride + lane_id + 64]);
        float partial = q0 * k0 + q1 * k1;

        // Wavefront-wide reduction (6 shuffles) → score uniform across lanes
        float score = wavefrontReduceSum(partial) * softmax_scale;

        // [Causal masking check — tile-level fast path applies here too]

        // Online softmax + O accumulation (per-lane, no cross-lane needed)
        float m_new = fmaxf(m_i[qi], score);
        float scale = fast_expf(m_i[qi] - m_new);
        float p = fast_expf(score - m_new);
        O_lane[qi][0] = O_lane[qi][0] * scale + p * V[kv_pos * kv_stride + lane_id];
        O_lane[qi][1] = O_lane[qi][1] * scale + p * V[kv_pos * kv_stride + lane_id + 64];
        l_i[qi] = l_i[qi] * scale + p;
        m_i[qi] = m_new;
    }
}
```

### Occupancy Analysis

| Config | LDS/block | Blocks/CU | VGPRs/lane | Waves/SIMD | **Waves/CU** | **Occupancy** | Grid blocks |
|--------|-----------|-----------|------------|------------|-------------|--------------|-------------|
| Current 64×32 | 58,368 | 1 | 36 | 7 | **4** | **10%** | 280 |
| **Coop tq=32** | **10,400** | **6** | **44** | **5** | **20** | **50%** | 532 |
| Coop tq=16 | 6,240 | 10 | 28 | 9 | **36** | 90% | 1064 |
| Coop tq=64 | 18,720 | 3 | 76 | 3 | **12** | 30% | 280 |

### VGPR Budget (tile_q=32, 8 Q-rows per wavefront)

| Register usage | VGPRs |
|---------------|-------|
| O_acc: 8 Q-rows × 2 dims | 16 |
| m_i: 8 Q-rows | 8 |
| l_i: 8 Q-rows | 8 |
| K values (k0, k1) | 2 |
| Q values (q0f, q1f) | 2 |
| partial_dot, score, p, m_new, scale | 5 |
| V loads (v0, v1) | 2 |
| Loop counters, addresses | ~3 |
| **Total** | **~44** |

At 44 VGPRs: `floor(256/44) = 5` waves/SIMD → 20 waves/CU (VGPR-limited, not LDS-limited).

### What This Eliminates

- Score matrix in LDS (8,448 bytes) — gone
- K tile in LDS (16,640 bytes × 2 stages) — gone
- V tile in LDS (16,640 bytes × 2 stages) — gone
- K/V double-buffer ping-pong logic — gone
- `__syncthreads()` barriers between phases — gone (no shared state between wavefronts)
- P@V as a separate loop — fused into the dot product loop

### What This Adds

- 6 shuffle instructions per Q-row per KV-position (wavefrontReduceSum)
- 2 `v_cvt_f32_f16` per Q-row per KV-position (Q FP16→FP32)

### Shuffle Overhead Estimate (tile_q=32, 8 Q-rows/wave)

- Shuffles/wavefront: 8 × 596 × 6 = 28,608
- At ~1 cycle/shuffle: 28.6K cycles / 1.5 GHz ≈ 19 μs
- Kernel budget: 15,670 μs → shuffle overhead is **0.12%** of runtime. Negligible.

---

## 4. Expected Performance Impact

| Factor | Effect | Magnitude |
|--------|--------|-----------|
| Occupancy: 4 → 20 waves | More wavefronts to hide latency | **Very large** (~2-4× latency hiding) |
| No LDS barriers | Eliminate `__syncthreads()` between phases | Moderate (19 barriers/block removed) |
| Fused P@V | Eliminate separate V loading + accumulation pass | Moderate |
| Higher grid (532 vs 280) | Better CU utilization at tail | Small |
| Shuffle overhead | Additional cross-lane communication | Negligible (-0.12%) |
| No double-buffer overlap | Lose async K/V prefetch | Small negative |

**Conservative estimate**: 20-40% kernel speedup (15.67ms → 9.4-12.5ms), translating to ~10-20% prefill throughput improvement (764 → 840-920 tok/s).

**ACTUAL RESULTS** (2026-02-25):
- Kernel avg: **6.49ms** (was 15.67ms) — **2.41× faster** (59% reduction)
- Prefill throughput: **1138 tok/s** (was 764) — **49% improvement**
- Decode unchanged: 58.45 tok/s (was 58.53)
- Kernel resources: 84 VGPRs, 5 spills, 24B scratch, `__launch_bounds__(256, 3)`
- Occupancy: 3 waves/SIMD = 12 waves/CU (was 4 waves/CU = 10%)

**Critical implementation lesson**: Template parameters are MANDATORY for register
promotion. Without compile-time loop bounds, the hipcc compiler places per-Q-row
accumulators in scratch memory (432B private segment), making the kernel 2.3× SLOWER
than the tiled kernel it was meant to replace.

---

## 5. Implementation Phases

### Phase 1: Write the Cooperative Kernel

**File**: `src/v2/kernels/rocm/attention/ROCmFlashAttentionKernels.hip`

Write `flash_attention_2_cooperative_mi50_kernel` as a new `__global__` function alongside the existing kernel. The existing kernel is preserved as fallback.

**Key implementation details**:
- Grid: `(n_heads, num_q_tiles, batch_size)` — same as current
- Block: 256 threads (4 wavefronts) — same as current
- `__launch_bounds__(256, 6)` — target 6 blocks/CU since LDS allows it (check if VGPR pressure overrides)
- Q loaded cooperatively into LDS as FP16 (same as current)
- No K/V tiles in LDS — loaded per-position from HBM
- Each wavefront independently handles `tile_q/4` Q rows against all KV positions
- Output written directly to global O buffer (each wavefront writes its own Q rows independently)
- Causal masking: tile-level fast path + early termination (when `kv_pos > max_q_pos + position_offset` for all Q rows in this wavefront, break)
- GQA: `kv_head_idx = head_idx / gqa_ratio` — same as current
- Sliding window attention: same per-element check as current (only in slow path)
- External mask: same as current (only in slow path)
- `softmax_scale` pre-multiplication: fused into `score = wavefrontReduceSum(partial) * softmax_scale`

**LDS layout**:
```c
struct CoopLDSLayout {
    // Q tile: tile_q × (head_dim + pad) × sizeof(__half)
    int q_tile_size;
    int q_stride;  // head_dim + FA2_LDS_PAD_FP16

    void compute(int head_dim, int tile_q) {
        q_stride = head_dim + FA2_LDS_PAD_FP16;
        q_tile_size = tile_q * q_stride * sizeof(__half);
    }
};
// Total LDS = q_tile_size only (~10KB for tile_q=32)
```

**Output write pattern**:
Each wavefront writes its Q rows independently — no inter-wavefront coordination needed.
Lane `i` writes `O[global_q_row, lane_id]` and `O[global_q_row, lane_id + 64]`.
This is coalesced because adjacent lanes write adjacent addresses.

### Phase 2: Wire Up the Launch Path

**File**: `src/v2/kernels/rocm/attention/ROCmFlashAttentionKernels.hip`

Modify `hipFlashAttn_prefill_fa2()` to select between the cooperative kernel and the existing tiled kernel based on head_dim. For head_dim >= 64, the cooperative kernel should be preferred since it dramatically reduces LDS pressure.

**Tile selection for cooperative kernel**:
- tile_q candidates: {64, 32, 16}
- Select based on VGPR budget: for each candidate, compute VGPRs needed and pick the largest tile_q that allows at least 3 waves/SIMD
- For head_dim=128: tile_q=32 is optimal (44 VGPRs, 5 waves/SIMD, 20 waves/CU)

**Launch configuration**:
```c
const int num_q_tiles = (seq_len + tile_q - 1) / tile_q;
dim3 grid(n_heads, num_q_tiles, batch_size);
dim3 block(FA2_THREADS_PER_BLOCK);
size_t lds_size = coop_layout.q_tile_size;  // ~10KB

hipLaunchKernelGGL(
    flash_attention_2_cooperative_mi50_kernel,
    grid, block, lds_size, stream, ...);
```

### Phase 3: Correctness Verification

**Test**: Run greedy sampling and compare output token-for-token with the existing kernel.

```bash
# Current kernel (reference)
./build_v2_release/llaminar2 -d rocm:0 -m models/Qwen2.5-7B-Instruct-Q8_0.gguf \
    -p "The answer to 2+2 is" -n 10 -t 0

# Expected output: "4. What is the answer to 2"
```

The cooperative kernel must produce identical greedy output. Small floating-point differences in intermediate values are expected (different accumulation order) but should not affect top-1 token selection.

### Phase 4: Benchmark and Profile

```bash
# End-to-end benchmark
./build_v2_release/llaminar2 --benchmark -d rocm:0 -m models/Qwen2.5-7B-Instruct-Q8_0.gguf

# Kernel-level profiling
rocprof --stats -o /tmp/coop_profile.csv \
    ./build_v2_release/llaminar2 --benchmark -d rocm:0 -m models/Qwen2.5-7B-Instruct-Q8_0.gguf

# ISA extraction for VGPR/occupancy verification
/opt/rocm/llvm/bin/llvm-objdump -d <code_object.elf> | grep -E 'private_segment|vgpr|sgpr'
```

**Success criteria**:
- Kernel avg < 12.5ms (vs 15.67ms current) — at least 20% improvement
- Prefill tok/s > 840 (vs 764.10 current) — at least 10% end-to-end improvement
- Identical greedy output for test prompt
- No increase in decode regression

### Phase 5: Cleanup and Fallback Logic

- Add runtime selection: cooperative for head_dim >= 64, tiled for head_dim < 64
- Update `README.md` documentation
- Remove any dead code from the tiled kernel (keep it intact as fallback)
- Update constants and comments

---

## 6. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| VGPR spills at tile_q=32 | Fall back to tile_q=16 (28 VGPRs, 36 waves/CU) |
| L2 cache thrashing from all-HBM K/V | 4 wavefronts share same KV position → perfect L2 reuse |
| Register pressure from 8 Q-rows of softmax state | Compiler may spill m_i/l_i arrays — profile ISA to verify |
| Causal masking divergence | Tile-level fast path eliminates branching for ~90% of tiles |
| Numerical differences | Expected due to different accumulation order; verify greedy output matches |

---

## 7. Files Modified

| File | Changes |
|------|---------|
| `src/v2/kernels/rocm/attention/ROCmFlashAttentionKernels.hip` | New kernel function + launch path modification |
| `src/v2/kernels/rocm/attention/README.md` | Updated documentation |

---

## 8. Profiling Reference Data

### Current Kernel Profile (rocprof)

| Rank | Kernel | Calls | Total ms | % GPU | Avg ms |
|------|--------|-------|----------|-------|--------|
| 1 | gemv_int8_int8_grid_kpar_vnni (decode GEMM) | 100352 | 4814 | 38.78% | 0.048 |
| 2 | flash_attention_2_mi50_kernel (AFTER opt) | 112 | 1755 | 14.26% | 15.67 |
| 3 | gemv_int8_wide_vnni_scaled (LM Head) | 516 | 1463 | 11.79% | 2.84 |
| 4 | qgemm_wide_tile_v3 (prefill GEMM) | 560 | 672 | 5.41% | 1.20 |
| 5 | qgemm_wide_tile_v7 (prefill GEMM) | 224 | 571 | 4.60% | 2.55 |
| 6 | flash_decoding_mi50_kernel | 14336 | 501 | 4.04% | 0.035 |

### Current ISA Resources

| Resource | Value |
|----------|-------|
| VGPRs | 36 |
| SGPRs | ~88 |
| Private seg | 144 bytes (spills) |
| Code size | 5072 bytes |
| v_dot2_f32_f16 | 5 (in Q@K^T loop) |
| v_fma_f32 | 16 |
| v_cvt_f16_f32 | 15 (in P@V loop) |
