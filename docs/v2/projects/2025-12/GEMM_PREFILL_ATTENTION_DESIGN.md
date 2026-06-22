# GEMM-Based Prefill Attention Kernel Design

**Author:** David Sanftenberg  
**Date:** December 2025  
**Status:** Proposal  

## Executive Summary

This document proposes replacing the current streaming decode-style prefill attention kernel with a GEMM-based implementation. The current kernel processes one KV position at a time (optimized for memory-bound decode), but prefill is compute-bound and should be structured as two GEMMs with an intervening softmax.

**Expected Impact:** 5-10× prefill throughput improvement by properly utilizing CPU GEMM capabilities.

---

## 1. Problem Statement

### 1.1 Current Implementation Analysis

The existing `JitFusedAttentionWo` prefill kernel uses a **streaming decode pattern**:

```
for each tile of Q (4 queries):
    for each head:
        for each KV position:           ← KV is INNER loop
            for each query in tile:
                score[q] += Q[q] · K[kv]    // Individual dot product (VNNI)
                online_softmax_update(q)
                context[q] += weight * V[kv]
```

**Problems:**

| Issue | Impact |
|-------|--------|
| Individual dot products | 16K VNNI ops instead of 1 GEMM for seq=128 |
| Poor K/V reuse | Each K[kv] used for only 4 queries (tile size) |
| No GEMM batching | Cannot exploit CPU GEMM microarchitecture |
| Online softmax overhead | Per-KV-position exp/max/sum operations |
| Instruction mix | Memory-bound pattern on compute-bound workload |

### 1.2 Compute Characteristics

| Metric | Decode | Prefill |
|--------|--------|---------|
| seq_len_q | 1 | N (128-4096) |
| Work complexity | O(N × d) | O(N² × d) |
| Bottleneck | Memory bandwidth | Compute throughput |
| Optimal pattern | Stream K/V once | Batched GEMM |

For Qwen2.5-7B with seq_len=128, head_dim=128:
- **Current:** 128 × 128 × 2 = 32,768 individual VNNI dot products per head
- **GEMM:** 2 matrix multiplications of ~2M FLOPs each per head

### 1.3 Memory Analysis

Materializing the attention score matrix requires:
```
Memory = seq_len² × sizeof(float) × num_heads
       = 128² × 4 × 28
       = 2.3 MB (total)
       = 64 KB (per head)
```

Per-head working set (64KB) fits comfortably in L2 cache (256KB-1MB typical).
For seq_len=512: 1MB per head - may need tiled softmax.

---

## 2. Proposed Architecture

### 2.1 High-Level Structure

Replace streaming pattern with **two-GEMM attention**:

```
┌─────────────────────────────────────────────────────────────────┐
│  GEMM-BASED PREFILL ATTENTION                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  for each head (parallel across heads):                         │
│                                                                 │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │ PHASE 1: Score Computation (GEMM)                       │  │
│    │   S = Q × K^T                                           │  │
│    │   [N, d] × [d, N] → [N, N]                              │  │
│    └─────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │ PHASE 2: Softmax (Row-wise)                             │  │
│    │   P = softmax(S × scale, dim=-1)                        │  │
│    │   [N, N] → [N, N]                                       │  │
│    │   + Causal mask application                             │  │
│    └─────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │ PHASE 3: Context Computation (GEMM)                     │  │
│    │   C = P × V                                             │  │
│    │   [N, N] × [N, d] → [N, d]                              │  │
│    └─────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │ PHASE 4: Output Projection                              │  │
│    │   O = C × Wo                                            │  │
│    │   [N, d] → [N, d_model]                                 │  │
│    └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Tiled Variant for Long Sequences

For sequences exceeding L2 cache capacity (seq_len > ~512), use tiled attention:

```
┌─────────────────────────────────────────────────────────────────┐
│  TILED GEMM ATTENTION (Flash-Attention Style)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  for each Q_tile (Bq queries):                                  │
│    Initialize: context[Bq, d] = 0, max[Bq] = -inf, sum[Bq] = 0  │
│                                                                 │
│    for each KV_tile (Bkv positions):                            │
│      ┌───────────────────────────────────────────────────────┐  │
│      │ S_tile = Q_tile × K_tile^T     [Bq, d] × [d, Bkv]     │  │
│      │ S_tile = S_tile × scale + causal_mask                 │  │
│      │                                                       │  │
│      │ // Online softmax update                              │  │
│      │ new_max = max(old_max, rowmax(S_tile))                │  │
│      │ correction = exp(old_max - new_max)                   │  │
│      │ P_tile = exp(S_tile - new_max)                        │  │
│      │ new_sum = old_sum * correction + rowsum(P_tile)       │  │
│      │                                                       │  │
│      │ // Context update with correction                     │  │
│      │ context = context * correction + P_tile × V_tile      │  │
│      └───────────────────────────────────────────────────────┘  │
│                                                                 │
│    context = context / sum   // Final normalization             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Tile sizes:** Bq = 64-128, Bkv = 64-128 (tunable based on L2 size)

---

## 3. Implementation Design

### 3.1 Kernel Variants

| Kernel | Use Case | Score Storage | Algorithm |
|--------|----------|---------------|-----------|
| `JitGemmPrefillSmall` | seq ≤ 256 | Full N×N in L2 | Two GEMMs + softmax |
| `JitGemmPrefillTiled` | seq > 256 | Bq×Bkv tiles | Flash-style tiled |
| `JitFusedAttentionWo` | seq = 1 | None (streaming) | Current decode kernel |

### 3.2 Quantized GEMM Strategy (AVX-512 VNNI)

**Key Insight:** Keep GEMMs in integer domain using AVX-512 VNNI (`vpdpbusd`).

AVX-512 VNNI provides ~4× throughput over FP32 FMA by computing 4 int8×int8 
multiply-accumulates per element per cycle. Since Q, K, V are already Q8_1,
we can exploit this for the score GEMM without any dequantization.

**Data Flow:**

```
┌─────────────────────────────────────────────────────────────────┐
│  FULL VNNI ATTENTION DATA FLOW                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Q [Q8_1] ──┐                                                   │
│             ├──► S = Q × K^T  [INT32 accum] ──► [FP32 scores]   │
│  K [Q8_1] ──┘         VNNI GEMM                 (scale correct) │
│                            │                                    │
│                            ▼                                    │
│                    softmax(S × scale)  [FP32]                   │
│                            │                                    │
│                            ▼                                    │
│                    quantize(P) ──► P [Q8_1]                     │
│                            │       (per-row scale)              │
│                            ▼                                    │
│  P [Q8_1] ──┐                                                   │
│             ├──► C = P × V  [INT32 accum] ──► [FP32 context]    │
│  V [Q8_1] ──┘         VNNI GEMM                 (scale correct) │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Both GEMMs use full VNNI:**

| GEMM | A type | B type | Accumulator | Notes |
|------|--------|--------|-------------|-------|
| Q×K^T | Q8_1 | Q8_1 | INT32 → FP32 | Full VNNI, per-block scales |
| P×V | Q8_1 | Q8_1 | INT32 → FP32 | Full VNNI, P has per-row scale |

**P Quantization Strategy:**

After softmax, P values are in [0, 1] range (probabilities). Quantize per-row:
```
For each row q:
  scale_P[q] = max(P[q, :]) / 127.0f  // Per-row scale
  P_q8[q, :] = round(P[q, :] / scale_P[q])  // Quantized values
  sum_qs_P[q] = sum(P_q8[q, :])  // For VNNI correction
```

This is efficient because:
1. Softmax already computed row-wise max (can reuse)
2. Per-row quantization matches GEMM access pattern
3. P values are non-negative, so half the int8 range is sufficient

**Q8_1 Block GEMM Math:**

For Q8_1 × Q8_1 dot product over blocks:
```
score = Σ_b [ (Σ_i Q[b,i] * K[b,i]) * d_Q[b] * d_K[b] 
              - 16 * sum_qs_K[b] * d_Q[b] * d_K[b] ]

Where:
  Q[b,i], K[b,i] = int8 quantized values
  d_Q[b], d_K[b] = FP16 per-block scales
  sum_qs_K[b] = INT16 sum of K's quantized values (for unsigned correction)
  16 = correction factor for unsigned→signed conversion in vpdpbusd
```

**Memory Working Set (per head):**

```
Q blocks: N × (head_dim/32) × 36 bytes = 128 × 4 × 36 = 18 KB
K blocks: N × (head_dim/32) × 36 bytes = 18 KB
V blocks: N × (head_dim/32) × 36 bytes = 18 KB
P (Q8_1): N × N bytes + N × 4 (scales) = 16 KB + 0.5 KB = 16.5 KB
Context: tile × head_dim × 4 bytes = 4 × 128 × 4 = 2 KB (tile only)
Total: ~73 KB per head (fits L2 easily)
```

**Advantages over FP32 approach:**
1. **No dequant cost** for either GEMM (Q, K, V, P all quantized)
2. **4× arithmetic density** from VNNI on both GEMMs
3. **Lower memory bandwidth** (P is 1/4 size of FP32)
4. **Cache efficiency** (4× smaller P working set)

### 3.3 Memory Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  STACK FRAME LAYOUT (VNNI Version)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [rsp + 0]:      Callee-saved registers (64 bytes)              │
│  [rsp + 64]:     Spill slots for loop variables (128 bytes)     │
│  [rsp + 192]:    S_fp32 buffer [N, N] (N×N×4 bytes)             │
│  [rsp + S_off]:  Context buffer [tile, head_dim] (tile×d×4)     │
│                                                                 │
│  Note: Q, K, V remain in original Q8_1 format in input buffers  │
│        No dequantization buffers needed!                        │
│                                                                 │
│  For N=128, d=128, tile=4:                                      │
│    S (scores): 64 KB                                            │
│    Context tile: 2 KB (only current tile, not full N×d)         │
│    Total stack: ~66 KB                                          │
│                                                                 │
│  Compare to FP32 dequant approach: 320 KB                       │
│  Memory savings: ~5×                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 Q8_1 Scale Correction Details

**VNNI computes unsigned×signed products:**

```
vpdpbusd: acc += Σ (uint8_a[i] × int8_b[i])

But Q8_1 stores signed int8 values. We convert:
  uint8_a = int8_a + 128  (stored as unsigned for vpdpbusd)
  
This introduces an offset:
  acc = Σ (int8_a + 128) × int8_b
      = Σ int8_a × int8_b + 128 × Σ int8_b
      = true_dot + 128 × sum_qs_b
      
To correct: true_dot = acc - 128 × sum_qs_b

With Q8_1 block scales:
  score = (acc - 128 × sum_qs_K) × d_Q × d_K
```

**Implementation:**

```cpp
void emit_score_scale_correction(int q_idx, int num_kv_outputs) {
    // After VNNI accumulation, zmm_acc[q] contains INT32 partial sums
    // Each lane is sum over all K-blocks for one (q, kv) pair
    
    // 1. Convert INT32 → FP32
    vcvtdq2ps(zmm_acc, zmm_acc);
    
    // 2. Load combined scales: d_Q[q] × d_K[kv] for all kv positions
    //    Pre-computed and stored during K loading phase
    vmulps(zmm_acc, zmm_acc, zmm_combined_scales);
    
    // 3. Subtract correction: 128 × sum_qs_K[kv] × d_Q[q] × d_K[kv]
    //    Pre-computed: correction[kv] = 128 × sum_qs_K_total[kv] × d_K[kv]
    //    Final: acc -= correction[kv] × d_Q[q]
    vfnmadd231ps(zmm_acc, zmm_correction, zmm_d_Q_broadcast);
    
    // 4. Apply attention scale: 1/sqrt(head_dim)
    vmulps(zmm_acc, zmm_acc, zmm_attn_scale);
    
    // Store to score matrix S[q, :]
    vmovups(ptr[score_ptr + q_idx * seq_len * 4], zmm_acc);
}
```

### 3.4 JIT Code Structure

```cpp
/**
 * @class JitVnniPrefillAttention
 * @brief JIT-compiled GEMM-based prefill attention kernel using AVX-512 VNNI
 * 
 * Based on QuantisedGemmJit_M2 pattern but specialized for attention:
 * - Score GEMM: Q × K^T with per-head scale application
 * - Softmax: Vectorized exp with JitFastExpEmitter pattern
 * - Context GEMM: P × V with P quantized on-the-fly
 * 
 * Code size: ~300KB (larger than M2 due to additional phases)
 * Buffer: 4096 * 80 bytes to accommodate score/context/softmax code
 */
class JitVnniPrefillAttention : public Xbyak::CodeGenerator {
public:
    JitVnniPrefillAttention() : Xbyak::CodeGenerator(4096 * 80) {
        generate();
    }

    using kernel_func_t = void (*)(const VnniPrefillAttentionParams* params);
    
    kernel_func_t get_kernel() {
        return getCode<kernel_func_t>();
    }

private:
    void generate() {
        using namespace Xbyak;
        
        // ================== REGISTER DEFINITIONS ==================
        // Following QuantisedGemmJit_M2 conventions:
        const Reg64& reg_params = rdi;      // params pointer
        const Reg64& reg_Q = rsi;           // Q pointer (Q8_1 blocks)
        const Reg64& reg_K = rdx;           // K pointer (Q8_1 blocks)
        const Reg64& reg_V = rcx;           // V pointer (Q8_1 blocks)
        const Reg64& reg_C = r8;            // Output context pointer
        const Reg64& reg_S = r9;            // Score buffer pointer
        
        // Working registers
        const Reg64& reg_loop_q = r10;      // Query loop counter
        const Reg64& reg_loop_kv = r11;     // KV position loop counter
        const Reg64& reg_loop_k = r12;      // K-dimension loop counter
        const Reg64& reg_tmp = rax;
        
        // ZMM allocation follows M2 pattern
        // zmm0-7: Accumulators (4 queries × 2 for FP32 results or 8 for INT32)
        // zmm8-15: Temp accumulators for VNNI
        // zmm16-19: K/V data blocks
        // zmm20-21: Q/P data (broadcast)
        // zmm22-31: Constants, scales, temps

        // ================== PROLOGUE ==================
        // Save callee-saved registers (same as M2)
        push(rbx);
        push(rbp);
        push(r12);
        push(r13);
        push(r14);
        push(r15);
        
        // Allocate stack frame
        // Layout: [callee_saved(48)] [spill(128)] [score_buf(N²×4)] [P_q8(N²)] [P_scales(N×4)]
        // For N=128: 48 + 128 + 65536 + 16384 + 512 = 82608 bytes
        // Note: For larger N, may need heap allocation
        mov(rax, params.seq_len);
        imul(rax, rax);              // N²
        shl(rax, 2);                 // × sizeof(float) for scores
        add(rax, params.seq_len * params.seq_len); // + P_q8 size
        add(rax, params.seq_len * 4);  // + P_scales size
        add(rax, 176);               // + spill slots + alignment
        and_(rax, ~63);              // Align to 64 bytes
        sub(rsp, rax);
        
        // Load parameters
        mov(reg_Q, ptr[reg_params + 0]);   // Q blocks pointer
        mov(reg_K, ptr[reg_params + 8]);   // K blocks pointer
        mov(reg_V, ptr[reg_params + 16]);  // V blocks pointer
        mov(reg_C, ptr[reg_params + 24]);  // Output pointer
        // S buffer at [rsp + spill_offset]
        lea(reg_S, ptr[rsp + 176]);
        
        // Initialize constants
        mov(eax, 0x80808080);
        vpbroadcastd(zmm23, eax);          // INT8→UINT8 conversion constant
        mov(eax, -128);
        vpbroadcastd(zmm24, eax);          // Compensation constant
        
        // ================== PHASE 1: SCORE GEMM ==================
        // S[N, N] = Q[N, d] × K^T[d, N]
        emit_score_gemm_vnni();
        
        // ================== PHASE 2: SOFTMAX + P QUANTIZE ==================
        // P_q8[N, N] = quantize(softmax(S * scale))
        emit_softmax_quantize_all_rows();
        
        // ================== PHASE 3: CONTEXT GEMM ==================
        // C[N, d] = P_q8[N, N] × V[N, d]
        emit_context_gemm_vnni();
        
        // ================== EPILOGUE ==================
        // Restore stack and callee-saved registers
        add(rsp, rax);  // rax still holds stack size
        pop(r15);
        pop(r14);
        pop(r13);
        pop(r12);
        pop(rbp);
        pop(rbx);
        ret();
    }
    
    /**
     * Score GEMM implementation
     * 
     * Processes queries in tiles of 2 (like M2 kernel) for register efficiency.
     * Inner K-loop processes one Q8_1 block (32 elements) per iteration.
     */
    void emit_score_gemm_vnni() {
        // For each query tile (2 queries at a time):
        //   For each KV tile (64 positions at a time):
        //     Reset INT32 accumulators (zmm8-15)
        //     For each K-block (head_dim/32):
        //       Load Q[q, kb], K[kv, kb]
        //       8× VNNI accumulate (32 elements per block)
        //     Scale and store to S[q, kv]
        
        // This mirrors QuantisedGemmJit_M2::generate() lines 350-750
    }
    
    /**
     * Softmax + P quantization for all rows
     * 
     * Vectorized 3-pass algorithm per row:
     * - Pass 1: Find max (horizontal max reduction)
     * - Pass 2: exp(x - max) and sum (JitFastExp pattern)
     * - Pass 3: Normalize and quantize to UINT8
     */
    void emit_softmax_quantize_all_rows() {
        // For each query row:
        //   emit_softmax_quantize_row(row_idx)
        
        // This uses the pattern from QuantisedGemmJit_M2 softmax code (lines 760-900)
    }
    
    /**
     * Context GEMM implementation
     * 
     * Similar to score GEMM but with P (UINT8) × V (INT8) inputs.
     * Outputs to provided context buffer in FP32.
     */
    void emit_context_gemm_vnni() {
        // For each query tile (2 queries at a time):
        //   For each head_dim tile (64 elements at a time):
        //     Reset INT32 accumulators
        //     For each KV position (seq_len):
        //       Load P_q8[q, kv], V[kv, d]
        //       8× VNNI accumulate
        //     Scale and store to C[q, d]
    }
};

/**
 * @struct VnniPrefillAttentionParams
 * @brief Parameters for the JIT prefill attention kernel
 */
struct VnniPrefillAttentionParams {
    const void* Q;           ///< Q blocks [seq_len, head_dim/32] Q8_1
    const void* K;           ///< K blocks [seq_len, head_dim/32] Q8_1
    const void* V;           ///< V blocks [seq_len, head_dim/32] Q8_1
    float* output;           ///< Output context [seq_len, head_dim] FP32
    float* score_buffer;     ///< Scratch for scores [seq_len, seq_len] FP32
    uint8_t* P_q8_buffer;    ///< Scratch for quantized P [seq_len, seq_len] UINT8
    float* P_scales;         ///< Per-row P scales [seq_len] FP32
    int32_t* P_row_sums;     ///< Per-row P sums for compensation [seq_len] INT32
    int seq_len;             ///< Sequence length (N)
    int head_dim;            ///< Head dimension (d)
    float scale;             ///< Attention scale = 1/sqrt(head_dim)
    int position_offset;     ///< For causal mask alignment
    const float* mask;       ///< Optional attention mask [seq_len, seq_len] or nullptr
};
```

---

## 4. VNNI GEMM Implementation

### 4.1 Score GEMM: Q × K^T (Full VNNI)

**Reference Implementation Pattern: QuantisedGemmJit_M2**

The score GEMM follows the same pattern as our existing `QuantisedGemmJit_M2` kernel,
which computes `C = A × B` where both A (activations) and B (weights) are Q8_1.

**Key Implementation Details from Existing Kernel:**

```cpp
/**
 * Register allocation (from QuantisedGemmJit_M2.h):
 * 
 * | Register    | Purpose                              |
 * |-------------|--------------------------------------|
 * | zmm0-3      | Row 0 FP32 accumulators (64 outputs) |
 * | zmm4-7      | Row 1 FP32 accumulators (64 outputs) |
 * | zmm8-11     | Row 0 INT32 temp accumulators        |
 * | zmm12-15    | Row 1 INT32 temp accumulators        |
 * | zmm16-19    | B data (4 blocks × 16 columns)       |
 * | zmm20-21    | A data (row 0 and row 1 broadcast)   |
 * | zmm22       | Scale broadcast                      |
 * | zmm23       | Constant 0x80808080 for INT8→UINT8   |
 * | zmm24       | Constant -128 for compensation       |
 * | zmm26-29    | B scales (4 × 16 floats)             |
 */

/**
 * VNNI inner loop structure (8 iterations per Q8_1 block):
 * 
 * vpdpbusd computes: acc += Σ (uint8)a[i] × (int8)b[i] for i in [0,3]
 * Each iteration processes 4 INT8 elements, 8 iters × 4 = 32 per K-block.
 */
for (int i = 0; i < 8; ++i) {
    // Load 4 bytes from Q (row 0), convert INT8→UINT8
    // Q8_1 layout: [d:FP16(2B)][sum_qs:INT16(2B)][qs[32]:INT8]
    vpbroadcastd(zmm_q0, ptr[reg_Q_cursor + 4 + i * 4]);
    vpxord(zmm_q0, zmm_q0, zmm_128);  // XOR 0x80 converts signed→unsigned
    
    // Load 4 bytes from Q (row 1) at A_stride offset
    vpbroadcastd(zmm_q1, ptr[reg_Q_cursor + A_stride + 4 + i * 4]);
    vpxord(zmm_q1, zmm_q1, zmm_128);
    
    // Load K data for 64 KV positions (4 ZMM × 16 positions each)
    vmovups(zmm_k0, ptr[reg_K_cursor + 0 * 64]);
    vmovups(zmm_k1, ptr[reg_K_cursor + 1 * 64]);
    vmovups(zmm_k2, ptr[reg_K_cursor + 2 * 64]);
    vmovups(zmm_k3, ptr[reg_K_cursor + 3 * 64]);
    
    // VNNI for query row 0 against all 64 KV positions
    vpdpbusd(zmm8, zmm_q0, zmm_k0);   // acc[0, 0:16]
    vpdpbusd(zmm9, zmm_q0, zmm_k1);   // acc[0, 16:32]
    vpdpbusd(zmm10, zmm_q0, zmm_k2);  // acc[0, 32:48]
    vpdpbusd(zmm11, zmm_q0, zmm_k3);  // acc[0, 48:64]
    
    // VNNI for query row 1
    vpdpbusd(zmm12, zmm_q1, zmm_k0);  // acc[1, 0:16]
    vpdpbusd(zmm13, zmm_q1, zmm_k1);  // acc[1, 16:32]
    vpdpbusd(zmm14, zmm_q1, zmm_k2);  // acc[1, 32:48]
    vpdpbusd(zmm15, zmm_q1, zmm_k3);  // acc[1, 48:64]
    
    // Prefetch next K data (4 iterations ahead)
    prefetcht0(ptr[reg_K_cursor + 1024]);
    
    // Advance K cursor by 256 bytes
    add(reg_K_cursor, 256);
}
```

**INT8→UINT8 Compensation (Existing Pattern):**

```cpp
// vpdpbusd requires unsigned×signed. Our Q values are signed INT8.
// Convert: Q_uint8 = Q_int8 + 128 (via XOR with 0x80)
// This introduces bias: acc_raw = Σ (Q+128) × K = Σ Q×K + 128×Σ K
// Correction: acc_true = acc_raw - 128 × sum_qs_K

// From QuantisedGemmJit_M2: Initialize accumulators with -128×compensation
vmovups(zmm_k0, ptr[reg_Comp_cursor + 0 * 64]);  // comp = Σ K[k] per column
vpmulld(zmm_k0, zmm_k0, zmm_neg_128);            // comp × (-128)
vmovdqa64(zmm8, zmm_k0);   // Row 0 acc init
vmovdqa64(zmm12, zmm_k0);  // Row 1 acc init (same B weights)
```

**Scale Application (After K-loop):**

```cpp
// Convert INT32 accumulators → FP32
vcvtdq2ps(zmm8, zmm8);
vcvtdq2ps(zmm9, zmm9);
// ... (4 per row)

// Load K scales (FP32, pre-expanded from FP16 during K packing)
vmovups(zmm_scale_k0, ptr[reg_K_scales + 0 * 64]);
vmovups(zmm_scale_k1, ptr[reg_K_scales + 1 * 64]);
// ...

// Load Q scale (FP16 → FP32 conversion)
vpbroadcastw(ymm_tmp, ptr[reg_Q_cursor]);  // Q8_1.d at offset 0
vcvtph2ps(zmm_scale_q, ymm_tmp);           // FP16 → FP32

// Apply scales: result = acc × scale_K × scale_Q
vmulps(zmm8, zmm8, zmm_scale_k0);
vmulps(zmm8, zmm8, zmm_scale_q);
// ...

// Accumulate into FP32 C accumulators (across K-blocks)
vaddps(zmm0, zmm0, zmm8);
```

**Asymmetric Correction (For IQ4_NL Weights):**

```cpp
// IQ4_NL uses asymmetric quantization with per-column min values.
// Correction: C += sum_qs_A × scale_A × mins[col]
// 
// From QuantisedGemmJit_M2.h lines 500-540:

// Load sum_qs from Q's Q8_1 block (INT16 at offset 2)
vpbroadcastw(ymm_tmp, ptr[reg_Q_cursor + 2]);
vpmovsxwd(zmm_sum_qs, ymm_tmp);   // INT16 → INT32
vcvtdq2ps(zmm_sum_qs, zmm_sum_qs); // INT32 → FP32

// Load Q scale
vpbroadcastw(ymm_tmp, ptr[reg_Q_cursor]);
vcvtph2ps(zmm_scale_q, ymm_tmp);

// sum_qs_scaled = sum_qs × scale_Q
vmulps(zmm_sum_qs_scaled, zmm_sum_qs, zmm_scale_q);

// Load mins and apply correction
vmovups(zmm_mins0, ptr[reg_Mins_cursor + 0 * 64]);
vmulps(zmm_corr0, zmm_mins0, zmm_sum_qs_scaled);
vaddps(zmm0, zmm0, zmm_corr0);  // C[0:16] += correction
```

**VNNI Throughput Analysis:**

```
vpdpbusd: 4 int8×int8 MACs per element, 16 elements per ZMM
         = 64 MACs per instruction
         = 128 ops per instruction (multiply + add)
         
For seq=128, head_dim=128:
  Output elements: 128 × 128 = 16,384
  K-blocks: 128 / 32 = 4
  Total VNNI ops: 16,384 × 4 × 64 = 4.2M int8 MACs
  At 2 vpdpbusd/cycle: ~2M cycles = ~1ms at 2 GHz
```

**Score GEMM Tile Configuration:**

```
For score GEMM (Q × K^T), adapt M2 kernel pattern:

Tile dimensions:
  M_tile = 2 (process 2 query rows simultaneously, like M2 kernel)
  N_tile = 64 (64 KV positions per N-block, like M2 kernel)
  K_tile = 32 (one Q8_1 block)

Register usage per tile (32 ZMM registers):
  Accumulators: 8 ZMM (2 rows × 4 vectors × 16 floats = 128 outputs)
  INT32 temps: 8 ZMM (for VNNI accumulation before scale)
  Q data: 2 ZMM (one per row, broadcast for all KV)
  K data: 4 ZMM (64 KV positions = 4 × 16)
  Constants: 6 ZMM (scales, 128, -128, etc.)
  Temps: 4 ZMM (scale application, correction)
  Total: 32 ZMM ✓
```

**K Packing Strategy:**

```cpp
/**
 * K needs to be packed into VNNI-optimal layout.
 * Follow QuantisedGemmKernel::pack_weights_generic() pattern.
 * 
 * Original K: [seq_len, head_dim] Q8_1 blocks
 * Packed K:   [N_blocks][K_blocks][64][4] INT8 + auxiliary arrays
 * 
 * Auxiliary:
 *   compensation[K_blocks][N_padded]: Σ K_q8[k, n] per K-block, per column
 *   scales[K_blocks][N_padded]: FP32 scales from Q8_1 blocks
 * 
 * Packing is done once per attention call (K doesn't change during attention).
 */
```

### 4.2 Context GEMM: P × V (Full VNNI)

**Both P and V are Q8_1 for maximum VNNI throughput:**

```cpp
/**
 * Compute C[M, D] = P[M, N] × V[N, D] using Q8_1 × Q8_1 VNNI
 * 
 * P: Q8_1 quantized attention weights [seq_len, seq_len]
 *    - Per-row scale (computed during softmax+quantize fusion)
 *    - Values in [0, 127] range (softmax outputs are non-negative)
 * V: Q8_1 quantized values [seq_len, head_dim]
 *    - Per-block scale (original format)
 * C: FP32 context output [seq_len, head_dim]
 * 
 * This follows the same VNNI pattern as score GEMM.
 */
```

**P Storage Layout (Optimized Q8_1):**

```cpp
/**
 * P has special properties that simplify storage:
 * 1. Values are in [0, 1] after softmax → quantized to [0, 127]
 * 2. Per-ROW scale (not per-block) since softmax is row-wise
 * 3. No sum_qs needed for symmetric [0, 127] range
 * 
 * Layout:
 *   P_scales[seq_len]: FP32 per-row scales
 *   P_q8[seq_len][seq_len]: UINT8 quantized values (packed contiguously)
 * 
 * Memory: seq_len × seq_len × 1 + seq_len × 4 = N² + 4N bytes
 * For seq=128: 16 KB + 512 B = 16.5 KB (vs 64 KB for FP32)
 */
```

**Context GEMM Inner Loop:**

```cpp
/**
 * Context GEMM reuses the M2 kernel pattern but with P as "activations".
 * 
 * Key difference: P is already UINT8 [0, 127], so no INT8→UINT8 conversion.
 * This simplifies the compensation term:
 *   acc_true = acc_raw - 0 × sum_qs_V = acc_raw  (no compensation if P is unsigned)
 * 
 * However, we still need sum_qs_P for the V compensation:
 *   acc_raw = Σ P × (V + 128) = Σ P × V + 128 × Σ P
 *   acc_true = acc_raw - 128 × sum_qs_P
 * 
 * Wait, V is signed INT8, so we convert V→UINT8 and compensate with sum_qs_P.
 */

// Inner loop structure (similar to score GEMM):
for (int i = 0; i < 8; ++i) {  // 8 iters × 4 elements = 32 per V-block
    // Load P values (already UINT8, no conversion needed)
    // P is row-major: P[q, n:n+4] for query q
    vpbroadcastd(zmm_p0, ptr[reg_P_cursor + i * 4]);
    vpbroadcastd(zmm_p1, ptr[reg_P_cursor + P_row_stride + i * 4]);
    
    // Load V values and convert INT8→UINT8
    vmovups(zmm_v0, ptr[reg_V_cursor + 0 * 64]);
    vmovups(zmm_v1, ptr[reg_V_cursor + 1 * 64]);
    // ...
    vpxord(zmm_v0, zmm_v0, zmm_128);  // V + 128
    vpxord(zmm_v1, zmm_v1, zmm_128);
    
    // VNNI for context rows 0 and 1
    vpdpbusd(zmm_acc0_0, zmm_p0, zmm_v0);  // row 0, head_dim [0:16]
    vpdpbusd(zmm_acc0_1, zmm_p0, zmm_v1);  // row 0, head_dim [16:32]
    vpdpbusd(zmm_acc1_0, zmm_p1, zmm_v0);  // row 1, head_dim [0:16]
    vpdpbusd(zmm_acc1_1, zmm_p1, zmm_v1);  // row 1, head_dim [16:32]
    
    add(reg_V_cursor, 256);
}

// Compensation: acc -= 128 × row_sum_P
// Note: row_sum_P = Σ P_q8[q, :] is computed during softmax+quantize phase
vpbroadcastd(zmm_row_sum_p0, ptr[reg_P_row_sums + q0 * 4]);
vpbroadcastd(zmm_row_sum_p1, ptr[reg_P_row_sums + q1 * 4]);
vpmulld(zmm_comp0, zmm_row_sum_p0, zmm_neg_128);
vpmulld(zmm_comp1, zmm_row_sum_p1, zmm_neg_128);
vpaddd(zmm_acc0_0, zmm_acc0_0, zmm_comp0);  // Apply compensation
// ...
```

**Scale Correction for P×V:**

```cpp
/**
 * Final scale: context = acc × scale_P[row] × scale_V[col]
 * 
 * Unlike score GEMM where both Q and K have per-block scales,
 * P has per-ROW scale and V has per-block scale.
 */

// After N-loop (all V positions accumulated):
vcvtdq2ps(zmm_acc0_0, zmm_acc0_0);  // INT32 → FP32

// Load V scales (FP32, 4 per head_dim block)
vmovups(zmm_scale_v0, ptr[reg_V_scales + d_block * 64 + 0 * 64]);
vmovups(zmm_scale_v1, ptr[reg_V_scales + d_block * 64 + 1 * 64]);

// Load P scale (per-row, broadcast)
vbroadcastss(zmm_scale_p0, ptr[reg_P_scales + q0 * 4]);
vbroadcastss(zmm_scale_p1, ptr[reg_P_scales + q1 * 4]);

// Apply scales
vmulps(zmm_acc0_0, zmm_acc0_0, zmm_scale_v0);  // × scale_V
vmulps(zmm_acc0_0, zmm_acc0_0, zmm_scale_p0);  // × scale_P
```

### 4.3 Fused Softmax + P Quantization

**Fuse softmax output directly into Q8_1 format:**

```cpp
/**
 * Compute softmax and quantize to UINT8 in one pass per row.
 * 
 * For each row q:
 *   1. Find max(S[q, :]) - vectorized horizontal max
 *   2. Compute exp(S[q, k] - max) for all k - JitFastExpEmitter
 *   3. Compute sum = Σ exp(...) - vectorized horizontal sum
 *   4. P_q8[q, k] = round(127.0f * exp(...) / sum) - fused normalize+quantize
 *   5. scale_P[q] = sum / 127.0f - for dequant during P×V
 *   6. row_sum_P[q] = Σ P_q8[q, k] - for VNNI compensation
 * 
 * Key insight: We can fuse steps 4-5 because max(softmax) = exp(0)/sum = 1/sum
 *   P_q8[q, k] = round(127.0f * exp(S[q,k] - max) / sum)
 */
```

**FP32→Q8_1 Quantization Reference (from SIMDHelpers.h):**

```cpp
/**
 * Our existing simd::quantize_single_block_avx512() shows the pattern:
 * 
 * 1. Load 32 floats (2 × zmm)
 * 2. Compute max_abs via _mm512_reduce_max_ps
 * 3. Compute scale = max_abs / 127.0f
 * 4. Scale values: scaled = v * (127.0f / max_abs)
 * 5. Round and convert: _mm512_cvtps_epi32 (rounds to nearest-even)
 * 6. Pack to int8: _mm512_cvtsepi32_epi8 (saturates to [-128, 127])
 * 7. Compute sum_qs from packed values (for consistency)
 * 
 * For P quantization, we adapt this:
 * - Values are already in [0, 1] from softmax
 * - Use per-row scale instead of per-block
 * - Output is UINT8 [0, 127] instead of INT8 [-128, 127]
 */
```

**Vectorized Softmax + Quantize:**

```cpp
void emit_softmax_quantize_row(int row_idx) {
    // Score row starts at S + row_idx * seq_len * 4
    mov(reg_score_row, reg_S_base);
    imul(rax, row_idx, seq_len * 4);
    add(reg_score_row, rax);
    
    // =========================================================
    // PASS 1: Find row maximum (vectorized horizontal max)
    // =========================================================
    // From QuantisedGemmJit_M2 softmax implementation:
    vmovups(zmm_max, ptr[reg_score_row + 0 * 64]);
    for (int i = 1; i < seq_len / 16; ++i) {
        vmaxps(zmm_max, zmm_max, ptr[reg_score_row + i * 64]);
    }
    // Horizontal reduction: max across 16 lanes
    vshuff32x4(zmm_tmp, zmm_max, zmm_max, 0x4E);  // Swap halves
    vmaxps(zmm_max, zmm_max, zmm_tmp);
    vshuff32x4(zmm_tmp, zmm_max, zmm_max, 0xB1);  // Swap 128-bit lanes
    vmaxps(zmm_max, zmm_max, zmm_tmp);
    vshufps(zmm_tmp, zmm_max, zmm_max, 0x4E);     // Swap 64-bit pairs
    vmaxps(zmm_max, zmm_max, zmm_tmp);
    vshufps(zmm_tmp, zmm_max, zmm_max, 0xB1);     // Swap 32-bit pairs
    vmaxps(zmm_max, zmm_max, zmm_tmp);
    // Now zmm_max[0] = row_max, broadcast it
    vbroadcastss(zmm_max, xmm_max);
    
    // =========================================================
    // PASS 2: exp(x - max) and sum
    // =========================================================
    // Use JitFastExpEmitter pattern from existing kernels
    vxorps(zmm_sum, zmm_sum, zmm_sum);  // Running sum
    
    for (int i = 0; i < seq_len / 16; ++i) {
        vmovups(zmm_score, ptr[reg_score_row + i * 64]);
        vsubps(zmm_x, zmm_score, zmm_max);           // x = score - max
        
        // Fast exp polynomial (from QuantisedGemmJit_M2.h):
        // exp(x) ≈ 2^n * P(f) where n = floor(x/ln2), f = frac(x/ln2)
        vmulps(zmm_fx, zmm_x, zmm_inv_ln2);          // x / ln(2)
        vrndscaleps(zmm_n, zmm_fx, 0x01);            // n = floor(...)
        vsubps(zmm_f, zmm_fx, zmm_n);                // f = fractional part
        
        // Polynomial: P(f) = c1 + f*(c2 + f*(c3 + f*(c4 + f*c5)))
        vmovaps(zmm_p, zmm_c5);
        vfmadd213ps(zmm_p, zmm_f, zmm_c4);
        vfmadd213ps(zmm_p, zmm_f, zmm_c3);
        vfmadd213ps(zmm_p, zmm_f, zmm_c2);
        vfmadd213ps(zmm_p, zmm_f, zmm_c1);
        
        // 2^n via IEEE754 exponent manipulation
        vcvtps2dq(zmm_ni, zmm_n);
        vpaddd(zmm_ni, zmm_ni, zmm_127);             // Add bias
        vpslld(zmm_2n, zmm_ni, 23);                  // Shift to exponent
        vmulps(zmm_exp, zmm_p, zmm_2n);              // P(f) * 2^n
        
        vmovups(ptr[reg_exp_row + i * 64], zmm_exp); // Store for pass 3
        vaddps(zmm_sum, zmm_sum, zmm_exp);           // Accumulate sum
    }
    
    // Horizontal sum reduction
    // ... (similar pattern to max reduction)
    vbroadcastss(zmm_sum, xmm_sum);
    
    // =========================================================
    // PASS 3: Normalize and quantize to UINT8
    // =========================================================
    // Compute inv_sum = 127.0f / sum
    vdivps(zmm_inv_sum, zmm_127, zmm_sum);
    
    // Store per-row scale for dequant: scale_P = sum / 127.0f
    vrcpps(zmm_scale_p, zmm_inv_sum);  // Approximate 1/(127/sum) = sum/127
    vmovss(ptr[reg_P_scales + row_idx * 4], xmm_scale_p);
    
    vxorps(zmm_row_sum, zmm_row_sum, zmm_row_sum);  // For compensation
    
    for (int i = 0; i < seq_len / 16; ++i) {
        vmovups(zmm_exp, ptr[reg_exp_row + i * 64]);
        
        // P_fp = exp * (127 / sum)
        vmulps(zmm_p_fp, zmm_exp, zmm_inv_sum);
        
        // Round to nearest and convert to UINT8
        vrndscaleps(zmm_p_fp, zmm_p_fp, 0);         // Round to nearest
        vcvtps2dq(zmm_p_i32, zmm_p_fp);             // FP32 → INT32
        vpmovusdb(xmm_p_u8, zmm_p_i32);             // INT32 → UINT8 (saturate)
        
        // Store quantized P (16 UINT8 values)
        vmovups(ptr[reg_P_q8 + row_idx * seq_len + i * 16], xmm_p_u8);
        
        // Accumulate row sum for compensation
        vpaddd(zmm_row_sum, zmm_row_sum, zmm_p_i32);
    }
    
    // Store row sum (horizontal sum of zmm_row_sum)
    // ... horizontal reduction ...
    mov(ptr[reg_P_row_sums + row_idx * 4], eax);
}
```

**Causal Mask Integration:**

```cpp
/**
 * Apply causal mask BEFORE softmax by setting masked positions to -inf.
 * This ensures exp(-inf) = 0 and they don't contribute to the sum.
 * 
 * From QuantisedGemmJit_M2 mask application pattern:
 */

// After loading scores, before exp:
mov(rdx, ptr[params + offset_mask]);
test(rdx, rdx);
jz(skip_mask);
{
    // Mask layout: mask[q, k] where mask[q, k] = -inf if k > q + pos_offset
    lea(rcx, ptr[rdx + row_idx * seq_len * 4]);
    vmovups(zmm_mask, ptr[rcx + i * 64]);
    vaddps(zmm_score, zmm_score, zmm_mask);  // score += mask (adds -inf)
}
skip_mask:
```

### 4.4 Tile Size Selection and Register Blocking

**L1/L2 Cache Optimization (Based on QuantisedGemmKernel Patterns):**

```
L1 Data Cache: 32-48 KB typical
L2 Cache: 256 KB - 1 MB typical (detected via cpu_l2_cache_size())

From QuantisedGemmKernel cache-aware blocking (lines 782-830):
  - L2 constraint: Block B data should fit in L2 × gemm_l2_limit_pct (0.6)
  - L3 constraint: All threads' blocks fit in L3 × gemm_l3_share_pct (0.7)
  - Dynamic block size: max_n_block = block_size_limit / k
```

**Score GEMM (Q×K^T) Working Set Analysis:**

```
Per tile iteration (M_tile=2, N_tile=64, K_tile=32):
  Q data: 2 rows × 36 bytes = 72 bytes (one Q8_1 block per row)
  K data: 64 positions × 4 bytes × 4 = 1024 bytes (VNNI-packed K)
  Compensation: 64 × 4 = 256 bytes (INT32)
  K scales: 64 × 4 = 256 bytes (FP32)
  Accumulators: 8 ZMM registers (in registers, not memory)
  Total memory traffic per K-block iteration: ~1.6 KB

Full K-dimension (head_dim=128 → 4 K-blocks):
  Total K data per 64 KV positions: 4 × 1024 = 4 KB
  This fits comfortably in L1 cache (32 KB)
```

**Context GEMM (P×V) Working Set Analysis:**

```
Per tile iteration (M_tile=2, N_tile=32, D_tile=64):
  P data: 2 rows × 32 bytes = 64 bytes (UINT8)
  V data: 32 positions × 64 floats equivalent = 2 KB (packed Q8_1)
  V scales: 64 × 4 = 256 bytes
  Accumulators: 4 ZMM registers per row (in registers)
  Total per N-tile: ~2.3 KB

Full N-dimension (seq_len=128 → 4 N-tiles):
  V data reused across rows: stays in L1/L2
```

**Recommended Tile Sizes (Matched to M2 Kernel Pattern):**

| GEMM | M_tile | N_tile | K/D_tile | Notes |
|------|--------|--------|----------|-------|
| Q×K^T | 2 | 64 | 32 | Match M2's 2-row, 64-col pattern |
| P×V | 2 | 32 | 64 | V is row-major, tile N (KV positions) |

**Register Blocking Strategy (32 ZMM Total):**

```
Score GEMM registers:
  zmm0-3:   Row 0 FP32 accumulators (4 × 16 = 64 outputs)
  zmm4-7:   Row 1 FP32 accumulators
  zmm8-11:  Row 0 INT32 temp accumulators (for VNNI)
  zmm12-15: Row 1 INT32 temp accumulators
  zmm16-19: K data (4 blocks × 16 positions)
  zmm20-21: Q data (broadcast for rows 0, 1)
  zmm22:    Q scale broadcast
  zmm23:    128 constant (INT8→UINT8)
  zmm24:    -128 constant (compensation)
  zmm25:    Temp for horizontal reduction
  zmm26-29: K scales (4 × 16)
  zmm30-31: Temps for scale application
  Total: 32 ZMM ✓

Context GEMM registers:
  Similar allocation, but P replaces Q and V replaces K
```

---

## 5. Softmax Implementation

### 5.1 Fused Scale + Mask + Softmax

```cpp
void emit_scaled_masked_softmax() {
    // For each row q in [0, N):
    //   1. Apply scale: S[q, :] *= 1/sqrt(head_dim)
    //   2. Apply causal mask: S[q, k] = -inf for k > q + position_offset
    //   3. Compute row max: max_val = max(S[q, :])
    //   4. Compute exp sum: sum = Σ exp(S[q, k] - max_val)
    //   5. Normalize: P[q, k] = exp(S[q, k] - max_val) / sum
    
    const int vec_width = 16;  // AVX-512: 16 floats
    
    for (int q = 0; q < N; ++q) {
        // Vectorized row processing
        // Load 16 elements at a time, apply scale, mask, find max
        // Second pass: exp and sum
        // Third pass: normalize
    }
}
```

### 5.2 Vectorized Exp

Use the existing `JitFastExpEmitter` for vectorized exp:

```cpp
// Process 16 floats at once
vsubps(zmm_x, zmm_score, zmm_max);  // x = score - max
emit_fast_exp(zmm_x, zmm_result);    // result = exp(x)
vaddps(zmm_sum, zmm_sum, zmm_result); // sum += result
```

### 5.3 Causal Mask Handling

**Compile-time mask generation** for fixed sequence lengths:

```cpp
void emit_causal_mask_apply(int row_idx) {
    // For row q, mask out positions k > q + position_offset
    int mask_start = row_idx + position_offset + 1;
    
    // Generate blend mask: 1s for valid, 0s for masked
    // Use vblendmps with mask register
    for (int k = 0; k < N; k += 16) {
        uint16_t mask = compute_mask(k, mask_start);
        mov(eax, mask);
        kmovw(k1, eax);
        vblendmps(zmm_score | k1, zmm_neginf, zmm_score);
    }
}
```

---

## 6. Performance Projections

### 6.1 FLOP Analysis

For seq_len=128, head_dim=128, num_heads=28:

| Operation | Ops per Head | Total Ops | Domain |
|-----------|--------------|-----------|--------|
| Q×K^T VNNI | 128×128×128 int8 MACs = 2.1M | 58M | INT8 (VNNI) |
| Scale correction | 128×128×~6 = 98K | 2.7M | FP32 |
| Softmax + P quant | ~5×128×128 = 82K | 2.3M | FP32 |
| P×V VNNI | 128×128×128 int8 MACs = 2.1M | 58M | INT8 (VNNI) |
| Context scale corr | 128×128×~6 = 98K | 2.7M | FP32 |
| **Total** | ~4.5M | **124M** | Full VNNI |

**Full VNNI Advantage:**

```
Both GEMMs now use INT8 VNNI:
  Q×K^T: 2.1M int8 MACs @ 4× throughput
  P×V:   2.1M int8 MACs @ 4× throughput
  
vs Mixed precision (P×V with FP32):
  Q×K^T: 2.1M int8 MACs @ 4× throughput  
  P×V:   2.1M FP32 FMAs @ 1× throughput  ← 4× slower

Effective speedup on context GEMM: ~4×
Overall attention speedup vs mixed: ~2×
```

### 6.2 Expected Throughput

| Implementation | Est. Peak | Prefill Time (128 tok) |
|----------------|-----------|------------------------|
| Current streaming | ~5-10 GFLOPS equiv | 500 ms |
| Full VNNI (both GEMMs) | ~400 GOPS int8 | - |
| **Combined** | - | **25-50 ms** |

**Speedup: 10-20×** from proper GEMM structure + full VNNI for both GEMMs.

### 6.3 Memory Bandwidth

Score matrix read/write per head:
```
Write S: 128×128×4 = 64 KB
Read S (softmax): 64 KB  
Write P: 64 KB
Read P (context GEMM): 64 KB
Total: 256 KB per head
```

At 50 GB/s memory bandwidth: 256 KB / 50 GB/s = 5 μs (negligible vs compute).

---

## 7. Integration Plan

### 7.1 Phase 1: VNNI Score GEMM (1 day)

**Tasks:**

1. Create `JitVnniPrefillAttention` class inheriting from `Xbyak::CodeGenerator`
   - Follow `QuantisedGemmJit_M2` structure (prologue, N-loop, K-loop, epilogue)
   - Stack frame layout for score buffer (see Section 3.4)

2. Implement K packing using existing infrastructure:
   ```cpp
   // Reuse QuantisedGemmKernel::pack_weights_generic() pattern
   // Pack K[seq_len, head_dim] Q8_1 blocks into VNNI-optimal layout:
   //   packed_K[N_blocks][K_blocks][64][4] INT8
   //   compensation[K_blocks][N_padded] INT32
   //   scales[K_blocks][N_padded] FP32
   
   // Can either:
   // a) Create temporary QuantisedPackedWeights for K at attention call time
   // b) Add dedicated pack_k_for_attention() function
   ```

3. Implement VNNI-based Q×K^T GEMM:
   - Adapt M2 inner loop structure (lines 600-750 of QuantisedGemmJit_M2.h)
   - Q replaces A (activations), K replaces B (weights)
   - 8 VNNI iterations per K-block, 4 elements each

4. Implement Q8_1 scale correction:
   - INT32→FP32 conversion (`vcvtdq2ps`)
   - Load K scales, Q scale, apply via `vmulps`
   - Compensation: `acc -= 128 × sum_qs_K` (precomputed during K pack)

5. Validation: Compare score matrix against reference implementation
   ```cpp
   // Reference: simd::dequant_row() + cblas_sgemm()
   float* Q_fp32 = dequant(Q_q8_1);
   float* K_fp32 = dequant(K_q8_1);
   cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
               seq_len, seq_len, head_dim,
               1.0f, Q_fp32, head_dim, K_fp32, head_dim,
               0.0f, S_ref, seq_len);
   // Compare S_vnni vs S_ref within tolerance
   ```

### 7.2 Phase 2: Fused Softmax + P Quantization (1 day)

**Tasks:**

1. Implement vectorized softmax using existing pattern:
   - Copy polynomial coefficients from `QuantisedGemmJit_M2.h` (lines 830-860)
   - 3-pass algorithm: max → exp+sum → normalize
   - Use `JitFastExpEmitter` polynomial: `exp(x) ≈ 2^n × P(f)`

2. Fuse P quantization with softmax output:
   ```cpp
   // In pass 3, instead of storing FP32 probabilities:
   // P_q8[q, k] = round(127.0f * exp(S[q,k] - max) / sum)
   // scale_P[q] = sum / 127.0f
   // row_sum_P[q] = Σ P_q8[q, k]  // For VNNI compensation
   
   // Use vpmovusdb for INT32→UINT8 pack with saturation
   vcvtps2dq(zmm_p_i32, zmm_p_scaled);
   vpmovusdb(xmm_p_u8, zmm_p_i32);  // 16 INT32 → 16 UINT8
   ```

3. Storage layout for P (optimized for context GEMM):
   ```cpp
   // P is row-major UINT8 [seq_len, seq_len]
   // P_scales is per-row FP32 [seq_len]
   // P_row_sums is per-row INT32 [seq_len] (for VNNI compensation)
   
   // Total: N² + N×4 + N×4 = N² + 8N bytes
   // For N=128: 16 KB + 1 KB = 17 KB
   ```

4. Add causal mask application before exp:
   - Load mask vector (pre-filled with -inf for masked positions)
   - `vaddps(zmm_score, zmm_score, zmm_mask)` before exp computation
   - Masked positions become exp(-inf) = 0

5. Validation: Compare P_q8 dequantized vs FP32 softmax
   ```cpp
   // Reference: FP32 softmax
   softmax_fp32(S, P_ref, seq_len, seq_len);
   
   // Dequant P_q8 and compare
   float P_dequant[q] = P_q8[q] * scale_P[row];
   assert(abs(P_dequant - P_ref) < tolerance);
   ```

### 7.3 Phase 3: VNNI Context GEMM (1 day)

**Tasks:**

1. Implement V packing (similar to K packing):
   ```cpp
   // V[seq_len, head_dim] Q8_1 → packed_V[D_blocks][N_blocks][64][4]
   // Note: V layout differs from K (V is transposed for P×V)
   
   // Auxiliary arrays:
   //   V_compensation[N_blocks][D_padded]: Σ V_q8[n, d] per N-block
   //   V_scales[N_blocks][D_padded]: FP32 scales
   ```

2. Implement VNNI-based P×V GEMM:
   - P is UINT8 (already unsigned from softmax quantization)
   - V is INT8 → convert to UINT8 via XOR 0x80
   - Compensation uses P_row_sums: `acc -= 128 × P_row_sums[q]`

3. Scale correction for P×V:
   ```cpp
   // context = acc × scale_P[row] × scale_V[block]
   // Note: scale_P is per-row (broadcast), scale_V is per-D-block (vectorized)
   
   vcvtdq2ps(zmm_acc, zmm_acc);
   vbroadcastss(zmm_scale_p, ptr[reg_P_scales + row * 4]);
   vmovups(zmm_scale_v, ptr[reg_V_scales + d_block * 64]);
   vmulps(zmm_acc, zmm_acc, zmm_scale_p);
   vmulps(zmm_acc, zmm_acc, zmm_scale_v);
   ```

4. Output to context buffer in FP32

5. End-to-end validation:
   ```cpp
   // Reference: FP32 attention
   float* context_ref = softmax(Q_fp32 @ K_fp32.T) @ V_fp32;
   
   // Compare VNNI context vs reference
   // Tolerance: ~1e-3 relative error (quantization noise)
   for (int i = 0; i < seq_len * head_dim; ++i) {
       float rel_err = abs(context_vnni[i] - context_ref[i]) / abs(context_ref[i]);
       assert(rel_err < 1e-3);
   }
   ```

### 7.4 Phase 4: Optimization + Integration (1-2 days)

**Optimization Tasks:**

1. Profile hotspots using `LLAMINAR_PROFILE_KERNELS`:
   ```bash
   LLAMINAR_PROFILE_KERNELS=1 ./run_llaminar.sh --benchmark -m model.gguf
   ```
   
2. Tune tile sizes based on L2 cache:
   ```cpp
   // Use cpu_l2_cache_size() from CPUFeatures.h
   // Adjust N_tile and K_tile to maximize cache reuse
   // Target: score tile fits in 60% of L2 (matches QuantisedGemmKernel pattern)
   ```

3. Add software prefetching (matches M2 pattern):
   ```cpp
   // In K-loop: prefetch next K data
   prefetcht0(ptr[reg_K_cursor + 1024]);  // 4 iters ahead
   prefetcht0(ptr[reg_Q_cursor + 144]);   // Next Q block
   ```

4. Consider OpenMP parallelization for multi-head:
   ```cpp
   // Parallelize across heads (independent attention per head)
   #pragma omp parallel for
   for (int h = 0; h < num_heads; ++h) {
       prefill_attention_single_head(Q + h*stride, K + h*stride, V + h*stride, ...);
   }
   ```

**Integration Tasks:**

1. Add tiled variant for long sequences (seq > 256):
   - Implement online softmax (Flash-Attention style)
   - Tile both Q and KV dimensions
   - Track running max/sum across tiles

2. Add kernel selection logic in `JitAttentionKernelCache`:
   ```cpp
   // In GQAAttention::compute() or similar:
   if (seq_len_q == 1) {
       return use_decode_kernel();  // Current JitFusedAttentionWo
   } else if (seq_len_q <= 256) {
       return use_vnni_prefill_small();  // New JitVnniPrefillAttention
   } else {
       return use_vnni_prefill_tiled();  // Tiled variant
   }
   ```

3. Update `AttentionMode::AUTO` selection:
   ```cpp
   // In JitAttentionKernelCache::getKernel():
   AttentionMode effective_mode = config.mode;
   if (effective_mode == AttentionMode::AUTO) {
       if (config.seq_len_q == 1) {
           effective_mode = AttentionMode::DECODE;
       } else {
           effective_mode = AttentionMode::PREFILL;  // New GEMM kernel
       }
   }
   ```

4. Full test suite validation:
   - Unit tests: Score GEMM, softmax, context GEMM individually
   - Integration tests: Full attention against PyTorch reference
   - E2E tests: Token prediction parity with existing kernel

---

## 8. API Design

### 8.1 Kernel Selection

```cpp
enum class AttentionMode {
    DECODE,       // Streaming (current JitFusedAttentionWo)
    PREFILL,      // GEMM-based (new JitGemmPrefillAttention)
    PREFILL_TILED, // Tiled GEMM for long sequences
    AUTO          // Select based on seq_len_q
};

// In JitAttentionKernelCache::getKernel():
if (config.effectiveMode() == AttentionMode::DECODE) {
    return generate_decode_kernel(config);
} else if (config.seq_len_q <= 256) {
    return generate_gemm_prefill_small(config);
} else {
    return generate_gemm_prefill_tiled(config);
}
```

### 8.2 New Kernel Interface

```cpp
/**
 * @brief VNNI-based prefill attention kernel
 * 
 * Computes: Output = softmax(Q × K^T / sqrt(d)) × V × Wo
 * 
 * Uses AVX-512 VNNI for Q×K^T score computation (Q8_1 integer domain).
 * Softmax in FP32, context GEMM as mixed-precision FP32 × Q8_1.
 * 
 * Unlike decode kernel, this materializes the N×N attention score matrix
 * and uses batched GEMM operations instead of streaming dot products.
 */
using JitVnniPrefillKernelFn = void (*)(
    const void* Q,       // [seq_len, num_heads, head_dim] Q8_1 blocks
    const void* K,       // [seq_len, num_kv_heads, head_dim] Q8_1 blocks
    const void* V,       // [seq_len, num_kv_heads, head_dim] Q8_1 blocks
    const void* Wo,      // [d_model, d_model] Q8_1 or FP32
    float* output,       // [seq_len, d_model] FP32
    float* score_buffer, // [seq_len, seq_len] FP32 scratch (can be stack-allocated for small N)
    int seq_len,
    float scale,         // 1/sqrt(head_dim)
    int position_offset  // For causal mask alignment
);
```

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| VNNI scale correction precision | Accuracy | Use FP32 for all scale arithmetic, validate vs reference |
| P quantization precision loss | Accuracy | 127 levels sufficient for softmax; validate KL divergence |
| INT32 overflow in long sequences | Correctness | Monitor accumulator range, split K-blocks if needed |
| Memory pressure for long seq | OOM | Tiled implementation, P stored as Q8_1 (1/4 FP32 size) |
| Correctness regression | High | Extensive parity tests vs streaming kernel |
| Causal mask branching | Performance | Compile-time mask generation, vectorized blend |

---

## 10. Success Criteria

1. **Performance:** ≥10× prefill speedup for seq_len=128
2. **Correctness:** Match streaming kernel output within FP32 tolerance
3. **Memory:** Peak memory ≤ current (P in Q8_1 saves 3× vs FP32)
4. **VNNI utilization:** >80% VNNI throughput on both GEMMs
5. **P quantization:** KL divergence < 0.001 vs FP32 softmax

---

## 11. Appendix: Reference Implementations

### A. PyTorch Reference

```python
def attention_prefill(Q, K, V, scale, causal=True):
    """
    Q, K, V: [batch, heads, seq_len, head_dim]
    """
    # Score computation (GEMM 1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Causal mask
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    
    # Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Context computation (GEMM 2)
    context = torch.matmul(attn_weights, V)
    
    return context
```

### B. llama.cpp FA Implementation

See `ggml/src/ggml-cpu/ggml-cpu-aarch64.cpp` for reference tiled attention with NEON.

### C. Flash Attention Paper

Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
- Tiling strategy for GPU, adaptable to CPU L2 cache
- Online softmax for memory efficiency
