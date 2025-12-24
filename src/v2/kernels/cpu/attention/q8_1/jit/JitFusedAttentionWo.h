/**
 * @file JitFusedAttentionWo.h
 * @brief Composed JIT kernel: Fused Attention + Wo projection
 * @author David Sanftenberg
 * @date December 2025
 *
 * This is the composed JIT kernel that uses the modular JIT microkernels:
 *   - JitQ8DotProduct (μK1): Q*K^T attention score
 *   - JitOnlineSoftmax (μK2): Online softmax state management
 *   - JitVWeightedAccum (μK3): Weighted V accumulation
 *   - JitWoProjection (μK4): Output projection
 *   - JitFastExp (μK5): Fast exponential approximation
 *
 * The composed kernel generates optimized code for the entire attention
 * computation, avoiding function call overhead while maintaining the same
 * structure as the reference implementation for testability.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * ARCHITECTURE
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * FA1 (Single-Score) Mode:
 *   For each query position q:
 *     For each KV position kv:
 *       score = Q8DotProduct(Q[q], K[kv]) * scale
 *       weight = OnlineSoftmax.update(score)
 *       context[q] += weight * V[kv]
 *     context[q] *= 1/sum
 *     output[q] = context[q] * Wo
 *
 * FA2 (4x Tiled) Mode:
 *   For each query position q:
 *     For each KV tile (kv, kv+1, kv+2, kv+3):
 *       scores[0..3] = Q8DotProduct_4x(Q[q], K[kv..kv+3]) * scale
 *       tile_max = max(scores[0..3])
 *       if tile_max > running_max:
 *         correction = exp(running_max - tile_max)
 *         context[q] *= correction        // Rescale accumulated context
 *         running_max = tile_max
 *       weights[0..3] = exp(scores[0..3] - running_max)
 *       context[q] += weights[0] * V[kv+0]
 *       context[q] += weights[1] * V[kv+1]
 *       context[q] += weights[2] * V[kv+2]
 *       context[q] += weights[3] * V[kv+3]
 *     context[q] *= 1/sum
 *     output[q] = context[q] * Wo
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * REGISTER ALLOCATION ZONES (via RegisterAllocation.h)
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * AccumZone (zmm0-7): Context accumulators
 *   - zmm0-7: Up to 512 floats (128 head_dim × 4 tile, or 64 head_dim × 8)
 *   - Layout: [context_chunk_0, context_chunk_1, ...]
 *   - For 7B (head_dim=128): zmm0-7 = 8 chunks × 16 floats = 128 floats
 *
 * InputZone (zmm8-15): Q/K/V data loading
 *   - zmm8-9: Currently free (V accum uses these for output)
 *   - zmm10-13: Pre-loaded unsigned Q data (persists across KV loop)
 *   - zmm14-15: Used by emit_exp2_poly (safe - exp called before V accum)
 *
 * StateZone (zmm16-19): Online softmax state
 *   - StateMax (zmm16): Running max for numerical stability
 *   - StateSum (zmm17): Running sum of exp weights
 *   - StateWeight (zmm18): Current attention weight (FA1 only)
 *   - StateCorr (zmm19): Correction factor for context rescaling
 *
 * ScratchZone (zmm20-25): Temporary registers + FA2 scores
 *   - Score0-3 (xmm20-23): FA2 tile scores (alias Scratch0-3 low 128-bits)
 *   - zmm20: CRITICAL - Holds broadcasted weight for V accumulation!
 *   - zmm21-23: V weight broadcast, temp scratch
 *   - Scratch4 (zmm24): Safe scratch - tile_max storage
 *   - Scratch5 (zmm25): Safe scratch - correction term
 *
 * ConstZone (zmm26-31): Preloaded constants
 *   - zmm26: log2(e) for exp approximation
 *   - zmm27: -87.0f exp min clamp
 *   - zmm28: 1.0f
 *   - zmm29: 0x80808080 (Q8 unsigned bias)
 *   - zmm30: -infinity
 *   - zmm31: Reserved
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * FA2 REGISTER LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Phase 1 - Score Computation:
 *   [Input]  Q data in zmm10-13 (persists)
 *   [Input]  K data loaded into ymm8 (per-KV, temporary)
 *   [Output] Scores in Score0-3 (xmm20-23)
 *
 * Phase 2 - Tile State Update:
 *   [Input]  Scores in Score0-3 (xmm20-23)
 *   [Output] tile_max in Scratch4 (zmm24)
 *   [Output] StateCorr in zmm19 (if rescale needed)
 *   [Clobber] Score0-3 overwritten with weights via Scratch0-3
 *
 * Phase 3 - Context Rescaling (if correction != 1.0):
 *   [Input]  StateCorr in zmm19
 *   [Modify] Accum0-7 (zmm0-7) *= StateCorr
 *
 * Phase 4 - V Accumulation:
 *   [Input]  Weights in Scratch0-3 (zmm20-23) - broadcasted from xmm
 *   [Input]  V data loaded from memory
 *   [Modify] Accum0-7 (zmm0-7) += weight * V_chunk
 *   [CRITICAL] zmm20 holds broadcasted weight - cannot use as scratch!
 */

#pragma once

#include "JitMicrokernelBase.h"
#include "JitQ8DotProduct.h"
#include "JitOnlineSoftmax.h"
#include "JitVWeightedAccum.h"
#include "JitWoProjection.h"
#include "JitWoProjectionOptimized.h"
#include "JitFastExp.h"
#include "../../../jit/RegisterAllocation.h" // Typed register allocation for FA2

#include <memory>
#include <cstdint>
#include <unordered_map>
#include <mutex>
#include <functional>

namespace llaminar::v2::kernels::jit
{
    // Import typed register aliases for FA2 tile processing
    using namespace llaminar2::jit;

    /**
     * @brief Attention computation mode
     *
     * DECODE: Optimized for single-token generation (seq_len_q = 1)
     *   - Loop order: Q → H → KV (current implementation)
     *   - K/V cache loaded once per head per query
     *   - Minimal memory overhead
     *
     * PREFILL: Optimized for multi-token processing (seq_len_q > 1)
     *   - Loop order: Q_tile → H → KV → Q_inner (K/V cache reuse within tile)
     *   - K/V cache loaded once per head per KV position, reused across queries in tile
     *   - Higher memory overhead for per-query softmax state and context
     *   - Better throughput for prefill workloads
     */
    enum class AttentionMode
    {
        DECODE,  // seq_len_q == 1, optimize for latency
        PREFILL, // seq_len_q > 1, optimize for K/V cache reuse
        AUTO     // Automatically select based on batch_size
    };

    /**
     * @brief Coarse work-size bucket for JIT specialization.
     *
     * We keep this intentionally small so the runtime can cheaply cache and
     * swap between a couple of specialized decode kernels.
     */
    enum class WorkSizeClass : uint8_t
    {
        SMALL = 0,
        LARGE = 1,
        XL = 2,
    };

    /**
     * @brief Configuration for JIT attention kernel
     */
    struct JitAttentionConfig
    {
        int head_dim = 0;                         // Dimension per head (e.g., 64)
        int num_heads = 0;                        // Number of Q heads
        int num_kv_heads = 0;                     // Number of KV heads (GQA support)
        int batch_size = 0;                       // Batch size (1 for decode, >1 for prefill)
        WoFormat wo_format = WoFormat::Q8_1;      // Output projection weight format
        bool causal = true;                       // Whether to apply causal masking
        AttentionMode mode = AttentionMode::AUTO; // Kernel mode selection
        bool use_fa2_tiling = true;               // Enable FA2-style KV tile processing (4x batched)
        enum class Fa2TileWidth : uint8_t
        {
            KV4 = 4,
            KV8 = 8,
        };
        Fa2TileWidth fa2_tile_width = Fa2TileWidth::KV4; // KV tile width for FA2 (decode-focused)
        WorkSizeClass work_size = WorkSizeClass::SMALL;  // Coarse work-size class for specialization
        bool enable_register_tracking = false;           // Enable register conflict detection during JIT
        bool use_optimized_wo = false;                   // Use BLAS-style optimized Wo projection (FP32 only)

        /**
         * @brief Get the effective mode based on batch_size
         * @return DECODE if batch_size == 1, PREFILL otherwise
         */
        AttentionMode effectiveMode() const
        {
            if (mode == AttentionMode::AUTO)
            {
                return (batch_size <= 1) ? AttentionMode::DECODE : AttentionMode::PREFILL;
            }
            return mode;
        }

        bool operator==(const JitAttentionConfig &other) const
        {
            return head_dim == other.head_dim &&
                   num_heads == other.num_heads &&
                   num_kv_heads == other.num_kv_heads &&
                   batch_size == other.batch_size &&
                   wo_format == other.wo_format &&
                   causal == other.causal &&
                   effectiveMode() == other.effectiveMode() &&
                   use_fa2_tiling == other.use_fa2_tiling &&
                   work_size == other.work_size &&
                   enable_register_tracking == other.enable_register_tracking &&
                   use_optimized_wo == other.use_optimized_wo;
            fa2_tile_width == other.fa2_tile_width;
        }
    };

} // namespace llaminar::v2::kernels::jit

// Hash for JitAttentionConfig (for cache lookup)
namespace std
{
    template <>
    struct hash<llaminar::v2::kernels::jit::JitAttentionConfig>
    {
        size_t operator()(const llaminar::v2::kernels::jit::JitAttentionConfig &cfg) const
        {
            size_t h = 0;
            h ^= std::hash<int>()(cfg.head_dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(cfg.num_heads) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(cfg.num_kv_heads) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(cfg.batch_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(static_cast<int>(cfg.wo_format)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<bool>()(cfg.causal) + 0x9e3779b9 + (h << 6) + (h >> 2);
            // Include effective mode in hash to cache decode and prefill kernels separately
            h ^= std::hash<int>()(static_cast<int>(cfg.effectiveMode())) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<bool>()(cfg.use_fa2_tiling) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(static_cast<int>(cfg.work_size)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<bool>()(cfg.enable_register_tracking) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<bool>()(cfg.use_optimized_wo) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(static_cast<int>(cfg.fa2_tile_width)) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
}

namespace llaminar::v2::kernels::jit
{
    // Global debug buffer that JIT can write to
    /**
     * @brief Function signature for generated attention kernel
     *
     * @param Q Pointer to Q tensor (Q8_1 blocks)
     * @param K Pointer to K tensor (Q8_1 blocks)
     * @param V Pointer to V tensor (Q8_1 blocks)
     * @param Wo Pointer to Wo weights (format depends on config)
     * @param output Pointer to output buffer
     * @param seq_len_q Number of query positions
     * @param seq_len_kv Number of KV positions
     * @param scale Attention scale factor (1/sqrt(head_dim))
     * @param position_offset Position offset for causal masking (decode mode)
     */
    using JitAttentionKernelFn = void (*)(
        const void *Q,
        const void *K,
        const void *V,
        const void *Wo,
        float *output,
        int seq_len_q,
        int seq_len_kv,
        float scale,
        int position_offset);

    // External helper for packed VNNI Wo projection.
    // Wo points to a gemm_v4::QuantisedPackedWeights, A is FP32 context [m,k], C is FP32 output [m,n].
    extern "C" void llaminar2_wo_q8_1_vnni_packed_gemm(
        const void *wo_packed,
        const float *A,
        float *C,
        int m,
        int n,
        int k);

    /**
     * @brief JIT code generator for fused attention + Wo
     *
     * Generates optimized x86-64 AVX-512 code at runtime.
     */
    class JitFusedAttentionWoGenerator : public JitMicrokernelBase
    {
    public:
        explicit JitFusedAttentionWoGenerator(const JitAttentionConfig &config)
            : JitMicrokernelBase(512 * 1024) // 512KB code buffer for large models (72B has 64 heads)
              ,
              config_(config)
        {
            // Enable register tracking if requested
            if (config_.enable_register_tracking)
            {
                enable_register_tracking();
                debug_emit("[TRACKING] Register tracking ENABLED for this kernel");
            }

            // Calculate optimal Q_TILE_SIZE based on kernel mode and head_dim
            //
            // CRITICAL CONSTRAINT: Context accumulators must fit in registers!
            //
            // We follow the decode kernel's pattern:
            //   - Blocks 0-1 (64 floats): kept in registers (4 ZMM per query)
            //   - Blocks 2+: spilled to stack (accessed once per V block per KV)
            //
            // This avoids loading/storing the FULL context every KV iteration.
            // The spilled blocks are still touched per-KV but it's much less
            // traffic than the previous approach of spilling everything.
            //
            // Register allocation for context:
            //   - 4 ZMM per query for blocks 0-1 (zmm0-3 for Q0, zmm4-7 for Q1, etc.)
            //   - Need to leave room for V data, scores, softmax state
            //
            // Available: zmm0-15 (16 regs) for context
            // Reserve: zmm12-15 for V data and intermediates (4 regs)
            // Effective: zmm0-11 (12 regs) for context = 3 queries × 4 ZMM
            //
            // However, we also limit to 2 queries for simplicity with softmax
            // vectorization using XMM (4 floats, but we want simpler code).
            //
            int num_blocks = config_.head_dim / 32;

            if (config_.effectiveMode() == AttentionMode::PREFILL)
            {
                // Tile size for prefill: maximize K/V cache reuse
                // Process 4 queries per K/V load, allowing K/V data to be
                // amortized across multiple queries before eviction from L1.
                // All context blocks are stored in memory (stack buffer).

                q_tile_size_ = 4; // Process 4 queries per K/V load
            }
            else
            {
                // Decode: Constrained by Q register allocation
                int available_q_regs = 4;
                if (num_blocks > 0)
                {
                    q_tile_size_ = available_q_regs / num_blocks;
                }
                else
                {
                    q_tile_size_ = 4;
                }
            }

            // Clamp to reasonable limits
            if (q_tile_size_ > 8)
                q_tile_size_ = 8;
            if (q_tile_size_ < 1)
                q_tile_size_ = 1;

            generate();
        }

        /**
         * @brief Get the generated kernel function pointer
         */
        JitAttentionKernelFn getKernel()
        {
            return getCode<JitAttentionKernelFn>();
        }

    private:
        JitAttentionConfig config_;
        int q_tile_size_ = 8;

        // Microkernel emitters
        JitQ8DotProductEmitter dot_emitter_;
        JitOnlineSoftmaxEmitter softmax_emitter_;
        JitVWeightedAccumEmitter v_accum_emitter_;
        JitWoProjectionEmitter wo_emitter_;
        JitFastExpEmitter exp_emitter_; // Vectorized exp for prefill softmax

        /**
         * @brief Generate the complete kernel
         *
         * Dispatches to decode or prefill code generation based on config.
         */
        void generate()
        {
            if (config_.effectiveMode() == AttentionMode::PREFILL)
            {
                generate_prefill_kernel();
            }
            else
            {
                generate_decode_kernel();
            }
        }

        /**
         * @brief Generate decode-optimized kernel (seq_len_q == 1)
         *
         * This kernel is specialized for autoregressive token generation where
         * seq_len_q == 1 (single query token at a time). The loop order is
         * Q → H → KV, which is optimal when there's only one query position.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * ALGORITHM OVERVIEW
         * ═══════════════════════════════════════════════════════════════════════
         *
         * For each query position q (0 to seq_len_q-1):
         *   For each attention head h (0 to num_heads-1):
         *     1. Initialize context accumulators to zero
         *     2. Initialize softmax state: max=-infinity, sum=0
         *     3. Load Q[q,h] blocks into ZMM registers (persist across KV loop)
         *     4. For each KV position kv (0 to max_kv_pos-1):  // causal-aware
         *        a. Compute score = Q[q,h] · K[kv,kv_h] * scale
         *        b. Online softmax update (track running max and sum)
         *        c. Rescale existing context by correction factor
         *        d. Accumulate weighted V[kv,kv_h] into context
         *     5. Normalize context by 1/sum
         *     6. Store context to buffer
         *   Apply Wo projection: output[q] = context × Wo^T
         *
         * ═══════════════════════════════════════════════════════════════════════
         * CALLING CONVENTION (System V AMD64)
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Input registers (function entry):
         *   rdi = Q        Pointer to Q tensor (Q8_1 blocks) [seq_len_q, num_heads, head_dim]
         *   rsi = K        Pointer to K tensor (Q8_1 blocks) [seq_len_kv, num_kv_heads, head_dim]
         *   rdx = V        Pointer to V tensor (Q8_1 blocks) [seq_len_kv, num_kv_heads, head_dim]
         *   rcx = Wo       Pointer to Wo weights [d_model, d_model] (format per config)
         *   r8  = output   Pointer to output buffer (FP32) [seq_len_q, d_model]
         *   r9  = seq_len_q Number of query positions
         *   xmm0 = scale   Attention scale factor (1/sqrt(head_dim))
         *
         * Stack parameters (pushed before our frame):
         *   [rsp + frame + 8]  = seq_len_kv      Number of KV cache positions
         *   [rsp + frame + 16] = position_offset  Causal mask offset
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER ALLOCATION
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Callee-Saved GPRs (preserved across function, store input pointers):
         *   ┌─────────┬───────────────────────────────────────────────────────┐
         *   │ r12     │ Q pointer (base address of query tensor)              │
         *   │ r13     │ K pointer (base address of key tensor)                │
         *   │ rbx     │ V pointer (base address of value tensor)              │
         *   │ rbp     │ Wo pointer (base address of output projection)        │
         *   │ r14     │ output pointer (base address of output buffer)        │
         *   │ r15     │ seq_len_q (number of query positions)                 │
         *   └─────────┴───────────────────────────────────────────────────────┘
         *
         * Caller-Saved GPRs (scratch, clobbered freely):
         *   ┌─────────┬───────────────────────────────────────────────────────┐
         *   │ rax     │ General scratch, loop index (q_idx)                   │
         *   │ rcx     │ KV loop counter, scratch                              │
         *   │ rdx     │ Arithmetic scratch                                    │
         *   │ rdi     │ Pointer calculations, emitter scratch                 │
         *   │ rsi     │ Pointer calculations, emitter scratch                 │
         *   │ r8-r11  │ General scratch for intermediate values               │
         *   └─────────┴───────────────────────────────────────────────────────┘
         *
         * ZMM Register Zones (see JitMicrokernelBase.h for details):
         *   ┌───────────┬────────────────────────────────────────────────────┐
         *   │ zmm0-7    │ ACCUM: Context accumulators (32 floats each)       │
         *   │ zmm8-9    │ INPUT: V accumulation scratch                      │
         *   │ zmm10-13  │ INPUT: Q head data (persists across KV loop)       │
         *   │ zmm14-15  │ INPUT: emit_fast_exp scratch (available)           │
         *   │ zmm16     │ STATE: Running softmax maximum                     │
         *   │ zmm17     │ STATE: Running softmax sum                         │
         *   │ zmm18     │ STATE: Current attention weight (broadcasted)      │
         *   │ zmm19     │ STATE: Correction factor (broadcasted)             │
         *   │ zmm20-25  │ SCRATCH: Temporaries (freely clobbered)            │
         *   │ zmm26     │ CONST: 0x80808080 (unsigned→signed conversion)     │
         *   │ zmm27     │ CONST: scale (1/sqrt(head_dim), broadcasted)       │
         *   │ zmm28     │ CONST: -infinity (softmax init)                    │
         *   │ zmm29     │ CONST: 1.0f                                        │
         *   │ zmm30     │ CONST: log2(e) for fast exp                        │
         *   │ zmm31     │ CONST: -87.0f (exp underflow clamp)                │
         *   └───────────┴────────────────────────────────────────────────────┘
         *
         * ═══════════════════════════════════════════════════════════════════════
         * STACK LAYOUT
         * ═══════════════════════════════════════════════════════════════════════
         *
         * After sub rsp, stack_size:
         *   ┌─────────────────────────────────────────────────────────────────┐
         *   │ [rsp + 0]                    Q blocks for current head         │
         *   │                              (num_blocks * 64 bytes, padded)   │
         *   ├─────────────────────────────────────────────────────────────────┤
         *   │ [rsp + q_stack_size]         Context accumulator spill area    │
         *   │                              ((num_blocks-2) * 128 bytes)      │
         *   ├─────────────────────────────────────────────────────────────────┤
         *   │ [rsp + context_buffer_off]   Context buffer for Wo projection  │
         *   │                              (d_model * 4 bytes, FP32)         │
         *   ├─────────────────────────────────────────────────────────────────┤
         *   │ [rsp + q_idx_spill]          Saved q_idx (8 bytes)             │
         *   │ [rsp + seq_len_kv_spill]     Saved seq_len_kv (8 bytes)        │
         *   │ [rsp + position_offset_spill] Saved position_offset (8 bytes)  │
         *   ├─────────────────────────────────────────────────────────────────┤
         *   │ ... alignment padding to 64-byte boundary ...                  │
         *   └─────────────────────────────────────────────────────────────────┘
         *
         * ═══════════════════════════════════════════════════════════════════════
         * CAUSAL MASKING
         * ═══════════════════════════════════════════════════════════════════════
         *
         * When config_.causal is true, each query position q can only attend
         * to KV positions [0, q + position_offset]. The position_offset allows
         * for incremental decoding where position_offset = number of previously
         * processed tokens.
         *
         * max_kv_pos = min(q_idx + position_offset + 1, seq_len_kv)
         *
         * ═══════════════════════════════════════════════════════════════════════
         * GQA (GROUPED QUERY ATTENTION) SUPPORT
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Supports grouped query attention where num_heads > num_kv_heads.
         * Multiple Q heads share the same K/V heads:
         *   kv_head_idx = head_idx / (num_heads / num_kv_heads)
         *
         */
        void generate_decode_kernel()
        {
            using namespace Xbyak;

            // ═══════════════════════════════════════════════════════════════════
            // FUNCTION PROLOGUE
            // ═══════════════════════════════════════════════════════════════════
            // Save callee-saved registers and set up stack frame.
            // After this: rsp is 48 bytes lower, callee-saved regs preserved.

            push_callee_saved();

            // ═══════════════════════════════════════════════════════════════════
            // SAVE INPUT PARAMETERS TO CALLEE-SAVED REGISTERS
            // ═══════════════════════════════════════════════════════════════════
            // Move function arguments from caller-saved to callee-saved registers
            // so they persist across all emitter calls and loop iterations.

            Reg64 reg_Q = r12;         // Preserved: Q tensor pointer
            Reg64 reg_K = r13;         // Preserved: K tensor pointer
            Reg64 reg_V = rbx;         // Preserved: V tensor pointer
            Reg64 reg_Wo = rbp;        // Preserved: Wo weights pointer
            Reg64 reg_output = r14;    // Preserved: Output buffer pointer
            Reg64 reg_seq_len_q = r15; // Preserved: Number of query positions

            mov(reg_Q, rdi);        // rdi = 1st arg (Q)
            mov(reg_K, rsi);        // rsi = 2nd arg (K)
            mov(reg_V, rdx);        // rdx = 3rd arg (V)
            mov(reg_Wo, rcx);       // rcx = 4th arg (Wo)
            mov(reg_output, r8);    // r8  = 5th arg (output)
            mov(reg_seq_len_q, r9); // r9  = 6th arg (seq_len_q)

            // ═══════════════════════════════════════════════════════════════════
            // LOAD STACK PARAMETERS
            // ═══════════════════════════════════════════════════════════════════
            // 7th and 8th integer arguments are passed on the stack in System V.
            // Stack layout after push_callee_saved (48 bytes) + return address (8):
            //   [rsp + 48 + 8 + 0]  = seq_len_kv (7th arg)
            //   [rsp + 48 + 8 + 8]  = position_offset (8th arg)
            //
            // Note: r10 is caller-saved and may be clobbered during emit_wo_projection,
            // so we spill seq_len_kv to our stack frame later.

            Reg64 reg_seq_len_kv = r10; // Temporary: will be spilled to stack
            mov(reg_seq_len_kv, ptr[rsp + stack_frame_size() + 8]);

            // ═══════════════════════════════════════════════════════════════════
            // INITIALIZE CONSTANT REGISTERS (zmm26-31)
            // ═══════════════════════════════════════════════════════════════════
            // These constants persist for the entire kernel execution.
            // See JitMicrokernelBase.h ConstRegs for register assignments.

            // zmm27: Broadcast attention scale from xmm0 (1st float arg)
            vbroadcastss(const_scale().zmm(), xmm0);

            // zmm30: log2(e) = 1.44269504089f for fast exp computation
            load_constant_f32(const_log2e().zmm(), 1.44269504089f);

            // zmm31: -87.0f exp underflow clamp (exp(-87) ≈ 1e-38)
            load_constant_f32(const_exp_min().zmm(), -87.0f);

            // zmm29: 1.0f for various computations
            load_constant_f32(const_one().zmm(), 1.0f);

            // zmm26: 0x80808080 pattern for Q8_1 unsigned-to-signed conversion
            // Q8_1 stores values as unsigned [0,255], we XOR to get signed [-128,127]
            emit_broadcast_i32_const(const_128().zmm(), 0x80808080, rax);

            // ═══════════════════════════════════════════════════════════════════
            // CALCULATE STACK FRAME SIZE AND LAYOUT
            // ═══════════════════════════════════════════════════════════════════
            // Stack is used for:
            //   1. Q blocks: Padded Q8_1 blocks for one head (64 bytes each)
            //   2. Spill area: Context accumulators that don't fit in zmm0-3
            //   3. Context buffer: FP32 intermediate for Wo projection
            //   4. Spill slots: Values that must survive emitter calls

            int num_blocks = config_.head_dim / 32;                          // Q8_1 blocks per head
            int d_model = config_.num_heads * config_.head_dim;              // Total hidden dimension
            int q_stack_size = num_blocks * 64;                              // Padded Q blocks (36→64 bytes each)
            int spill_bytes = (num_blocks > 2) ? (num_blocks - 2) * 128 : 0; // 2 ZMMs per extra block
            int context_buffer_size = d_model * 4;                           // FP32 context for all heads
            int stack_size = q_stack_size + spill_bytes + context_buffer_size + 256;
            stack_size = (stack_size + 63) & ~63; // Align to 64-byte cache line

            // ═══════════════════════════════════════════════════════════════════
            // STACK OFFSET ASSIGNMENTS
            // ═══════════════════════════════════════════════════════════════════
            //
            // Layout from rsp after allocation:
            //   ┌─────────────────────────────────────────────────────────────┐
            //   │ [rsp + q_stack_offset]        Q blocks (num_blocks * 64)   │
            //   │ [rsp + spill_base_offset]     Accumulator spill area       │
            //   │ [rsp + context_buffer_offset] Context buffer (d_model * 4) │
            //   │ [rsp + q_idx_spill_offset]    Saved q_idx (8 bytes)        │
            //   │ [rsp + seq_len_kv_spill]      Saved seq_len_kv (8 bytes)   │
            //   │ [rsp + position_offset_spill] Saved position_offset (8)    │
            //   └─────────────────────────────────────────────────────────────┘

            int q_stack_offset = 0;
            int spill_base_offset = q_stack_size;
            int context_buffer_offset = q_stack_size + spill_bytes;
            int q_idx_spill_offset = context_buffer_offset + d_model * 4;
            int seq_len_kv_spill_offset = q_idx_spill_offset + 8;
            int position_offset_spill_offset = seq_len_kv_spill_offset + 8;
            // Scratch: spill scores for FA2 wider tiles (8x needs 8 scalar scores + 1 scalar max)
            int fa2_scores_spill_offset = position_offset_spill_offset + 8;
            int fa2_max1_spill_offset = fa2_scores_spill_offset + 32; // 8 floats = 32 bytes

            // ═══════════════════════════════════════════════════════════════════
            // ALLOCATE STACK FRAME
            // ═══════════════════════════════════════════════════════════════════

            sub(rsp, stack_size);

            // ═══════════════════════════════════════════════════════════════════
            // SPILL STACK PARAMETERS
            // ═══════════════════════════════════════════════════════════════════
            // Save values that are passed in caller-saved registers or on the
            // original stack. These must be accessible throughout the kernel.

            // Save seq_len_kv (it gets clobbered by emit_wo_projection)
            mov(ptr[rsp + seq_len_kv_spill_offset], reg_seq_len_kv);

            // Load and save position_offset from original stack
            // After our stack allocation, the original stack is at:
            //   [rsp + stack_size + stack_frame_size() + 16]
            Reg64 reg_tmp = rdi; // Temporarily reuse rdi (will be overwritten later)
            mov(reg_tmp, ptr[rsp + stack_size + stack_frame_size() + 16]);
            mov(ptr[rsp + position_offset_spill_offset], reg_tmp);

            // ═══════════════════════════════════════════════════════════════════
            // MAIN QUERY LOOP
            // ═══════════════════════════════════════════════════════════════════
            // For decode mode with seq_len_q == 1, this loop executes once.
            // For batch decode, it processes each query position sequentially.

            Reg64 reg_q_idx = rax;      // Loop counter: current query index
            xor_(reg_q_idx, reg_q_idx); // q_idx = 0

            L("main_loop_q");
            cmp(reg_q_idx, reg_seq_len_q);
            jge("main_loop_q_end", T_NEAR);

            // ═══════════════════════════════════════════════════════════════════
            // PROCESS ONE QUERY POSITION
            // ═══════════════════════════════════════════════════════════════════
            // Restore seq_len_kv from spill (may have been clobbered in previous
            // iteration by Wo projection code)

            mov(reg_seq_len_kv, ptr[rsp + seq_len_kv_spill_offset]);

            // emit_single_query_attention processes all heads for one query,
            // then applies Wo projection to produce the final output.
            emit_single_query_attention(
                reg_Q, reg_K, reg_V, reg_Wo, reg_output,
                reg_q_idx, reg_seq_len_kv,
                num_blocks, spill_base_offset, q_stack_offset, context_buffer_offset,
                q_idx_spill_offset, position_offset_spill_offset,
                fa2_scores_spill_offset, fa2_max1_spill_offset);

            inc(reg_q_idx); // q_idx++
            jmp("main_loop_q", T_NEAR);

            L("main_loop_q_end");

            // ═══════════════════════════════════════════════════════════════════
            // FUNCTION EPILOGUE
            // ═══════════════════════════════════════════════════════════════════
            // Deallocate stack frame and restore callee-saved registers.

            add(rsp, stack_size);
            pop_callee_saved();
            ret();

            ready(); // Finalize code generation
        }

        /**
         * @brief Generate prefill-optimized kernel (seq_len_q > 1)
         *
         * This kernel is optimized for multi-token processing (prefill phase)
         * where we process multiple query positions simultaneously. The key
         * optimization is K/V cache reuse: each K/V block is loaded once per
         * head and reused across all queries in a tile.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * ALGORITHM OVERVIEW
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Loop order: Q_tile → H → KV → Q_inner
         *
         * For each tile of Q_TILE_SIZE queries:
         *   Initialize softmax state for all queries × all heads
         *   Initialize context buffer to zero for all queries
         *   For each head h:
         *     Copy Q blocks for tile[h] to stack
         *     For each KV position kv:
         *       Load K[kv,kv_h] once  ← KEY OPTIMIZATION: reused across tile
         *       Load V[kv,kv_h] once  ← KEY OPTIMIZATION: reused across tile
         *       For each query q in tile (q_local_start to tile_size):
         *         Skip if causal mask rejects this (q,kv) pair
         *         Compute score = Q[tile+q, h] · K[kv, kv_h] * scale
         *         Vectorized online softmax update
         *         Accumulate weighted V into context[q,h]
         *   Normalize context by 1/sum for all queries × all heads
         *   Apply Wo projection for all queries in tile
         *
         * ═══════════════════════════════════════════════════════════════════════
         * CALLING CONVENTION (System V AMD64)
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Same as decode kernel - see generate_decode_kernel() for details.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER ALLOCATION
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Callee-Saved GPRs (preserved, store input pointers):
         *   ┌─────────┬───────────────────────────────────────────────────────┐
         *   │ r12     │ Q pointer                                             │
         *   │ r13     │ K pointer                                             │
         *   │ rbx     │ V pointer                                             │
         *   │ rbp     │ Wo pointer                                            │
         *   │ r14     │ output pointer                                        │
         *   │ r15     │ seq_len_q                                             │
         *   └─────────┴───────────────────────────────────────────────────────┘
         *
         * Caller-Saved GPRs (scratch, reloaded from stack as needed):
         *   ┌─────────┬───────────────────────────────────────────────────────┐
         *   │ rax     │ Loop variables, arithmetic, spill load/store          │
         *   │ rcx     │ Loop counter, div/mul scratch                         │
         *   │ rdx     │ Division remainder, offset calculations               │
         *   │ rdi     │ K/V pointer calculations                              │
         *   │ rsi     │ K/V pointer calculations                              │
         *   │ r8-r11  │ Tile loop variables, temporary pointers               │
         *   └─────────┴───────────────────────────────────────────────────────┘
         *
         * ZMM Register Allocation for Prefill:
         *   ┌───────────┬────────────────────────────────────────────────────┐
         *   │ zmm0-7    │ Score accumulators for queries 0-7 in tile         │
         *   │           │ After softmax: corr[0-7] (broadcasted)             │
         *   │ zmm8-15   │ After softmax: weight[0-7] (broadcasted)           │
         *   │           │ (Context uses memory buffer, not registers)        │
         *   │ zmm16     │ Packed old_max for vectorized softmax              │
         *   │ zmm17     │ Packed old_sum / new_sum                           │
         *   │ zmm18     │ Packed scores for vectorized softmax               │
         *   │ zmm19     │ Packed new_max                                     │
         *   │ zmm20     │ corr_input (old_max - new_max)                     │
         *   │ zmm21     │ weight_input (score - new_max)                     │
         *   │ zmm22-25  │ Scratch for exp_emitter_ and other operations      │
         *   │ zmm26     │ CONST: 0x80808080                                  │
         *   │ zmm27     │ CONST: scale                                       │
         *   │ zmm28     │ CONST: 16.0f (prefill mode)                        │
         *   │ zmm29     │ CONST: 1.0f                                        │
         *   │ zmm30     │ CONST: log2(e)                                     │
         *   │ zmm31     │ CONST: -87.0f                                      │
         *   └───────────┴────────────────────────────────────────────────────┘
         *
         * ═══════════════════════════════════════════════════════════════════════
         * STACK LAYOUT (Q_TILE_SIZE queries per tile)
         * ═══════════════════════════════════════════════════════════════════════
         *
         *   ┌─────────────────────────────────────────────────────────────────┐
         *   │ [rsp + q_blocks_offset]      Q blocks for tile                 │
         *   │                              (tile_size * num_blocks * 64)     │
         *   ├─────────────────────────────────────────────────────────────────┤
         *   │ [rsp + softmax_offset]       Softmax state per query per head  │
         *   │                              (tile_size * num_heads * 8 bytes) │
         *   │                              Format: [max, sum] pairs (FP32)   │
         *   ├─────────────────────────────────────────────────────────────────┤
         *   │ [rsp + context_offset]       Context buffer                    │
         *   │                              (tile_size * d_model * 4 bytes)   │
         *   ├─────────────────────────────────────────────────────────────────┤
         *   │ [rsp + tile_start_spill]     Current tile start index (8)      │
         *   │ [rsp + tile_size_spill]      Current tile size (8)             │
         *   │ [rsp + seq_len_kv_spill]     seq_len_kv (8)                     │
         *   │ [rsp + position_offset_spill] position_offset (8)              │
         *   │ [rsp + kv_idx_spill]         KV loop index (8)                 │
         *   │ [rsp + q_local_spill]        Query local index (8)             │
         *   │ [rsp + k_ptr_spill]          K pointer for current KV (8)      │
         *   │ [rsp + v_ptr_spill]          V pointer for current KV (8)      │
         *   │ [rsp + head_idx_spill]       Head loop index (8)               │
         *   │ [rsp + head_offset_spill]    Head offset in Q blocks (8)       │
         *   │ [rsp + softmax_head_offset]  Softmax offset for head (8)       │
         *   │ [rsp + context_head_offset]  Context offset for head (8)       │
         *   │ [rsp + q_loop_spill]         Query loop counter (8)            │
         *   │ [rsp + kv_loop_end_spill]    KV loop end bound (8)             │
         *   │ [rsp + kv_head_offset_spill] Pre-computed kv_head offset (8)   │
         *   └─────────────────────────────────────────────────────────────────┘
         *
         * ═══════════════════════════════════════════════════════════════════════
         * TILE PROCESSING STRATEGY
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Queries are processed in tiles of Q_TILE_SIZE (typically 8, but
         * dynamically sized based on head_dim to fit Q data in registers).
         * This bounds stack usage while enabling K/V reuse within the tile.
         *
         * For each tile:
         *   - All queries in the tile share the same K/V loads
         *   - Vectorized softmax processes all queries in parallel
         *   - Context updates are batched per K/V position
         *
         * ═══════════════════════════════════════════════════════════════════════
         * CAUSAL MASKING IN PREFILL
         * ═══════════════════════════════════════════════════════════════════════
         *
         * For causal attention, query q can only attend to KV positions ≤ q.
         * Within a tile, different queries have different valid KV ranges.
         *
         * For a given KV position kv:
         *   q_start = max(0, kv - position_offset)  // First valid query
         *   q_local_start = max(0, q_start - tile_start)  // Local to tile
         *
         * Queries q < q_local_start skip attention for this KV position.
         *
         */
        void generate_prefill_kernel()
        {
            using namespace Xbyak;

            // ═══════════════════════════════════════════════════════════════════
            // FUNCTION PROLOGUE
            // ═══════════════════════════════════════════════════════════════════

            push_callee_saved();

            // ═══════════════════════════════════════════════════════════════════
            // SAVE INPUT PARAMETERS TO CALLEE-SAVED REGISTERS
            // ═══════════════════════════════════════════════════════════════════

            Reg64 reg_Q = r12;         // Preserved: Q tensor pointer
            Reg64 reg_K = r13;         // Preserved: K tensor pointer
            Reg64 reg_V = rbx;         // Preserved: V tensor pointer
            Reg64 reg_Wo = rbp;        // Preserved: Wo weights pointer
            Reg64 reg_output = r14;    // Preserved: Output buffer pointer
            Reg64 reg_seq_len_q = r15; // Preserved: Number of query positions

            mov(reg_Q, rdi);
            mov(reg_K, rsi);
            mov(reg_V, rdx);
            mov(reg_Wo, rcx);
            mov(reg_output, r8);
            mov(reg_seq_len_q, r9);

            // ═══════════════════════════════════════════════════════════════════
            // INITIALIZE CONSTANT REGISTERS
            // ═══════════════════════════════════════════════════════════════════

            // zmm27: Attention scale factor (broadcasted from xmm0)
            vbroadcastss(const_scale().zmm(), xmm0);

            // zmm30: log2(e) for fast exp computation
            load_constant_f32(const_log2e().zmm(), 1.44269504089f);

            // zmm29: 1.0f for various computations
            load_constant_f32(const_one().zmm(), 1.0f);

            // zmm31: -87.0f exp underflow clamp
            load_constant_f32(const_exp_min().zmm(), -87.0f);

            // zmm26: 0x80808080 for unsigned-to-signed conversion
            emit_broadcast_i32_const(const_128().zmm(), 0x80808080, rax);

            // ═══════════════════════════════════════════════════════════════════
            // CALCULATE DIMENSIONS AND STACK LAYOUT
            // ═══════════════════════════════════════════════════════════════════

            int num_blocks = config_.head_dim / 32;                      // Q8_1 blocks per head
            int d_model = config_.num_heads * config_.head_dim;          // Total hidden dimension
            int head_dim = config_.head_dim;                             // Dimension per head
            int heads_per_kv = config_.num_heads / config_.num_kv_heads; // GQA ratio

            // ═══════════════════════════════════════════════════════════════════
            // STACK OFFSET CALCULATIONS
            // ═══════════════════════════════════════════════════════════════════

            int q_blocks_size = q_tile_size_ * num_blocks * 64;            // Q blocks for tile
            int softmax_state_size = q_tile_size_ * config_.num_heads * 8; // (max,sum) per q per head
            int context_size = q_tile_size_ * d_model * 4;                 // FP32 context per query

            int q_blocks_offset = 0;
            int softmax_offset = q_blocks_size;
            int context_offset = softmax_offset + softmax_state_size;
            int spill_vars_offset = context_offset + context_size;

            // ═══════════════════════════════════════════════════════════════════
            // SPILL SLOT ASSIGNMENTS
            // ═══════════════════════════════════════════════════════════════════
            // All loop variables and temporary values that must survive emitter
            // calls are stored on the stack. GPRs are reloaded as needed.

            int tile_start_spill = spill_vars_offset;         // Current tile start
            int tile_size_spill = tile_start_spill + 8;       // Current tile size
            int seq_len_kv_spill = tile_size_spill + 8;       // KV sequence length
            int position_offset_spill = seq_len_kv_spill + 8; // Causal offset
            int kv_idx_spill = position_offset_spill + 8;     // KV loop index
            int q_local_spill = kv_idx_spill + 8;             // Query local index
            int k_ptr_spill = q_local_spill + 8;              // K pointer for current KV
            int v_ptr_spill = k_ptr_spill + 8;                // V pointer for current KV

            // Additional spill slots for head loop
            int head_idx_spill = v_ptr_spill + 8;                          // Head loop index
            int head_offset_spill = head_idx_spill + 8;                    // Head offset in Q blocks
            int softmax_head_offset_spill = head_offset_spill + 8;         // Softmax offset for head
            int context_head_offset_spill = softmax_head_offset_spill + 8; // Context offset for head
            int q_loop_spill = context_head_offset_spill + 8;              // Query loop counter
            int kv_loop_end_spill = q_loop_spill + 8;                      // KV loop end bound
            int kv_head_offset_spill = kv_loop_end_spill + 8;              // Pre-computed kv_head * kv_head_stride

            // Total stack size (64-byte aligned for AVX-512)
            int stack_size = kv_head_offset_spill + 8 + 64;
            stack_size = (stack_size + 63) & ~63;

            // ═══════════════════════════════════════════════════════════════════
            // ALLOCATE STACK FRAME AND SPILL PARAMETERS
            // ═══════════════════════════════════════════════════════════════════

            sub(rsp, stack_size);

            // Spill seq_len_kv (7th arg from original stack)
            mov(rax, ptr[rsp + stack_size + stack_frame_size() + 8]);
            mov(ptr[rsp + seq_len_kv_spill], rax);

            // Spill position_offset (8th arg from original stack)
            mov(rax, ptr[rsp + stack_size + stack_frame_size() + 16]);
            mov(ptr[rsp + position_offset_spill], rax);

            // Re-initialize zmm26 (may have been clobbered)
            emit_broadcast_i32_const(const_128().zmm(), 0x80808080, rax);

            // ═══════════════════════════════════════════════════════════════════
            // TILE LOOP: Process queries in tiles of Q_TILE_SIZE
            // ═══════════════════════════════════════════════════════════════════

            mov(qword[rsp + tile_start_spill], 0); // tile_start = 0

            L("prefill_tile_loop");

            // Check if all queries processed
            mov(rax, ptr[rsp + tile_start_spill]);
            cmp(rax, reg_seq_len_q);
            jge("prefill_tile_end", T_NEAR);

            // ═══════════════════════════════════════════════════════════════════
            // CALCULATE TILE SIZE
            // ═══════════════════════════════════════════════════════════════════
            // tile_size = min(q_tile_size_, seq_len_q - tile_start)

            mov(rcx, reg_seq_len_q);
            sub(rcx, rax); // rcx = remaining queries
            cmp(rcx, q_tile_size_);
            jle("prefill_tile_size_ok", T_NEAR);
            mov(rcx, q_tile_size_); // Clamp to tile size
            L("prefill_tile_size_ok");
            mov(ptr[rsp + tile_size_spill], rcx);

            // ═══════════════════════════════════════════════════════════════════
            // INITIALIZE SOFTMAX STATE FOR ALL QUERIES × ALL HEADS
            // ═══════════════════════════════════════════════════════════════════
            // max = -infinity, sum = 0 for each (query, head) pair

            emit_prefill_init_softmax(q_tile_size_ * config_.num_heads, softmax_offset);

            // ═══════════════════════════════════════════════════════════════════
            // INITIALIZE CONTEXT BUFFER TO ZEROS
            // ═══════════════════════════════════════════════════════════════════

            emit_prefill_init_context(q_tile_size_, d_model, context_offset);

            // ═══════════════════════════════════════════════════════════════════
            // HEAD LOOP: Process each attention head
            // ═══════════════════════════════════════════════════════════════════

            mov(qword[rsp + head_idx_spill], 0);    // head_idx = 0
            mov(qword[rsp + head_offset_spill], 0); // head_offset = 0
            mov(rax, softmax_offset);
            mov(ptr[rsp + softmax_head_offset_spill], rax);
            mov(rax, context_offset);
            mov(ptr[rsp + context_head_offset_spill], rax);

            L("prefill_head_loop");
            mov(rax, ptr[rsp + head_idx_spill]);
            cmp(rax, config_.num_heads);
            jge("prefill_head_end", T_NEAR);

            // ═══════════════════════════════════════════════════════════════════
            // COPY Q BLOCKS FOR ALL QUERIES IN TILE FOR THIS HEAD
            // ═══════════════════════════════════════════════════════════════════

            mov(r11, ptr[rsp + head_offset_spill]);
            emit_prefill_copy_q_tile(reg_Q, r11, num_blocks, q_blocks_offset,
                                     tile_start_spill, tile_size_spill);

            // ═══════════════════════════════════════════════════════════════════
            // CALCULATE KV HEAD INDEX (GQA support) - HOISTED OUT OF KV LOOP
            // ═══════════════════════════════════════════════════════════════════
            // kv_head = head_idx / heads_per_kv
            // This is constant for the entire KV loop, so compute once here.
            // Division is expensive (~30-80 cycles), hoisting saves significant time.

            xor_(rdx, rdx);
            mov(rax, ptr[rsp + head_idx_spill]);
            mov(rcx, heads_per_kv);
            div(rcx); // rax = kv_head, rdx = remainder

            // Pre-compute kv_head_offset = kv_head * kv_head_stride
            // This is also constant for the KV loop
            int kv_stride = config_.num_kv_heads * num_blocks * 36; // Per KV position
            int kv_head_stride = num_blocks * 36;                   // Per KV head
            imul(rax, rax, kv_head_stride);
            mov(ptr[rsp + kv_head_offset_spill], rax); // Store kv_head_offset

            // ═══════════════════════════════════════════════════════════════════
            // KV LOOP: Iterate over K/V cache positions
            // ═══════════════════════════════════════════════════════════════════

            // Context buffer is zeroed at tile start (emit_init_context_buffer)

            // Calculate KV loop end
            mov(rax, ptr[rsp + seq_len_kv_spill]);
            if (config_.causal)
            {
                // kv_end = min(seq_len_kv, tile_start + tile_size + position_offset)
                mov(r10, ptr[rsp + tile_start_spill]);
                add(r10, ptr[rsp + tile_size_spill]);
                add(r10, ptr[rsp + position_offset_spill]);

                cmp(rax, r10);
                cmovg(rax, r10); // rax = min(rax, r10)
            }
            mov(ptr[rsp + kv_loop_end_spill], rax);

            mov(qword[rsp + kv_idx_spill], 0); // kv_idx = 0

            L(".kv_loop");
            mov(r8, ptr[rsp + kv_idx_spill]);
            mov(r9, ptr[rsp + kv_loop_end_spill]);
            cmp(r8, r9);
            jge(".kv_end", T_NEAR);

            // ═══════════════════════════════════════════════════════════════════
            // CALCULATE K/V POINTERS FOR CURRENT KV POSITION
            // ═══════════════════════════════════════════════════════════════════
            // K_ptr = reg_K + kv_idx * kv_stride + kv_head_offset
            // V_ptr = reg_V + kv_idx * kv_stride + kv_head_offset
            // (kv_head_offset was pre-computed above)

            mov(rdi, ptr[rsp + kv_idx_spill]);
            imul(rdi, rdi, kv_stride);
            add(rdi, ptr[rsp + kv_head_offset_spill]); // Add pre-computed kv_head_offset

            mov(rsi, rdi); // Copy offset for V

            add(rdi, reg_K);
            mov(ptr[rsp + k_ptr_spill], rdi);

            add(rsi, reg_V);
            mov(ptr[rsp + v_ptr_spill], rsi);

            // ═══════════════════════════════════════════════════════════════════
            // CALCULATE CAUSAL MASK START POSITION
            // ═══════════════════════════════════════════════════════════════════
            // For causal: q_start = max(0, kv_idx - position_offset)
            // q_local_start = max(0, q_start - tile_start)

            mov(r10, ptr[rsp + kv_idx_spill]);
            if (config_.causal)
            {
                sub(r10, ptr[rsp + position_offset_spill]);
                xor_(r11, r11);
                cmp(r10, 0);
                cmovl(r10, r11); // q_start = max(0, kv_idx - position_offset)
            }
            else
            {
                xor_(r10, r10); // Non-causal: all queries valid
            }

            mov(ptr[rsp + q_local_spill], r10); // Temporarily save q_start

            // Calculate q_local_start = max(0, q_start - tile_start)
            mov(r10, ptr[rsp + q_local_spill]);
            mov(rax, ptr[rsp + tile_start_spill]);
            sub(r10, rax);
            xor_(r11, r11);
            cmp(r10, 0);
            cmovl(r10, r11);
            mov(ptr[rsp + q_local_spill], r10); // Final q_local_start

            // ═══════════════════════════════════════════════════════════════════
            // PROCESS ATTENTION FOR ALL VALID QUERIES IN TILE
            // ═══════════════════════════════════════════════════════════════════

            // Reload K/V pointers from spill
            mov(rdi, ptr[rsp + k_ptr_spill]);
            mov(rsi, ptr[rsp + v_ptr_spill]);

            // ═══════════════════════════════════════════════════════════════════
            // SOFTWARE PREFETCH: Bring next KV position into L1/L2 cache
            // ═══════════════════════════════════════════════════════════════════
            // Prefetch next K/V blocks while processing current ones.
            // This hides memory latency for streaming KV access.
            // prefetcht0 = L1, prefetcht1 = L2, prefetcht2 = L3
            prefetcht0(ptr[rdi + kv_stride]); // Next K position, block 0
            prefetcht0(ptr[rsi + kv_stride]); // Next V position, block 0
            if (num_blocks > 1)
            {
                prefetcht0(ptr[rdi + kv_stride + 64]); // Next K, block 1
                prefetcht0(ptr[rsi + kv_stride + 64]); // Next V, block 1
            }

            // Load dynamic offsets for softmax and context
            mov(rdx, ptr[rsp + softmax_head_offset_spill]);
            mov(rcx, ptr[rsp + context_head_offset_spill]);

            emit_prefill_tile_attention(
                rdi, rsi, r10,              // K_ptr, V_ptr, q_local_start
                ptr[rsp + tile_size_spill], // tile_size
                num_blocks,
                q_blocks_offset,
                d_model,
                rdx, rcx); // softmax_head_offset, context_head_offset

            // ═══════════════════════════════════════════════════════════════════
            // INCREMENT KV INDEX AND CONTINUE
            // ═══════════════════════════════════════════════════════════════════

            mov(r8, ptr[rsp + kv_idx_spill]);
            inc(r8);
            mov(ptr[rsp + kv_idx_spill], r8);
            jmp(".kv_loop", T_NEAR);

            L(".kv_end");

            // Context buffer is accumulated in memory (no register store needed)

            // ═══════════════════════════════════════════════════════════════════
            // INCREMENT HEAD OFFSETS AND CONTINUE
            // ═══════════════════════════════════════════════════════════════════

            mov(rax, ptr[rsp + head_offset_spill]);
            add(rax, num_blocks * 36);
            mov(ptr[rsp + head_offset_spill], rax);

            mov(rax, ptr[rsp + softmax_head_offset_spill]);
            add(rax, q_tile_size_ * 8); // 8 bytes per query (max, sum)
            mov(ptr[rsp + softmax_head_offset_spill], rax);

            mov(rax, ptr[rsp + context_head_offset_spill]);
            add(rax, head_dim * 4); // head_dim floats per query
            mov(ptr[rsp + context_head_offset_spill], rax);

            mov(rax, ptr[rsp + head_idx_spill]);
            inc(rax);
            mov(ptr[rsp + head_idx_spill], rax);

            jmp("prefill_head_loop", T_NEAR);

            L("prefill_head_end");

            // ═══════════════════════════════════════════════════════════════════
            // NORMALIZE AND PROJECT: 1/sum normalization + Wo projection
            // ═══════════════════════════════════════════════════════════════════

            emit_prefill_normalize_project(
                reg_Wo, reg_output,
                num_blocks, q_tile_size_,
                softmax_offset, context_offset,
                tile_start_spill, tile_size_spill, d_model, q_local_spill, q_loop_spill);

            // ═══════════════════════════════════════════════════════════════════
            // ADVANCE TO NEXT TILE
            // ═══════════════════════════════════════════════════════════════════

            mov(rax, ptr[rsp + tile_start_spill]);
            add(rax, q_tile_size_);
            mov(ptr[rsp + tile_start_spill], rax);
            jmp("prefill_tile_loop", T_NEAR);

            L("prefill_tile_end");

            // ═══════════════════════════════════════════════════════════════════
            // FUNCTION EPILOGUE
            // ═══════════════════════════════════════════════════════════════════

            add(rsp, stack_size);
            pop_callee_saved();
            ret();

            ready(); // Finalize code generation
        }

        // ═════════════════════════════════════════════════════════════════════
        // PREFILL HELPER METHODS
        // ═════════════════════════════════════════════════════════════════════
        // These methods emit JIT code for specific operations in the prefill
        // kernel. They are called during code generation, not at runtime.

        /**
         * @brief Initialize softmax state for prefill tile
         *
         * Sets up the initial softmax state for all (query, head) pairs in a
         * tile. Each pair gets:
         *   - max = -FLT_MAX (negative infinity approximation)
         *   - sum = 0.0f
         *
         * ═══════════════════════════════════════════════════════════════════════
         * MEMORY LAYOUT
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Softmax state at [rsp + softmax_offset]:
         *   [offset + 0]: max[0] (FP32)
         *   [offset + 4]: sum[0] (FP32)
         *   [offset + 8]: max[1] (FP32)
         *   [offset + 12]: sum[1] (FP32)
         *   ... (tile_size entries total)
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER USAGE
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Scratch (clobbered):
         *   zmm_scratch(0): -FLT_MAX value
         *   zmm_scratch(1): Zero value
         *
         * @param tile_size Number of (query, head) pairs to initialize
         * @param softmax_offset Stack offset to softmax state buffer
         */
        void emit_prefill_init_softmax(int tile_size, int softmax_offset)
        {
            using namespace Xbyak;

            // Load -FLT_MAX for max initialization (approximation of -infinity)
            Zmm zmm_neg_inf = scratch0().zmm();
            load_constant_f32(zmm_neg_inf, -3.4028235e+38f);

            // Zero for sum initialization
            Zmm zmm_zero = scratch1().zmm();
            vxorps(zmm_zero, zmm_zero, zmm_zero);

            // Store (max, sum) pairs for each query
            for (int q = 0; q < tile_size; ++q)
            {
                int offset = softmax_offset + q * 8;                   // 8 bytes per pair
                vmovss(ptr[rsp + offset], Xmm(zmm_neg_inf.getIdx()));  // max
                vmovss(ptr[rsp + offset + 4], Xmm(zmm_zero.getIdx())); // sum
            }
        }

        // NOTE: emit_init_context_registers() and emit_store_context_registers()
        // were removed. The previous optimization of keeping context block 0 in
        // zmm8-15 was incorrect because each head's block 0 is at a DIFFERENT
        // memory location. Head N's accumulation would clobber head N-1's context.
        // All context blocks now use the memory buffer consistently.

        /**
         * @brief Initialize context buffer to zeros for prefill tile
         *
         * Clears the context accumulator buffer to prepare for weighted V
         * accumulation. Context stores intermediate FP32 values before
         * normalization and Wo projection.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * MEMORY LAYOUT
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Context buffer at [rsp + context_offset]:
         *   [offset + 0]: context[0, 0..d_model-1] (d_model floats)
         *   [offset + d_model*4]: context[1, 0..d_model-1]
         *   ... (tile_size queries × d_model floats)
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER USAGE
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Scratch (clobbered):
         *   zmm_scratch(0): Zero value for stores
         *
         * @param tile_size Number of queries in tile
         * @param d_model Total hidden dimension (num_heads × head_dim)
         * @param context_offset Stack offset to context buffer
         */
        void emit_prefill_init_context(int tile_size, int d_model, int context_offset)
        {
            using namespace Xbyak;

            Zmm zmm_zero = scratch0().zmm();
            vxorps(zmm_zero, zmm_zero, zmm_zero);

            // Zero-fill context buffer using 64-byte ZMM stores
            int total_floats = tile_size * d_model;
            // BUG FIX: Original loop was `i < total_floats - 16` which missed last 16 floats
            // Now correctly zeros ALL floats: [0, 16), [16, 32), ... up to total_floats
            for (int i = 0; i < total_floats; i += 16)
            {
                vmovups(ptr[rsp + context_offset + i * 4], zmm_zero);
            }
        }

        /**
         * @brief Vectorized exponential for ZMM register (16-way parallel)
         *
         * Computes exp(x) for all 16 FP32 elements in a ZMM register using
         * range reduction and polynomial approximation:
         *
         *   exp(x) = 2^(x * log2(e))
         *          = 2^n * 2^f  where n = floor(x * log2(e)), f = frac(x * log2(e))
         *          ≈ 2^n * P(f) where P is a polynomial approximation for 2^f
         *
         * ═══════════════════════════════════════════════════════════════════════
         * ALGORITHM
         * ═══════════════════════════════════════════════════════════════════════
         *
         * 1. t = x * log2(e)
         * 2. n = floor(t)              // Integer part
         * 3. f = t - n                 // Fractional part [0, 1)
         * 4. P(f) = c5*f^5 + c4*f^4 + c3*f^3 + c2*f^2 + c1*f + c0
         * 5. result = P(f) * 2^n       // vscalefps instruction
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER USAGE
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Input/Output:
         *   src: Input values (not modified)
         *   dst: Output exp(src) values
         *
         * Constants (preserved):
         *   zmm_log2e(): log2(e) = 1.44269504089f
         *
         * Scratch (clobbered):
         *   zmm_scratch(0): n (floor value)
         *   zmm_scratch(1): f (fractional part)
         *   zmm_scratch(2): P(f) polynomial result
         *   eax: Polynomial coefficients
         *
         * ═══════════════════════════════════════════════════════════════════════
         * POLYNOMIAL COEFFICIENTS (Minimax approximation for 2^x, x∈[0,1])
         * ═══════════════════════════════════════════════════════════════════════
         *
         *   c5 = 0.0013334f (0x3aaec3b4)
         *   c4 = 0.00961812f (0x3c1d955b)
         *   c3 = 0.05550411f (0x3d6356eb)
         *   c2 = 0.24022650f (0x3e75fdf0)
         *   c1 = 0.69314718f (0x3f317218) ≈ ln(2)
         *   c0 = 1.0f (0x3f800000)
         *
         * @param dst Destination ZMM for exp results
         * @param src Source ZMM with input values
         */
        void emit_vec_exp(const Xbyak::Zmm &dst, const Xbyak::Zmm &src)
        {
            using namespace Xbyak;

            Xmm xmm_log2e = Xmm(const_log2e().zmm().getIdx());

            Zmm zmm_n = scratch0().zmm(); // Integer part
            Zmm zmm_f = scratch1().zmm(); // Fractional part
            Zmm zmm_p = scratch2().zmm(); // Polynomial result

            // Step 1: t = x * log2(e)
            vmulps(zmm_f, src, const_log2e().zmm());

            // Step 2: n = floor(t) using vrndscaleps with round-down mode
            vrndscaleps(zmm_n, zmm_f, 0x01);

            // Step 3: f = t - n (fractional part in [0, 1))
            vsubps(zmm_f, zmm_f, zmm_n);

            // Step 4: Evaluate polynomial P(f) using Horner's method
            // P(f) = ((((c5*f + c4)*f + c3)*f + c2)*f + c1)*f + c0

            // c5 * f
            mov(eax, 0x3aaec3b4); // c5 = 0.0013334f
            vpbroadcastd(zmm_p, eax);
            vmulps(zmm_p, zmm_p, zmm_f);

            // + c4
            mov(eax, 0x3c1d955b); // c4 = 0.00961812f
            vpbroadcastd(dst, eax);
            vaddps(zmm_p, zmm_p, dst);
            vmulps(zmm_p, zmm_p, zmm_f);

            // + c3
            mov(eax, 0x3d6356eb); // c3 = 0.05550411f
            vpbroadcastd(dst, eax);
            vaddps(zmm_p, zmm_p, dst);
            vmulps(zmm_p, zmm_p, zmm_f);

            // + c2
            mov(eax, 0x3e75fdf0); // c2 = 0.24022650f
            vpbroadcastd(dst, eax);
            vaddps(zmm_p, zmm_p, dst);
            vmulps(zmm_p, zmm_p, zmm_f);

            // + c1
            mov(eax, 0x3f317218); // c1 = 0.69314718f
            vpbroadcastd(dst, eax);
            vaddps(zmm_p, zmm_p, dst);
            vmulps(zmm_p, zmm_p, zmm_f);

            // + c0 = 1.0f
            mov(eax, 0x3f800000); // c0 = 1.0f
            vpbroadcastd(dst, eax);
            vaddps(zmm_p, zmm_p, dst);

            // Step 5: result = P(f) * 2^n using vscalefps
            vscalefps(dst, zmm_p, zmm_n);
        }

        /**
         * @brief Process attention for all valid queries in prefill tile
         *
         * This is the core prefill attention routine. For each KV position, it:
         * 1. Computes Q·K^T dot products for all queries in tile
         * 2. Updates online softmax (max, sum) for each query
         * 3. Accumulates weighted V into context buffers
         *
         * ═══════════════════════════════════════════════════════════════════════
         * ALGORITHM OVERVIEW
         * ═══════════════════════════════════════════════════════════════════════
         *
         * K/V are loaded once and reused for all queries in the tile. This
         * maximizes data reuse since K/V access is the memory bottleneck.
         *
         * Loop structure (within this method):
         *   for each block b in [0, num_blocks):
         *     Load K[b] into registers (scale, sum_qs, data)
         *     for each query q in tile:
         *       Load Q[q,b] from stack
         *       Compute partial dot product += Q[q,b] · K[b]
         *   Horizontal sum and scale all Q·K scores
         *   Vectorized softmax update for all queries
         *   for each block b:
         *     Load V[b] into registers
         *     for each query q (if valid for this KV position):
         *       context[q,b] = context[q,b] * corr + V[b] * weight
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER ALLOCATION
         * ═══════════════════════════════════════════════════════════════════════
         *
         * ACCUM zone (zmm0-15): Score accumulators for queries
         *   zmm0..zmm(q_tile_size-1): Raw dot product accumulators
         *   After softmax: zmm0..zmm(q_tile_size-1) = corr (broadcasted)
         *                  zmm(q_tile_size)..zmm(2*q_tile_size-1) = weight
         *
         * STATE zone (zmm16-19): K/V data and context
         *   zmm16: V scale (broadcasted)
         *   zmm17: V data low (FP32, 16 elements)
         *   zmm18: V data high (FP32, 16 elements)
         *   zmm19-20: Context accumulators (lo/hi)
         *
         * SCRATCH zone (zmm20-25): Temporaries
         *   zmm20/xmm20: K scale (d_k)
         *   zmm21/ymm21: K sum_qs (8 floats)
         *   zmm22/ymm22: K data (32 int8 values)
         *   zmm23/xmm23: Q scale (d_q), then combined scale
         *   zmm24/ymm24: Q data or correction temp
         *   zmm25/ymm25: Dot product accumulator
         *
         * CONST zone (zmm26-31): Preserved constants
         *   zmm_128(): 0x80 pattern for unsigned→signed conversion
         *   zmm_scale(): 1/sqrt(head_dim) scaling factor
         *
         * GPRs:
         *   rdi: K_ptr (input)
         *   rsi: V_ptr (input)
         *   r8: tile_size (for V loop bounds)
         *   r9: Context pointer calculation
         *   r10: q_local_start (input)
         *   rdx: softmax_head_offset
         *   rcx: context_head_offset
         *
         * ═══════════════════════════════════════════════════════════════════════
         * Q8_1 DOT PRODUCT MATH
         * ═══════════════════════════════════════════════════════════════════════
         *
         * For Q8_1 × Q8_1 dot product:
         *   score = Σ_b (Q[b] · K[b])
         *   Where each block contribution:
         *     raw_dot = Σ_i Q_data[i] * K_data[i]  (VNNI: vpdpbusd)
         *     correction = 16 * sum_qs_k * d_q * d_k  (offset from unsigned conversion)
         *     block_score = (raw_dot - correction) * d_q * d_k
         *
         * @param reg_K_ptr Register holding K cache pointer for this KV position
         * @param reg_V_ptr Register holding V cache pointer for this KV position
         * @param reg_q_local_start First valid query index (for causal masking)
         * @param op_tile_size Number of queries in tile (memory operand)
         * @param num_blocks head_dim / 32 (compile-time constant)
         * @param q_blocks_offset Stack offset to Q blocks buffer
         * @param d_model Total hidden dimension (num_heads × head_dim)
         * @param reg_softmax_head_offset Dynamic offset into softmax buffer for current head
         * @param reg_context_head_offset Dynamic offset into context buffer for current head
         */
        void emit_prefill_tile_attention(
            const Xbyak::Reg64 &reg_K_ptr,
            const Xbyak::Reg64 &reg_V_ptr,
            const Xbyak::Reg64 &reg_q_local_start,
            const Xbyak::Operand &op_tile_size,
            int num_blocks,
            int q_blocks_offset,
            int d_model,
            const Xbyak::Reg64 &reg_softmax_head_offset,
            const Xbyak::Reg64 &reg_context_head_offset)
        {
            using namespace Xbyak;

            // ───────────────────────────────────────────────────────────────────
            // LOCAL LABEL SCOPING
            // ───────────────────────────────────────────────────────────────────
            // CRITICAL: Use local labels to avoid collisions across multiple calls
            // This function is called in a KV loop, so labels must be scoped!
            inLocalLabel();

            // ───────────────────────────────────────────────────────────────────
            // PHASE 1: COMPUTE Q·K^T DOT PRODUCTS
            // ───────────────────────────────────────────────────────────────────
            // Accumulate raw dot products in zmm0..zmm(q_tile_size-1)
            // Each accumulator will hold: Σ_b (Q[q,b] · K[b])

            // Initialize accumulators to zero
            for (int q = 0; q < q_tile_size_; ++q)
            {
                vxorps(Zmm(q), Zmm(q), Zmm(q));
            }

            // Optional: Pre-load Q data into registers if it fits
            // We have zmm16-31 available (16 registers)
            // Each query needs num_blocks registers for data
            // Reuse zmm16-31 since Q is only needed for Phase 1
            bool use_q_registers = (num_blocks * q_tile_size_ <= 16);

            if (use_q_registers)
            {
                // Pre-load Q data for all queries and blocks
                for (int b = 0; b < num_blocks; ++b)
                {
                    int q_off_base = q_blocks_offset + b * 64;
                    for (int q = 0; q < q_tile_size_; ++q)
                    {
                        int q_off = q_off_base + q * num_blocks * 64;
                        int reg_idx = 16 + b * q_tile_size_ + q;
                        vmovdqu8(Ymm(reg_idx), ptr[rsp + q_off + 4]);
                        // Convert unsigned (0..255) to signed (-128..127) for vpdpbusd
                        vpxord(Ymm(reg_idx), Ymm(reg_idx), Ymm(const_128().zmm().getIdx()));
                    }
                }
            }

            // ───────────────────────────────────────────────────────────────────
            // BLOCK LOOP: Accumulate dot products across all blocks
            // ───────────────────────────────────────────────────────────────────
            for (int b = 0; b < num_blocks; ++b)
            {
                int k_off = b * 36;                        // K block offset (36 bytes per Q8_1 block)
                int q_off_base = q_blocks_offset + b * 64; // Q block base on stack (64-byte aligned)

                // Register assignments for K block data
                Xmm xmm_d_k(20);          // K scale
                Ymm ymm_sum_qs_k(21);     // K sum_qs (8 floats for correction)
                Xmm xmm_sum_qs_k_low(21); // Low 128 bits for load
                Ymm ymm_k(22);            // K data (32 int8 values)

                // Load K block: scale (FP16 → FP32)
                vpbroadcastw(xmm_d_k, ptr[reg_K_ptr + k_off]);
                vcvtph2ps(xmm_d_k, xmm_d_k);

                // Load K block: sum_qs (INT16 → FP32, broadcasted to 8 lanes)
                vpbroadcastw(xmm_sum_qs_k_low, ptr[reg_K_ptr + k_off + 2]);
                vpmovsxwd(ymm_sum_qs_k, xmm_sum_qs_k_low);
                vcvtdq2ps(ymm_sum_qs_k, ymm_sum_qs_k);

                // Load K block: data (32 int8 values)
                vmovdqu8(ymm_k, ptr[reg_K_ptr + k_off + 4]);

                // ───────────────────────────────────────────────────────────────
                // QUERY LOOP: Compute partial dot product for each query
                // ───────────────────────────────────────────────────────────────
                for (int q = 0; q < q_tile_size_; ++q)
                {
                    int q_off = q_off_base + q * num_blocks * 64;

                    // Load Q scale
                    Xmm xmm_d_q(23);
                    vpbroadcastw(xmm_d_q, ptr[rsp + q_off]);
                    vcvtph2ps(xmm_d_q, xmm_d_q);

                    // Dot product accumulator for this block
                    Ymm ymm_dot(25);
                    vxorps(ymm_dot, ymm_dot, ymm_dot);

                    if (use_q_registers)
                    {
                        // Use pre-loaded Q data
                        int reg_idx = 16 + b * q_tile_size_ + q;
                        vpdpbusd(ymm_dot, Ymm(reg_idx), ymm_k);
                    }
                    else
                    {
                        // Load Q data from stack
                        Ymm ymm_q(24);
                        vmovdqu8(ymm_q, ptr[rsp + q_off + 4]);
                        // Convert unsigned to signed for VNNI
                        vpxord(ymm_q, ymm_q, Ymm(const_128().zmm().getIdx()));
                        // VNNI dot product: result += Q[i] * K[i] (accumulated in 8 lanes)
                        vpdpbusd(ymm_dot, ymm_q, ymm_k);
                    }

                    // Convert accumulated int32 to float
                    vcvtdq2ps(ymm_dot, ymm_dot);

                    // Compute combined scale: d_q * d_k
                    Xmm xmm_scale(23);
                    Ymm ymm_scale(23);
                    vmulps(xmm_scale, xmm_d_q, xmm_d_k);
                    vbroadcastss(ymm_scale, xmm_scale);
                    vmulps(ymm_dot, ymm_dot, ymm_scale);

                    // Compute correction term: 16 * sum_qs_k * d_q * d_k
                    // The 16 factor comes from the unsigned→signed conversion offset
                    Xmm xmm_corr_low(24);
                    Ymm ymm_corr(24);
                    mov(eax, 0x41800000); // 16.0f
                    vmovd(xmm_corr_low, eax);
                    vbroadcastss(ymm_corr, xmm_corr_low);
                    vmulps(ymm_corr, ymm_corr, ymm_sum_qs_k);
                    vmulps(ymm_corr, ymm_corr, ymm_scale);

                    // Subtract correction
                    vsubps(ymm_dot, ymm_dot, ymm_corr);

                    // Accumulate into query's running total
                    vaddps(Zmm(q), Zmm(q), Zmm(ymm_dot.getIdx()));
                }
            }

            // ───────────────────────────────────────────────────────────────────
            // PHASE 1.5: HORIZONTAL SUM AND SCALE
            // ───────────────────────────────────────────────────────────────────
            // Each zmm0..zmm(q_tile_size-1) contains partial sums in 8 lanes
            // Reduce to scalar and apply 1/sqrt(head_dim) scaling

            vmovss(xmm8, Xmm(const_scale().zmm().getIdx())); // Load 1/sqrt(head_dim)
            for (int q = 0; q < q_tile_size_; ++q)
            {
                // Reduce YMM to scalar: sum all 8 lanes
                emit_horizontal_sum_to_scalar(Zmm(q));
                // Apply attention scale: score *= 1/sqrt(head_dim)
                vmulss(Xmm(Zmm(q).getIdx()), Xmm(Zmm(q).getIdx()), xmm8);
            }

            // ───────────────────────────────────────────────────────────────────
            // PHASE 2: VECTORIZED SOFTMAX UPDATE
            // ───────────────────────────────────────────────────────────────────
            // Uses ZMM-based parallel exp for all queries in tile
            // This updates the running (max, sum) state and computes correction
            // factors for the existing context accumulator

            emit_prefill_tile_softmax_vectorized(reg_q_local_start, op_tile_size, reg_softmax_head_offset);

            // After vectorized softmax, registers contain:
            //   zmm0..zmm(q_tile_size-1) = corr[q] (broadcasted scalar)
            //     where corr = exp(old_max - new_max)
            //   zmm(q_tile_size)..zmm(2*q_tile_size-1) = weight[q] (broadcasted)
            //     where weight = exp(score - new_max)
            // Softmax state in memory is updated with new max and sum

            // Save tile_size to r8 for V accumulation bounds checking
            mov(r8, op_tile_size);

            // ───────────────────────────────────────────────────────────────────
            // PHASE 3: WEIGHTED V ACCUMULATION
            // ───────────────────────────────────────────────────────────────────
            // For each V block:
            //   context[q,block] = context[q,block] * corr[q] + V[block] * weight[q]
            //
            // All context blocks use the memory buffer on stack. This is correct
            // because each head's context offset is different, and register
            // persistence across heads would cause incorrect accumulation.
            //
            // OPTIMIZATION: Pre-multiply V * scale once per block, then apply
            // per-query weight using FMA instructions.

            for (int b = 0; b < num_blocks; ++b)
            {
                int v_off = b * 36;            // V block offset (36 bytes per Q8_1 block)
                int ctx_off_base = b * 32 * 4; // Context offset (32 floats × 4 bytes)

                // Load V block: scale (FP16 → FP32, broadcasted)
                vpbroadcastw(xmm16, ptr[reg_V_ptr + v_off]);
                vcvtph2ps(xmm16, xmm16);
                vbroadcastss(zmm16, xmm16);

                // Load V block: data (32 int8 → 32 FP32, split into lo/hi)
                // zmm17: V[0..15] as FP32
                // zmm18: V[16..31] as FP32
                vpmovsxbd(zmm17, ptr[reg_V_ptr + v_off + 4]);      // Low 16 bytes
                vpmovsxbd(zmm18, ptr[reg_V_ptr + v_off + 4 + 16]); // High 16 bytes
                vcvtdq2ps(zmm17, zmm17);
                vcvtdq2ps(zmm18, zmm18);

                // Pre-multiply V * scale (shared across all queries)
                vmulps(zmm17, zmm17, zmm16); // V_lo_scaled = V_lo * scale
                vmulps(zmm18, zmm18, zmm16); // V_hi_scaled = V_hi * scale

                // Process each query in tile
                for (int q = 0; q < q_tile_size_; ++q)
                {
                    std::string skip_v = ".skip_v_" + std::to_string(b) + "_" + std::to_string(q);

                    // Bounds check: skip if q < q_local_start (causal mask)
                    cmp(reg_q_local_start, q);
                    jg(skip_v, T_NEAR);

                    // Bounds check: skip if q >= tile_size (partial tile)
                    cmp(r8, q);
                    jle(skip_v, T_NEAR);

                    // Compute context address (compile-time offset optimization)
                    // context_ptr = rsp + context_head_offset + q * d_model * 4 + ctx_off_base
                    int q_ctx_offset = q * d_model * 4 + ctx_off_base;
                    mov(r9, reg_context_head_offset);
                    add(r9, rsp);

                    // Load existing context accumulator
                    Zmm zmm_ctx_lo = zmm19;
                    Zmm zmm_ctx_hi = zmm20;
                    vmovups(zmm_ctx_lo, ptr[r9 + q_ctx_offset]);
                    vmovups(zmm_ctx_hi, ptr[r9 + q_ctx_offset + 64]);

                    // Compute weighted V: V_scaled * weight[q]
                    // weight[q] is in Zmm(q + q_tile_size_)
                    Zmm zmm_wv_lo = zmm21;
                    Zmm zmm_wv_hi = zmm22;
                    Zmm zmm_weight_q = Zmm(q + q_tile_size_);

                    vmulps(zmm_wv_lo, zmm17, zmm_weight_q); // V_lo_scaled * weight[q]
                    vmulps(zmm_wv_hi, zmm18, zmm_weight_q); // V_hi_scaled * weight[q]

                    // Update context: ctx = ctx * corr[q] + weighted_V
                    // Use FMA: vfmadd231ps(a, b, c) = a + b*c
                    // We want: ctx = ctx * corr + wv
                    // Rewrite as: ctx = wv + ctx * corr → vfmadd231ps(wv, ctx, corr)
                    vfmadd231ps(zmm_wv_lo, zmm_ctx_lo, Zmm(q)); // wv_lo += ctx_lo * corr[q]
                    vfmadd231ps(zmm_wv_hi, zmm_ctx_hi, Zmm(q)); // wv_hi += ctx_hi * corr[q]

                    // Store updated context (result is now in zmm_wv)
                    vmovups(ptr[r9 + q_ctx_offset], zmm_wv_lo);
                    vmovups(ptr[r9 + q_ctx_offset + 64], zmm_wv_hi);

                    L(skip_v);
                }
            }

            outLocalLabel();
        }

        /**
         * @brief Copy Q blocks for tile queries from input to stack buffer
         *
         * Copies Q8_1 blocks for all queries in the current tile to a contiguous
         * stack buffer. This enables efficient repeated access during the KV loop
         * since the Q data layout is optimized for the tiled attention pattern.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * MEMORY LAYOUT
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Source (input Q tensor): Layout is [seq_len, num_heads, head_dim] packed
         *   Q[query_idx, head_idx, block] at:
         *     reg_Q + query_idx * q_stride + head_idx * num_blocks * 36 + block * 36
         *   Where q_stride = num_heads * num_blocks * 36
         *
         * Destination (stack buffer): Tile-optimized layout
         *   Q[q_local, block] at:
         *     rsp + q_blocks_offset + q_local * num_blocks * 64 + block * 64
         *   64-byte stride (not 36) for cache line alignment
         *
         * Q8_1 block format (36 bytes):
         *   [0-1]: scale (FP16)
         *   [2-3]: sum_qs (INT16)
         *   [4-35]: data (32 × INT8)
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER USAGE
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Input:
         *   reg_Q: Base pointer to Q tensor
         *   reg_head_offset: Byte offset to current head within Q
         *
         * Scratch (clobbered):
         *   r8: Loop counter (q_local)
         *   r9: Source pointer calculation
         *   r10: Destination offset calculation
         *   eax: 4-byte transfer temp
         *   ymm_scratch(0): 32-byte data copy
         *
         * @param reg_Q Base pointer to Q tensor
         * @param reg_head_offset Offset to current head (= head_idx * num_blocks * 36)
         * @param num_blocks Number of Q8_1 blocks per head (head_dim / 32)
         * @param q_blocks_offset Stack offset to Q tile buffer
         * @param tile_start_spill Stack offset to tile_start variable
         * @param tile_size_spill Stack offset to tile_size variable
         */
        void emit_prefill_copy_q_tile(
            const Xbyak::Reg64 &reg_Q,
            const Xbyak::Reg64 &reg_head_offset,
            int num_blocks,
            int q_blocks_offset,
            int tile_start_spill,
            int tile_size_spill)
        {
            using namespace Xbyak;

            // Q stride: bytes to advance for each query position
            int q_stride = config_.num_heads * num_blocks * 36;

            inLocalLabel();

            // ───────────────────────────────────────────────────────────────────
            // COPY LOOP: Copy Q blocks for each query in tile
            // ───────────────────────────────────────────────────────────────────

            xor_(r8, r8); // q_local = 0

            L(".copy_loop");
            cmp(r8, ptr[rsp + tile_size_spill]); // while (q_local < tile_size)
            jge(".copy_end", T_NEAR);

            // Calculate source address:
            //   src = reg_Q + (tile_start + q_local) * q_stride + head_offset
            mov(r9, ptr[rsp + tile_start_spill]);
            add(r9, r8);              // tile_start + q_local
            imul(r9, r9, q_stride);   // * q_stride
            add(r9, reg_head_offset); // + head_offset
            add(r9, reg_Q);           // + base pointer

            // Calculate destination offset on stack:
            //   dst = rsp + q_blocks_offset + q_local * num_blocks * 64
            mov(r10, r8);
            imul(r10, r10, num_blocks * 64);
            add(r10, q_blocks_offset);

            // Copy all blocks for this query
            for (int b = 0; b < num_blocks; ++b)
            {
                int src_b = b * 36; // Source: 36-byte blocks
                int dst_b = b * 64; // Dest: 64-byte aligned slots

                // Copy scale + sum_qs (4 bytes)
                mov(eax, ptr[r9 + src_b]);
                mov(ptr[rsp + r10 + dst_b], eax);

                // Copy data (32 bytes) using YMM
                Ymm ymm_tmp = Ymm(scratch0().zmm().getIdx());
                vmovdqu8(ymm_tmp, ptr[r9 + src_b + 4]);
                vmovdqu8(ptr[rsp + r10 + dst_b + 4], ymm_tmp);
            }

            inc(r8);
            jmp(".copy_loop", T_NEAR);

            L(".copy_end");

            outLocalLabel();
        }

        /**
         * @brief Process attention for one query in prefill tile (scalar path)
         *
         * This is the scalar fallback for processing a single query. It handles
         * the complete attention flow: Q·K dot product, online softmax update,
         * and weighted V accumulation. Used when vectorized tile processing
         * is not applicable.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * ALGORITHM
         * ═══════════════════════════════════════════════════════════════════════
         *
         * 1. Compute Q·K dot product (scaled)
         *    score = (Q · K) / sqrt(head_dim)
         *
         * 2. Online softmax update:
         *    new_max = max(old_max, score)
         *    corr = exp(old_max - new_max)    // Correction for existing sum
         *    weight = exp(score - new_max)    // Attention weight
         *    new_sum = old_sum * corr + weight
         *
         * 3. V accumulation with correction:
         *    context = context * corr + V * weight
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER USAGE
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Input:
         *   reg_K_ptr: Pointer to K cache for this KV position
         *   reg_V_ptr: Pointer to V cache for this KV position
         *   reg_q_local: Query index within tile [0, tile_size)
         *   reg_softmax_head_offset: Offset to softmax state for current head
         *   reg_context_head_offset: Offset to context buffer for current head
         *
         * STATE zone (zmm16-19):
         *   zmm16/xmm16: max (from zmm_max())
         *   zmm17/xmm17: sum (from zmm_sum())
         *   zmm18/xmm18: weight (from zmm_weight())
         *   zmm19/xmm19: corr (from zmm_corr())
         *
         * SCRATCH zone (clobbered):
         *   zmm_scratch(0)/xmm: score
         *   zmm_scratch(1)/xmm: scale temp
         *   zmm_scratch(2)/xmm: new_max
         *   r11: Pointer calculations
         *
         * @param reg_K_ptr K cache pointer for this KV position
         * @param reg_V_ptr V cache pointer for this KV position
         * @param reg_q_local Query index within tile
         * @param num_blocks head_dim / 32
         * @param q_blocks_offset Stack offset to Q blocks buffer
         * @param d_model num_heads × head_dim
         * @param reg_softmax_head_offset Dynamic offset to softmax state
         * @param reg_context_head_offset Dynamic offset to context buffer
         */
        void emit_prefill_query_attention(
            const Xbyak::Reg64 &reg_K_ptr,
            const Xbyak::Reg64 &reg_V_ptr,
            const Xbyak::Reg64 &reg_q_local,
            int num_blocks,
            int q_blocks_offset,
            int d_model,
            const Xbyak::Reg64 &reg_softmax_head_offset,
            const Xbyak::Reg64 &reg_context_head_offset)
        {
            using namespace Xbyak;

            int head_dim = num_blocks * 32;

            // ───────────────────────────────────────────────────────────────────
            // STEP 1: Calculate Q base address on stack
            // ───────────────────────────────────────────────────────────────────
            // Q_base = rsp + q_blocks_offset + q_local * num_blocks * 64
            Reg64 reg_q_base = r11;
            mov(reg_q_base, reg_q_local);
            imul(reg_q_base, reg_q_base, num_blocks * 64);
            add(reg_q_base, q_blocks_offset);
            add(reg_q_base, rsp);

            // ───────────────────────────────────────────────────────────────────
            // STEP 2: Compute Q·K dot product
            // ───────────────────────────────────────────────────────────────────
            Xmm xmm_score = Xmm(scratch0().zmm().getIdx());
            Xmm xmm_scale_tmp = Xmm(scratch1().zmm().getIdx());
            vmovss(xmm_scale_tmp, Xmm(const_scale().zmm().getIdx())); // 1/sqrt(head_dim)

            emit_prefill_dot_product(xmm_score, reg_K_ptr, reg_q_base, num_blocks, xmm_scale_tmp);

            // ───────────────────────────────────────────────────────────────────
            // STEP 3: Load softmax state for this query
            // ───────────────────────────────────────────────────────────────────
            // sm_ptr = rsp + softmax_head_offset + q_local * 8
            Reg64 reg_sm = r11; // Reuse r11
            mov(reg_sm, reg_q_local);
            shl(reg_sm, 3); // * 8 (sizeof max+sum)
            add(reg_sm, reg_softmax_head_offset);
            add(reg_sm, rsp);

            // Load current max and sum from state
            Xmm xmm_max = Xmm(state_max().zmm().getIdx()); // zmm16
            Xmm xmm_sum = Xmm(state_sum().zmm().getIdx()); // zmm17
            vmovss(xmm_max, ptr[reg_sm]);
            vmovss(xmm_sum, ptr[reg_sm + 4]);

            // ───────────────────────────────────────────────────────────────────
            // STEP 4: Online softmax update
            // ───────────────────────────────────────────────────────────────────
            Xmm xmm_new_max = Xmm(scratch2().zmm().getIdx()); // zmm22
            Xmm xmm_corr = Xmm(state_corr().zmm().getIdx());
            Xmm xmm_weight = Xmm(state_weight().zmm().getIdx());

            // new_max = max(old_max, score)
            vmaxss(xmm_new_max, xmm_score, xmm_max);

            // corr = exp(old_max - new_max)
            // This factor rescales the existing accumulated sum
            vsubss(xmm_corr, xmm_max, xmm_new_max);
            emit_prefill_exp_scalar(xmm_corr, xmm_corr);

            // weight = exp(score - new_max)
            // This is the attention weight for this KV position
            vsubss(xmm_weight, xmm_score, xmm_new_max);
            emit_prefill_exp_scalar(xmm_weight, xmm_weight);

            // new_sum = old_sum * corr + weight
            vmulss(xmm_sum, xmm_sum, xmm_corr);
            vaddss(xmm_sum, xmm_sum, xmm_weight);

            // Store updated softmax state
            vmovss(ptr[reg_sm], xmm_new_max);
            vmovss(ptr[reg_sm + 4], xmm_sum);

            // ───────────────────────────────────────────────────────────────────
            // STEP 5: Broadcast weight and corr for V accumulation
            // ───────────────────────────────────────────────────────────────────
            vbroadcastss(state_weight().zmm(), xmm_weight);
            vbroadcastss(state_corr().zmm(), xmm_corr);

            // ───────────────────────────────────────────────────────────────────
            // STEP 6: Calculate context pointer and accumulate V
            // ───────────────────────────────────────────────────────────────────
            // ctx_ptr = rsp + context_head_offset + q_local * d_model * 4
            Reg64 reg_ctx = r11;
            mov(reg_ctx, reg_q_local);
            imul(reg_ctx, reg_ctx, d_model * 4);
            add(reg_ctx, reg_context_head_offset);
            add(reg_ctx, rsp);

            // Accumulate weighted V into context
            emit_prefill_v_accum(reg_V_ptr, reg_ctx, num_blocks);
        }

        /**
         * @brief Q8_1 dot product for prefill (accumulate across all blocks)
         *
         * Computes the scaled dot product between Q and K vectors in Q8_1
         * format. Accumulates partial results across all blocks before
         * performing the horizontal sum, which is more efficient than
         * summing after each block.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * ALGORITHM
         * ═══════════════════════════════════════════════════════════════════════
         *
         * For each block b:
         *   raw_dot = Σ_i Q[i] * K[i]           // VNNI: vpdpbusd
         *   scale = d_q[b] * d_k[b]             // Combined scale factor
         *   correction = 16 * sum_qs_k * scale  // Offset from unsigned conversion
         *   block_result = (raw_dot - correction) * scale
         *   acc += block_result
         *
         * Final: dst = horizontal_sum(acc) * attention_scale
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER USAGE
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Output:
         *   dst: Final scaled dot product (scalar FP32)
         *
         * Input:
         *   reg_K: K block pointer
         *   reg_Q: Q block pointer (on stack)
         *   scale: Attention scale (1/sqrt(head_dim))
         *
         * Accumulator:
         *   ymm7: Running sum across blocks (8 FP32 lanes)
         *
         * Scratch (clobbered):
         *   ymm4: Q data
         *   ymm5: K data
         *   ymm6: Dot product result
         *   xmm8: Q scale (d_q)
         *   xmm9: K scale (d_k)
         *   xmm10: Combined scale (d_q * d_k)
         *   xmm14: Correction term
         *   xmm15: K sum_qs
         *
         * @param dst Output XMM register for final scalar result
         * @param reg_K K cache pointer for this KV position
         * @param reg_Q Q pointer on stack
         * @param num_blocks Number of Q8_1 blocks
         * @param scale Attention scale (1/sqrt(head_dim))
         */
        void emit_prefill_dot_product(
            const Xbyak::Xmm &dst,
            const Xbyak::Reg64 &reg_K,
            const Xbyak::Reg64 &reg_Q,
            int num_blocks,
            const Xbyak::Xmm &scale)
        {
            using namespace Xbyak;

            // Accumulator for scaled partial results (YMM = 8 FP32 lanes)
            Ymm ymm_acc(7);
            vxorps(ymm_acc, ymm_acc, ymm_acc);

            // Scratch registers for dot product computation
            Ymm ymm_q(4);            // Q data (32 int8)
            Ymm ymm_k(5);            // K data (32 int8)
            Ymm ymm_dot(6);          // Dot product accumulator
            Xmm xmm_d_q(8);          // Q scale
            Xmm xmm_d_k(9);          // K scale
            Xmm xmm_sum_qs_k(15);    // K sum_qs for correction
            Xmm xmm_correction(14);  // Correction term
            Xmm xmm_scale_block(10); // Combined scale (d_q * d_k)

            // ───────────────────────────────────────────────────────────────────
            // BLOCK LOOP: Accumulate scaled dot products
            // ───────────────────────────────────────────────────────────────────
            for (int b = 0; b < num_blocks; ++b)
            {
                int q_off = b * 36; // Q block is at 36-byte stride on stack
                int k_off = b * 36; // K block is at 36-byte stride

                // Load Q scale (FP16 → FP32)
                vpbroadcastw(xmm_d_q, ptr[reg_Q + q_off]);
                vcvtph2ps(xmm_d_q, xmm_d_q);

                // Load Q data (32 × int8, convert unsigned → signed)
                vmovdqu8(ymm_q, ptr[reg_Q + q_off + 4]);
                vpxord(ymm_q, ymm_q, Ymm(const_128().zmm().getIdx()));

                // Load K scale (FP16 → FP32)
                vpbroadcastw(xmm_d_k, ptr[reg_K + k_off]);
                vcvtph2ps(xmm_d_k, xmm_d_k);

                // Load K sum_qs (INT16 → FP32 for correction)
                vpbroadcastw(xmm_sum_qs_k, ptr[reg_K + k_off + 2]);
                vpmovsxwd(xmm_sum_qs_k, xmm_sum_qs_k);
                vcvtdq2ps(xmm_sum_qs_k, xmm_sum_qs_k);

                // Load K data (32 × int8)
                vmovdqu8(ymm_k, ptr[reg_K + k_off + 4]);

                // VNNI dot product: accumulates 4 products per lane → 8 int32 results
                vxorps(ymm_dot, ymm_dot, ymm_dot);
                vpdpbusd(ymm_dot, ymm_q, ymm_k);

                // Convert int32 results to float
                vcvtdq2ps(ymm_dot, ymm_dot);

                // Compute combined scale: d_q * d_k
                vmulps(xmm_scale_block, xmm_d_q, xmm_d_k);
                vbroadcastss(Ymm(xmm_scale_block.getIdx()), xmm_scale_block);

                // Apply scale to dot product
                vmulps(ymm_dot, ymm_dot, Ymm(xmm_scale_block.getIdx()));

                // Compute correction: 16 * sum_qs_k * (d_q * d_k)
                // The 16 factor accounts for unsigned→signed offset
                mov(eax, 0x41800000); // 16.0f
                vmovd(xmm_correction, eax);
                vbroadcastss(Ymm(xmm_correction.getIdx()), xmm_correction);

                vbroadcastss(Ymm(xmm_sum_qs_k.getIdx()), xmm_sum_qs_k);
                vmulps(Ymm(xmm_correction.getIdx()), Ymm(xmm_correction.getIdx()), Ymm(xmm_sum_qs_k.getIdx()));
                vmulps(Ymm(xmm_correction.getIdx()), Ymm(xmm_correction.getIdx()), Ymm(xmm_scale_block.getIdx()));

                // Subtract correction from dot product
                vsubps(ymm_dot, ymm_dot, Ymm(xmm_correction.getIdx()));

                // Accumulate into running total
                vaddps(ymm_acc, ymm_acc, ymm_dot);
            }

            // ───────────────────────────────────────────────────────────────────
            // HORIZONTAL SUM: Reduce 8-lane YMM to scalar
            // ───────────────────────────────────────────────────────────────────
            Xmm xmm_tmp = Xmm(ymm_dot.getIdx()); // Reuse ymm_dot as temp

            // Extract high 128 bits and add to low
            vextractf128(xmm_tmp, ymm_acc, 1);
            vaddps(dst, Xmm(ymm_acc.getIdx()), xmm_tmp);

            // Reduce 4 lanes to 2
            vshufps(xmm_tmp, dst, dst, 0x4E); // [2,3,0,1]
            vaddps(dst, dst, xmm_tmp);

            // Reduce 2 lanes to 1
            vshufps(xmm_tmp, dst, dst, 0xB1); // [1,0,3,2]
            vaddps(dst, dst, xmm_tmp);

            // Apply attention scale: score *= 1/sqrt(head_dim)
            vmulss(dst, dst, scale);
        }

        /**
         * @brief Vectorized softmax update for all queries in prefill tile
         *
         * This method performs the online softmax update for all queries in the
         * tile simultaneously using AVX-512 parallel operations. It replaces
         * the scalar loop approach with a single vectorized computation.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * ALGORITHM OVERVIEW
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Standard online softmax for single query:
         *   new_max = max(old_max, score)
         *   corr = exp(old_max - new_max)
         *   weight = exp(score - new_max)
         *   new_sum = old_sum * corr + weight
         *
         * Vectorized version (all queries in parallel):
         *   1. Pack scores[0..q_tile_size-1] into ZMM low elements
         *   2. Pack max[0..q_tile_size-1] into ZMM low elements
         *   3. Pack sum[0..q_tile_size-1] into ZMM low elements
         *   4. Compute all new_max, corr_input, weight_input in parallel
         *   5. Two emit_vec_exp() calls for corr and weight vectors
         *   6. Update sums and scatter back to memory
         *   7. Broadcast corr/weight for V accumulation
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER USAGE
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Input (scores):
         *   zmm0..zmm(q_tile_size-1): Scalar scores in element[0]
         *
         * Output:
         *   zmm0..zmm(q_tile_size-1): Broadcasted corr values
         *   zmm(q_tile_size)..zmm(2*q_tile_size-1): Broadcasted weight values
         *
         * STATE zone (temporary packing):
         *   zmm16: Packed old_max values
         *   zmm17: Packed old_sum values
         *   zmm18: Packed new_max values
         *   zmm19: Packed corr values
         *
         * SCRATCH zone:
         *   zmm20: Packed scores
         *   zmm21: Weight input (score - new_max)
         *   zmm22: Corr input (old_max - new_max)
         *   zmm23-25: exp() computation temps
         *
         * ═══════════════════════════════════════════════════════════════════════
         * CAUSAL MASKING
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Invalid queries (q < q_local_start due to causal mask) get:
         *   corr = 1.0 (identity for existing context)
         *   weight = 0.0 (no contribution from this KV position)
         *
         * @param reg_q_local_start First valid query index (causal cutoff)
         * @param op_tile_size Actual number of queries in tile
         * @param reg_softmax_head_offset Dynamic offset to softmax state buffer
         */
        void emit_prefill_tile_softmax_vectorized(
            const Xbyak::Reg64 &reg_q_local_start,
            const Xbyak::Operand &op_tile_size,
            const Xbyak::Reg64 &reg_softmax_head_offset)
        {
            using namespace Xbyak;

            debug_emit("emit_prefill_tile_softmax_vectorized");

            // ───────────────────────────────────────────────────────────────────
            // REGISTER ALLOCATION FOR VECTORIZED SOFTMAX
            // ───────────────────────────────────────────────────────────────────
            // Input scores: zmm0..zmm(q_tile_size-1) element[0]
            // zmm16: packed old_max values (low q_tile_size elements)
            // zmm17: packed old_sum values (low q_tile_size elements)
            // zmm18: packed scores (low q_tile_size elements)
            // zmm19: packed new_max
            // zmm20: corr_input (old_max - new_max) -> after exp: corr
            // zmm21: weight_input (score - new_max) -> after exp: weight
            // zmm22-25: scratch for exp_emitter_

            // ───────────────────────────────────────────────────────────────────
            // STEP 1: Pack scores from individual ZMMs into zmm18
            // ───────────────────────────────────────────────────────────────────
            // Each score is in element[0] of its respective ZMM (zmm0..zmm3)
            // We gather them into the low elements of zmm18 for parallel processing
            Zmm zmm_scores = Zmm(18);
            vxorps(zmm_scores, zmm_scores, zmm_scores); // Clear all elements first

            Xmm xmm_scores = Xmm(18);
            if (q_tile_size_ >= 1)
                vmovss(xmm_scores, Xmm(0)); // scores[0] = zmm0[0]
            if (q_tile_size_ >= 2)
                vinsertps(xmm_scores, xmm_scores, Xmm(1), 0x10); // scores[1] = zmm1[0]
            if (q_tile_size_ >= 3)
                vinsertps(xmm_scores, xmm_scores, Xmm(2), 0x20); // scores[2] = zmm2[0]
            if (q_tile_size_ >= 4)
                vinsertps(xmm_scores, xmm_scores, Xmm(3), 0x30); // scores[3] = zmm3[0]
            // Note: For q_tile_size > 4, would need YMM upper portion handling

            // ───────────────────────────────────────────────────────────────────
            // STEP 1.5: APPLY CAUSAL MASKING TO SCORES BEFORE SOFTMAX
            // ───────────────────────────────────────────────────────────────────
            // Set scores[q] = -infinity for q < q_local_start (causally masked)
            // and for q >= tile_size (partial tile). This ensures:
            //   - new_max = old_max (since -inf < old_max)
            //   - corr = 1.0 (exp(0) = 1)
            //   - weight = 0.0 (exp(-inf) = 0)
            // And crucially: softmax state is NOT updated for masked queries.

            // Load -FLT_MAX for masking
            Xmm xmm_neg_inf = Xmm(scratch0().zmm().getIdx()); // xmm20
            load_constant_f32(Zmm(scratch0().zmm().getIdx()), -3.4028235e+38f);

            mov(r8, op_tile_size);
            for (int q = 0; q < q_tile_size_; ++q)
            {
                std::string valid_label = ".mask_valid_" + std::to_string(q);
                std::string done_label = ".mask_done_" + std::to_string(q);

                // Check: q_local_start > q means this query is masked
                cmp(reg_q_local_start, q);
                jle(valid_label, T_NEAR);

                // Masked: set score[q] = -inf
                // Use insertps to set a single element
                if (q == 0)
                    vinsertps(xmm_scores, xmm_scores, xmm_neg_inf, 0x00);
                else if (q == 1)
                    vinsertps(xmm_scores, xmm_scores, xmm_neg_inf, 0x10);
                else if (q == 2)
                    vinsertps(xmm_scores, xmm_scores, xmm_neg_inf, 0x20);
                else if (q == 3)
                    vinsertps(xmm_scores, xmm_scores, xmm_neg_inf, 0x30);
                jmp(done_label, T_NEAR);

                L(valid_label);
                // Also check tile_size <= q (partial tile)
                cmp(r8, q);
                jg(done_label, T_NEAR);

                // Outside tile: set score[q] = -inf
                if (q == 0)
                    vinsertps(xmm_scores, xmm_scores, xmm_neg_inf, 0x00);
                else if (q == 1)
                    vinsertps(xmm_scores, xmm_scores, xmm_neg_inf, 0x10);
                else if (q == 2)
                    vinsertps(xmm_scores, xmm_scores, xmm_neg_inf, 0x20);
                else if (q == 3)
                    vinsertps(xmm_scores, xmm_scores, xmm_neg_inf, 0x30);

                L(done_label);
            }

            // ───────────────────────────────────────────────────────────────────
            // STEP 2: Load max/sum state for all queries from memory
            // ───────────────────────────────────────────────────────────────────
            // State layout in memory: [max0, sum0, max1, sum1, ...]
            // Each (max, sum) pair is 8 bytes
            //
            // We load into zmm16 (maxs) and zmm17 (sums) using scalar loads
            // and insertps to build up the packed vectors. This avoids the
            // complexity of cross-lane shuffle operations for deinterleaving.
            Zmm zmm_max = Zmm(16);
            Zmm zmm_sum = Zmm(17);
            vxorps(zmm_max, zmm_max, zmm_max);
            vxorps(zmm_sum, zmm_sum, zmm_sum);

            lea(r9, ptr[rsp + reg_softmax_head_offset]);

            // Load interleaved [max, sum] pairs and deinterleave using insertps
            // Memory: [max0, sum0, max1, sum1, max2, sum2, max3, sum3]
            // Result: zmm16[0..3] = [max0, max1, max2, max3]
            //         zmm17[0..3] = [sum0, sum1, sum2, sum3]
            Xmm xmm_max = Xmm(16);
            Xmm xmm_sum = Xmm(17);
            Xmm xmm_tmp = Xmm(22);

            // Load and deinterleave each (max, sum) pair
            // Query 0: max at offset 0, sum at offset 4
            vmovss(xmm_max, ptr[r9 + 0]); // max0
            vmovss(xmm_sum, ptr[r9 + 4]); // sum0

            if (q_tile_size_ >= 2)
            {
                // Query 1: max at offset 8, sum at offset 12
                vmovss(xmm_tmp, ptr[r9 + 8]);
                vinsertps(xmm_max, xmm_max, xmm_tmp, 0x10); // max1 -> slot 1
                vmovss(xmm_tmp, ptr[r9 + 12]);
                vinsertps(xmm_sum, xmm_sum, xmm_tmp, 0x10); // sum1 -> slot 1
            }
            if (q_tile_size_ >= 3)
            {
                // Query 2: max at offset 16, sum at offset 20
                vmovss(xmm_tmp, ptr[r9 + 16]);
                vinsertps(xmm_max, xmm_max, xmm_tmp, 0x20); // max2 -> slot 2
                vmovss(xmm_tmp, ptr[r9 + 20]);
                vinsertps(xmm_sum, xmm_sum, xmm_tmp, 0x20); // sum2 -> slot 2
            }
            if (q_tile_size_ >= 4)
            {
                // Query 3: max at offset 24, sum at offset 28
                vmovss(xmm_tmp, ptr[r9 + 24]);
                vinsertps(xmm_max, xmm_max, xmm_tmp, 0x30); // max3 -> slot 3
                vmovss(xmm_tmp, ptr[r9 + 28]);
                vinsertps(xmm_sum, xmm_sum, xmm_tmp, 0x30); // sum3 -> slot 3
            }

            // ───────────────────────────────────────────────────────────────────
            // STEP 3: Compute new_max = max(scores, old_max)
            // ───────────────────────────────────────────────────────────────────
            Zmm zmm_new_max = Zmm(19);
            vmaxps(zmm_new_max, zmm_scores, zmm_max);

            // ───────────────────────────────────────────────────────────────────
            // STEP 4: Compute exp inputs
            // ───────────────────────────────────────────────────────────────────
            // corr_input = old_max - new_max (will become exp() for correction)
            // weight_input = score - new_max (will become exp() for weight)
            //
            // CRITICAL: Use INPUT zone registers (zmm8, zmm9) instead of SCRATCH
            // zone (zmm20, zmm21). emit_fast_exp clobbers zmm_scratch(0-2) = zmm20-22
            // during polynomial evaluation. If we use zmm20/21 for corr/weight,
            // the exp computation would overwrite them before finishing!
            Zmm zmm_corr = Zmm(8); // INPUT zone - safe from exp clobber
            vsubps(zmm_corr, zmm_max, zmm_new_max);

            Zmm zmm_weight = Zmm(9); // INPUT zone - safe from exp clobber
            vsubps(zmm_weight, zmm_scores, zmm_new_max);

            // ───────────────────────────────────────────────────────────────────
            // STEP 5: Vectorized exp for corr and weight
            // ───────────────────────────────────────────────────────────────────
            // Uses 16-way parallel exp, only low q_tile_size elements are valid
            // NOTE: emit_fast_exp clobbers zmm20-22 (SCRATCH zone) but not INPUT zone
            exp_emitter_.emit_fast_exp(*this, zmm_corr, zmm_corr);     // corr = exp(old_max - new_max)
            exp_emitter_.emit_fast_exp(*this, zmm_weight, zmm_weight); // weight = exp(score - new_max)

            // ───────────────────────────────────────────────────────────────────
            // STEP 6: Compute new_sum = old_sum * corr + weight
            // ───────────────────────────────────────────────────────────────────
            Zmm zmm_new_sum = Zmm(17);                      // Reuse zmm_sum register
            vfmadd213ps(zmm_new_sum, zmm_corr, zmm_weight); // sum = sum * corr + weight

            // ───────────────────────────────────────────────────────────────────
            // STEP 7: Store updated softmax state back to memory
            // ───────────────────────────────────────────────────────────────────
            // Store [max, sum] pairs using scalar stores for correctness.
            // Memory layout: [max0, sum0, max1, sum1, max2, sum2, max3, sum3]
            // zmm_new_max (zmm19) has [max0, max1, max2, max3, ...] in low elements
            // zmm_new_sum (zmm17) has [sum0, sum1, sum2, sum3, ...] in low elements
            Xmm xmm_new_max = Xmm(19);
            Xmm xmm_new_sum = Xmm(17);

            // Query 0: store max at offset 0, sum at offset 4
            vmovss(ptr[r9 + 0], xmm_new_max);
            vmovss(ptr[r9 + 4], xmm_new_sum);

            if (q_tile_size_ >= 2)
            {
                // Query 1: store max at offset 8, sum at offset 12
                vextractps(ptr[r9 + 8], xmm_new_max, 1);
                vextractps(ptr[r9 + 12], xmm_new_sum, 1);
            }
            if (q_tile_size_ >= 3)
            {
                // Query 2: store max at offset 16, sum at offset 20
                vextractps(ptr[r9 + 16], xmm_new_max, 2);
                vextractps(ptr[r9 + 20], xmm_new_sum, 2);
            }
            if (q_tile_size_ >= 4)
            {
                // Query 3: store max at offset 24, sum at offset 28
                vextractps(ptr[r9 + 24], xmm_new_max, 3);
                vextractps(ptr[r9 + 28], xmm_new_sum, 3);
            }

            // ───────────────────────────────────────────────────────────────────
            // STEP 8: Scatter corr/weight to ZMMs for V accumulation
            // ───────────────────────────────────────────────────────────────────
            // Output layout for V accumulation:
            //   zmm0..zmm(q_tile_size-1): broadcasted corr[q]
            //   zmm(q_tile_size)..zmm(2*q_tile_size-1): broadcasted weight[q]
            for (int q = 0; q < q_tile_size_; ++q)
            {
                // Extract corr[q] and broadcast to zmm(q)
                if (q == 0)
                {
                    vbroadcastss(Zmm(q), Xmm(zmm_corr.getIdx()));
                }
                else
                {
                    vextractps(eax, Xmm(zmm_corr.getIdx()), q);
                    vmovd(Xmm(q), eax);
                    vbroadcastss(Zmm(q), Xmm(q));
                }

                // Extract weight[q] and broadcast to zmm(q + q_tile_size)
                if (q == 0)
                {
                    vbroadcastss(Zmm(q + q_tile_size_), Xmm(zmm_weight.getIdx()));
                }
                else
                {
                    vextractps(eax, Xmm(zmm_weight.getIdx()), q);
                    vmovd(Xmm(q + q_tile_size_), eax);
                    vbroadcastss(Zmm(q + q_tile_size_), Xmm(q + q_tile_size_));
                }
            }

            // ───────────────────────────────────────────────────────────────────
            // STEP 9: Apply causal masking for invalid queries
            // ───────────────────────────────────────────────────────────────────
            // Invalid query conditions:
            //   - q < q_local_start (future position, masked by causal attention)
            //   - q >= tile_size (partial tile, query doesn't exist)
            // For invalid queries:
            //   - corr = 1.0 (preserve existing context unchanged)
            //   - weight = 0.0 (no contribution from this KV position)
            mov(r8, op_tile_size);
            for (int q = 0; q < q_tile_size_; ++q)
            {
                std::string valid_label = ".valid_" + std::to_string(q);
                std::string done_label = ".done_mask_" + std::to_string(q);

                // Check: q_local_start > q means this query is masked (invalid)
                cmp(reg_q_local_start, q);
                jle(valid_label, T_NEAR);

                // Invalid: mask out
                vmovups(Zmm(q), const_one().zmm());
                vxorps(Zmm(q + q_tile_size_), Zmm(q + q_tile_size_), Zmm(q + q_tile_size_));
                jmp(done_label, T_NEAR);

                L(valid_label);
                // Check tile_size <= q (invalid)
                cmp(r8, q);
                jg(done_label, T_NEAR);

                // Invalid: mask out
                vmovups(Zmm(q), const_one().zmm());
                vxorps(Zmm(q + q_tile_size_), Zmm(q + q_tile_size_), Zmm(q + q_tile_size_));

                L(done_label);
            }
        }

        /**
         * @brief Fast scalar exp for prefill (using range reduction)
         */
        void emit_prefill_exp_scalar(const Xbyak::Xmm &dst, const Xbyak::Xmm &src)
        {
            using namespace Xbyak;

            // ───────────────────────────────────────────────────────────────────
            // SCRATCH REGISTER ALLOCATION
            // ───────────────────────────────────────────────────────────────────
            // Avoid zmm20 (xmm_score) and zmm22 (xmm_new_max) which are live
            // when this function is called from emit_prefill_query_attention
            Xmm xmm_n = Xmm(scratch3().zmm().getIdx()); // zmm23 - integer part
            Xmm xmm_f = Xmm(scratch4().zmm().getIdx()); // zmm24 - fractional part
            Xmm xmm_p = Xmm(scratch5().zmm().getIdx()); // zmm25 - polynomial result

            // Load log2(e) constant
            Xmm xmm_log2e = Xmm(const_log2e().zmm().getIdx());

            // ───────────────────────────────────────────────────────────────────
            // RANGE REDUCTION: Convert to base-2
            // ───────────────────────────────────────────────────────────────────
            // t = x * log2(e)
            vmulss(xmm_f, src, xmm_log2e);

            // n = floor(t)
            vrndscaless(xmm_n, xmm_n, xmm_f, 0x01); // Round down (floor)

            // f = t - n (fractional part in [0, 1))
            vsubss(xmm_f, xmm_f, xmm_n);

            // ───────────────────────────────────────────────────────────────────
            // POLYNOMIAL EVALUATION: P(f) ≈ 2^f using Horner's method
            // ───────────────────────────────────────────────────────────────────
            // P(f) = ((((c5*f + c4)*f + c3)*f + c2)*f + c1)*f + c0

            mov(eax, 0x3aaec3b4); // c5 = 0.0013334f
            vmovd(xmm_p, eax);

            mov(eax, 0x3c1d9250);           // c4 = 0.0096181f
            vmovd(dst, eax);                // Reuse dst as temp
            vfmadd213ss(xmm_p, xmm_f, dst); // p = p*f + c4

            mov(eax, 0x3d63570a); // c3 = 0.0555041f
            vmovd(dst, eax);
            vfmadd213ss(xmm_p, xmm_f, dst); // p = p*f + c3

            mov(eax, 0x3e75fdf0); // c2 = 0.2402265f
            vmovd(dst, eax);
            vfmadd213ss(xmm_p, xmm_f, dst); // p = p*f + c2

            mov(eax, 0x3f317218); // c1 = 0.6931472f (≈ ln(2))
            vmovd(dst, eax);
            vfmadd213ss(xmm_p, xmm_f, dst); // p = p*f + c1

            // p = p*f + 1.0 (c0)
            Xmm xmm_one = Xmm(const_one().zmm().getIdx());
            vfmadd213ss(xmm_p, xmm_f, xmm_one);

            // ───────────────────────────────────────────────────────────────────
            // FINAL SCALING: result = P(f) * 2^n
            // ───────────────────────────────────────────────────────────────────
            vscalefss(dst, xmm_p, xmm_n);
        }

        /**
         * @brief Accumulate weighted V values into context buffer
         *
         * For a single query, accumulates the contribution of the current V
         * position into the context buffer. Uses the online softmax correction
         * factor to rescale the existing context.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * ALGORITHM
         * ═══════════════════════════════════════════════════════════════════════
         *
         * For each block b:
         *   V_dequant = V_data[b] * V_scale[b]        // Dequantize V
         *   weighted_V = V_dequant * weight           // Apply softmax weight
         *   context[b] = context[b] * corr + weighted_V  // Update with correction
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER USAGE
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Input:
         *   reg_V: V cache pointer for this KV position
         *   reg_ctx: Context buffer pointer for this query
         *   zmm_weight(): Attention weight (broadcasted)
         *   zmm_corr(): Correction factor (broadcasted)
         *
         * SCRATCH zone (clobbered):
         *   zmm_scratch(0): V scale (broadcasted)
         *   zmm_scratch(1): Combined scale (V_scale * weight)
         *   zmm_scratch(2): V data low (FP32)
         *   zmm_scratch(3): V data high (FP32)
         *   zmm_scratch(4): Context low
         *   zmm_scratch(5): Context high
         *
         * @param reg_V V cache pointer for this KV position
         * @param reg_ctx Context buffer pointer (on stack)
         * @param num_blocks Number of Q8_1 blocks (head_dim / 32)
         */
        void emit_prefill_v_accum(
            const Xbyak::Reg64 &reg_V,
            const Xbyak::Reg64 &reg_ctx,
            int num_blocks)
        {
            using namespace Xbyak;

            for (int b = 0; b < num_blocks; ++b)
            {
                int v_off = b * 36;       // V block offset (36 bytes per Q8_1 block)
                int ctx_off = b * 32 * 4; // Context offset (32 floats × 4 bytes)

                // Load V scale (FP16 → FP32, broadcasted to ZMM)
                Zmm zmm_v_scale = scratch0().zmm();
                vpbroadcastw(Xmm(zmm_v_scale.getIdx()), ptr[reg_V + v_off]);
                vcvtph2ps(Xmm(zmm_v_scale.getIdx()), Xmm(zmm_v_scale.getIdx()));
                vbroadcastss(zmm_v_scale, Xmm(zmm_v_scale.getIdx()));

                // Compute combined = V_scale * weight
                Zmm zmm_comb = scratch1().zmm();
                vmulps(zmm_comb, zmm_v_scale, state_weight().zmm());

                // Load V data (32 × int8 → 2 × 16 FP32)
                Zmm zmm_v_lo = scratch2().zmm();
                Zmm zmm_v_hi = scratch3().zmm();
                vpmovsxbd(zmm_v_lo, ptr[reg_V + v_off + 4]);      // Low 16 bytes
                vpmovsxbd(zmm_v_hi, ptr[reg_V + v_off + 4 + 16]); // High 16 bytes
                vcvtdq2ps(zmm_v_lo, zmm_v_lo);
                vcvtdq2ps(zmm_v_hi, zmm_v_hi);

                // Apply combined scale: weighted_V = V_data * combined
                vmulps(zmm_v_lo, zmm_v_lo, zmm_comb);
                vmulps(zmm_v_hi, zmm_v_hi, zmm_comb);

                // Load current context
                Zmm zmm_ctx_lo = scratch4().zmm();
                Zmm zmm_ctx_hi = scratch5().zmm();
                vmovups(zmm_ctx_lo, ptr[reg_ctx + ctx_off]);
                vmovups(zmm_ctx_hi, ptr[reg_ctx + ctx_off + 64]);

                // Update context: ctx = ctx * corr + weighted_V
                vmulps(zmm_ctx_lo, zmm_ctx_lo, state_corr().zmm());
                vmulps(zmm_ctx_hi, zmm_ctx_hi, state_corr().zmm());
                vaddps(zmm_ctx_lo, zmm_ctx_lo, zmm_v_lo);
                vaddps(zmm_ctx_hi, zmm_ctx_hi, zmm_v_hi);

                // Store updated context
                vmovups(ptr[reg_ctx + ctx_off], zmm_ctx_lo);
                vmovups(ptr[reg_ctx + ctx_off + 64], zmm_ctx_hi);
            }
        }

        /**
         * @brief Normalize context and apply Wo projection for prefill tile
         *
         * After all KV positions have been processed, this method:
         * 1. Normalizes the context by dividing by the softmax sum
         * 2. Projects through Wo to produce the final output
         *
         * ═══════════════════════════════════════════════════════════════════════
         * ALGORITHM
         * ═══════════════════════════════════════════════════════════════════════
         *
         * For each query q in tile:
         *   For each head h:
         *     inv_sum = 1.0 / (softmax_sum[q,h] + epsilon)
         *     context[q,h] *= inv_sum                // Normalize
         *   output[q] = context[q] @ Wo^T            // Project all heads
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER USAGE
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Input:
         *   reg_Wo: Wo weight pointer
         *   reg_output: Output buffer pointer
         *
         * Loop variables:
         *   r8: q_local (query index within tile)
         *   r10: context_ptr_offset
         *   r11: softmax_ptr_offset
         *
         * SCRATCH zone:
         *   zmm_scratch(0): inv_sum
         *   zmm_scratch(1): 1.0 constant
         *   zmm_scratch(2): epsilon (1e-6)
         *
         * @param reg_Wo Wo weight matrix pointer
         * @param reg_output Output buffer pointer
         * @param num_blocks head_dim / 32
         * @param tile_size Maximum queries in tile
         * @param softmax_offset Stack offset to softmax state
         * @param context_offset Stack offset to context buffer
         * @param tile_start_spill Stack offset for tile_start variable
         * @param tile_size_spill Stack offset for tile_size variable
         * @param d_model num_heads × head_dim
         * @param q_local_spill Stack offset for q_local spill
         * @param q_loop_spill Stack offset for q_loop spill
         */
        void emit_prefill_normalize_project(
            const Xbyak::Reg64 &reg_Wo,
            const Xbyak::Reg64 &reg_output,
            int num_blocks,
            int tile_size,
            int softmax_offset,
            int context_offset,
            int tile_start_spill,
            int tile_size_spill,
            int d_model,
            int q_local_spill,
            int q_loop_spill)
        {
            using namespace Xbyak;

            int head_dim = num_blocks * 32;

            inLocalLabel();

            // ───────────────────────────────────────────────────────────────────
            // QUERY LOOP: Process each query in tile
            // ───────────────────────────────────────────────────────────────────
            xor_(r8, r8); // q_local = 0

            L(".proj_loop");
            cmp(r8, ptr[rsp + tile_size_spill]);
            jge(".proj_end", T_NEAR);

            // Spill q_local (will be clobbered by emit_prefill_wo_projection)
            mov(ptr[rsp + q_local_spill], r8);

            // Calculate context pointer offset for this query
            // context_ptr = rsp + context_offset + q_local * d_model * 4
            mov(r10, r8);
            imul(r10, r10, d_model * 4);
            add(r10, context_offset);

            // Calculate softmax state offset for this query
            // softmax_ptr = rsp + softmax_offset + q_local * 8
            mov(r11, r8);
            shl(r11, 3); // * 8 (sizeof max+sum pair)
            add(r11, softmax_offset);

            // ───────────────────────────────────────────────────────────────────
            // NORMALIZE: Divide context by softmax sum for all heads
            // ───────────────────────────────────────────────────────────────────
            Zmm zmm_inv_sum = scratch0().zmm();
            Zmm zmm_one = scratch1().zmm();
            load_constant_f32(zmm_one, 1.0f);
            Zmm zmm_epsilon = scratch2().zmm();
            load_constant_f32(zmm_epsilon, 1e-20f); // Small epsilon for numerical stability

            // ───────────────────────────────────────────────────────────────────
            // HEAD LOOP: Normalize context for each head
            // ───────────────────────────────────────────────────────────────────
            xor_(r9, r9); // head_idx = 0

            L(".head_loop");
            cmp(r9, config_.num_heads);
            jge(".head_end", T_NEAR);

            // Load 1/sum for this head: inv_sum = 1.0 / (sum + epsilon)
            // Sum is at offset 4 within each (max, sum) pair
            vbroadcastss(zmm_inv_sum, ptr[rsp + r11 + 4]);
            vmaxps(zmm_inv_sum, zmm_inv_sum, zmm_epsilon); // Clamp to avoid div by 0
            vdivps(zmm_inv_sum, zmm_one, zmm_inv_sum);

            // Normalize all blocks in this head's context
            for (int b = 0; b < num_blocks; ++b)
            {
                int b_off = b * 32 * 4; // 32 floats × 4 bytes

                Zmm zmm_lo_safe = scratch4().zmm();
                Zmm zmm_hi_safe = scratch3().zmm();

                // Load context[head, block]
                vmovups(zmm_lo_safe, ptr[rsp + r10 + b_off]);
                vmovups(zmm_hi_safe, ptr[rsp + r10 + b_off + 64]);

                // Normalize: context *= inv_sum
                vmulps(zmm_lo_safe, zmm_lo_safe, zmm_inv_sum);
                vmulps(zmm_hi_safe, zmm_hi_safe, zmm_inv_sum);

                // Store normalized context
                vmovups(ptr[rsp + r10 + b_off], zmm_lo_safe);
                vmovups(ptr[rsp + r10 + b_off + 64], zmm_hi_safe);
            }

            // Advance to next head
            add(r11, tile_size * 8); // Next head's softmax (tile_size pairs × 8 bytes)
            add(r10, head_dim * 4);  // Next head's context (head_dim floats × 4 bytes)

            inc(r9);
            jmp(".head_loop", T_NEAR);

            L(".head_end");

            // Restore q_local and continue query loop
            mov(r8, ptr[rsp + q_local_spill]);
            inc(r8);
            jmp(".proj_loop", T_NEAR);

            L(".proj_end");

            // ───────────────────────────────────────────────────────────────────
            // WO PROJECTION: Project normalized context through Wo
            // ───────────────────────────────────────────────────────────────────
            emit_prefill_wo_projection_tile(
                reg_Wo, reg_output, tile_size, context_offset,
                tile_start_spill, tile_size_spill, d_model, q_loop_spill);

            outLocalLabel();
        }

        /**
         * @brief Dispatch Wo projection for entire prefill tile
         *
         * Selects the appropriate Wo projection implementation based on the
         * configured weight format (FP32, FP16, BF16, Q8_1). Q8_1 has a
         * specialized tile implementation for efficiency.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * FORMAT DISPATCH
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Q8_1: Use emit_prefill_wo_q8_1_tile (optimized tile implementation)
         * Other: Fall back to sequential emit_prefill_wo_projection calls
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER USAGE
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Input:
         *   reg_Wo: Wo weight matrix pointer
         *   reg_output: Output buffer pointer
         *
         * Loop variables (clobbered):
         *   r8: Row counter / q_loop
         *   r10: context_ptr_offset
         *   r11: q_global (tile_start + q_local)
         *
         * @param reg_Wo Wo weight pointer
         * @param reg_output Output buffer pointer
         * @param tile_size Maximum queries in tile
         * @param context_offset Stack offset to context buffer
         * @param tile_start_spill Stack offset for tile_start
         * @param tile_size_spill Stack offset for tile_size
         * @param d_model num_heads × head_dim
         * @param q_loop_spill Stack offset for loop counter spill
         */
        void emit_prefill_wo_projection_tile(
            const Xbyak::Reg64 &reg_Wo,
            const Xbyak::Reg64 &reg_output,
            int tile_size,
            int context_offset,
            int tile_start_spill,
            int tile_size_spill,
            int d_model,
            int q_loop_spill)
        {
            using namespace Xbyak;
            inLocalLabel();

            // ───────────────────────────────────────────────────────────────────
            // FORMAT DISPATCH
            // ───────────────────────────────────────────────────────────────────
            switch (config_.wo_format)
            {
            case WoFormat::Q8_1:
                // Optimized Q8_1 tile implementation
                emit_prefill_wo_q8_1_tile(reg_Wo, reg_output, tile_size, context_offset, tile_start_spill, tile_size_spill, d_model);
                break;
            case WoFormat::Q8_1_VNNI_PACKED:
                // Packed VNNI GEMM path: one GEMM for the whole tile
                {
                    // m = actual tile size (dynamic)
                    mov(r8, ptr[rsp + tile_size_spill]);

                    // A = context buffer base
                    lea(r9, ptr[rsp + context_offset]);

                    // C = output + tile_start * d_model * 4
                    mov(r10, ptr[rsp + tile_start_spill]);
                    imul(r10, r10, d_model * 4);
                    add(r10, reg_output);

                    // SysV args: rdi=WoPacked, rsi=A, rdx=C, rcx=m, r8=n, r9=k
                    mov(rdi, reg_Wo);
                    mov(rsi, r9);
                    mov(rdx, r10);
                    mov(rcx, r8);
                    mov(r8d, d_model);
                    mov(r9d, d_model);

                    mov(rax, (size_t)llaminar2_wo_q8_1_vnni_packed_gemm);
                    call(rax);
                }
                break;
            default:
                // ───────────────────────────────────────────────────────────────
                // FALLBACK: Sequential projection for other formats
                // ───────────────────────────────────────────────────────────────
                // Loop over queries and call single-query projection
                // Uses q_loop_spill to avoid register conflicts with emit_prefill_wo_projection
                {
                    mov(qword[rsp + q_loop_spill], 0); // Initialize loop counter

                    L(".fallback_loop");
                    mov(r8, ptr[rsp + q_loop_spill]);
                    cmp(r8, ptr[rsp + tile_size_spill]);
                    jge(".fallback_end", T_NEAR);

                    // Calculate q_global = tile_start + q_local
                    mov(r11, ptr[rsp + tile_start_spill]);
                    add(r11, r8);

                    // Calculate context pointer offset
                    mov(r10, r8);
                    imul(r10, r10, d_model * 4);
                    add(r10, context_offset);

                    // Call single-query projection
                    emit_prefill_wo_projection(reg_Wo, reg_output, r11, r10, d_model);

                    // Increment loop counter
                    mov(r8, ptr[rsp + q_loop_spill]);
                    inc(r8);
                    mov(ptr[rsp + q_loop_spill], r8);
                    jmp(".fallback_loop", T_NEAR);
                    L(".fallback_end");
                }
                break;
            }
            outLocalLabel();
        }

        /**
         * @brief Optimized Q8_1 Wo projection for entire prefill tile
         *
         * This is a highly-optimized implementation that processes multiple
         * output rows and queries simultaneously using aggressive loop unrolling
         * and register blocking.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * ALGORITHM OVERVIEW
         * ═══════════════════════════════════════════════════════════════════════
         *
         * For each output row i ∈ [0, d_model):
         *   For each query q ∈ [0, tile_size):
         *     output[tile_start + q, i] = Σ_k context[q, k] * Wo[i, k]
         *
         * Optimization strategy:
         * - Unroll output rows by 2 (process i and i+1 simultaneously)
         * - Unroll queries by tile_size (up to 8 queries in parallel)
         * - Use ZMM accumulators for each (query, row) pair
         * - Load Wo weights for 2 rows, reuse across all queries
         * - Horizontal sum at end of k-loop for final reduction
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER ALLOCATION
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Accumulators (persistent across k-loop):
         *   zmm0-7:  acc[q=0..7] for row i   (OUTPUT[q][i])
         *   zmm8-15: acc[q=0..7] for row i+1 (OUTPUT[q][i+1])
         *
         * Weight registers (reload each k block):
         *   zmm16: d0 = scale for Wo[i]
         *   zmm17: w0_lo = Wo[i, k*32 .. k*32+15]  (dequantized)
         *   zmm18: w0_hi = Wo[i, k*32+16 .. k*32+31]
         *   zmm19: d1 = scale for Wo[i+1]
         *   zmm20: w1_lo = Wo[i+1, k*32 .. k*32+15]
         *   zmm21: w1_hi = Wo[i+1, k*32+16 .. k*32+31]
         *
         * Context registers (reload for each query):
         *   zmm22: c_lo = context[q, k*32 .. k*32+15]
         *   zmm23: c_hi = context[q, k*32+16 .. k*32+31]
         *
         * Loop counters (GPR):
         *   r8:  i (output row counter)
         *   rcx: k_blk (block counter within row)
         *   r12: Wo pointer for row i
         *   r13: Wo pointer for row i+1
         *
         * ═══════════════════════════════════════════════════════════════════════
         * Q8_1 WEIGHT DEQUANTIZATION
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Wo is stored in Q8_1 format (36 bytes per block):
         *   - Bytes [0-1]: FP16 scale (d)
         *   - Bytes [2-3]: INT16 sum_qs (unused here, needed for quantized input)
         *   - Bytes [4-35]: 32 x INT8 quantized weights (qs)
         *
         * Dequantization: w_fp32 = int32(qs) * float(d)
         *
         * We skip sum_qs because context is already in FP32, so no correction
         * term is needed. The standard Q8_1 formula with quantized input would
         * need: output += sum_qs * input_scale to correct for zero-centering.
         *
         * @param reg_Wo Pointer to Wo weight matrix (Q8_1 format)
         * @param reg_output Pointer to output buffer
         * @param tile_size Maximum queries in tile (compile-time constant)
         * @param context_offset Stack offset to context buffer
         * @param tile_start_spill Stack offset for tile_start value
         * @param tile_size_spill Stack offset for actual tile_size value
         * @param d_model Output dimension (num_heads × head_dim)
         */
        void emit_prefill_wo_q8_1_tile(
            const Xbyak::Reg64 &reg_Wo,
            const Xbyak::Reg64 &reg_output,
            int tile_size,
            int context_offset,
            int tile_start_spill,
            int tile_size_spill,
            int d_model)
        {
            using namespace Xbyak;
            inLocalLabel();

            int num_blocks = d_model / 32;

            // ───────────────────────────────────────────────────────────────────
            // UNROLLING STRATEGY
            // ───────────────────────────────────────────────────────────────────
            // We unroll i (output rows) by 2
            // We unroll q (tile queries) by tile_size (up to 8)
            // Registers:
            // acc[q] for i: zmm0..zmm7
            // acc[q] for i+1: zmm8..zmm15
            // W[i]: zmm16, zmm17, zmm18 (scale, lo, hi)
            // W[i+1]: zmm19, zmm20, zmm21
            // C[q]: zmm22, zmm23 (lo, hi)

            // ───────────────────────────────────────────────────────────────────
            // OUTER LOOP: Output rows (i)
            // ───────────────────────────────────────────────────────────────────
            xor_(r8, r8); // i = 0

            L(".row_loop");
            cmp(r8, d_model);
            jge(".row_end", T_NEAR);

            // Check if we can process 2 rows (i and i+1)
            mov(r9, d_model);
            sub(r9, r8);
            cmp(r9, 2);
            jl(".single_row", T_NEAR);

            // ═══════════════════════════════════════════════════════════════════
            // DOUBLE ROW PROCESSING (i and i+1)
            // ═══════════════════════════════════════════════════════════════════

            // ───────────────────────────────────────────────────────────────────
            // STEP 1: Clear accumulators for all queries
            // ───────────────────────────────────────────────────────────────────
            for (int q = 0; q < tile_size; ++q)
            {
                vxorps(Zmm(q), Zmm(q), Zmm(q));             // acc[q] for row i
                vxorps(Zmm(q + 8), Zmm(q + 8), Zmm(q + 8)); // acc[q] for row i+1
            }

            // ───────────────────────────────────────────────────────────────────
            // STEP 2: Compute Wo row pointers
            // ───────────────────────────────────────────────────────────────────
            // r12 = Wo + i * row_stride (row_stride = num_blocks * 36)
            // r13 = Wo + (i+1) * row_stride
            mov(r12, r8);
            imul(r12, r12, num_blocks * 36);
            add(r12, reg_Wo);

            mov(r13, r12);
            add(r13, num_blocks * 36); // Next row

            // ───────────────────────────────────────────────────────────────────
            // STEP 3: Inner loop over k blocks
            // ───────────────────────────────────────────────────────────────────
            xor_(rcx, rcx); // k_blk = 0

            L(".blk_loop_2");
            cmp(rcx, num_blocks);
            jge(".blk_end_2", T_NEAR);

            // Compute block offset = k_blk * 36
            mov(rax, rcx);
            imul(rax, rax, 36);

            // Load and dequantize Wo[i] block
            {
                Zmm d0 = Zmm(16);    // Scale for row i
                Zmm w0_lo = Zmm(17); // Weights [0..15] for row i
                Zmm w0_hi = Zmm(18); // Weights [16..31] for row i

                // Load FP16 scale → broadcast to all 16 lanes
                vpbroadcastw(Xmm(d0.getIdx()), ptr[r12 + rax]); // Load scale (FP16)
                vcvtph2ps(Xmm(d0.getIdx()), Xmm(d0.getIdx()));  // FP16 → FP32
                vbroadcastss(d0, Xmm(d0.getIdx()));             // Broadcast to all lanes

                // Load INT8 weights → sign-extend to INT32 → convert to FP32 → scale
                vpmovsxbd(w0_lo, ptr[r12 + rax + 4]);      // qs[0..15] INT8 → INT32
                vpmovsxbd(w0_hi, ptr[r12 + rax + 4 + 16]); // qs[16..31] INT8 → INT32
                vcvtdq2ps(w0_lo, w0_lo);                   // INT32 → FP32
                vcvtdq2ps(w0_hi, w0_hi);
                vmulps(w0_lo, w0_lo, d0); // Dequantize: qs * scale
                vmulps(w0_hi, w0_hi, d0);

                // Load and dequantize Wo[i+1] block
                Zmm d1 = Zmm(19);    // Scale for row i+1
                Zmm w1_lo = Zmm(20); // Weights [0..15] for row i+1
                Zmm w1_hi = Zmm(21); // Weights [16..31] for row i+1

                vpbroadcastw(Xmm(d1.getIdx()), ptr[r13 + rax]);
                vcvtph2ps(Xmm(d1.getIdx()), Xmm(d1.getIdx()));
                vbroadcastss(d1, Xmm(d1.getIdx()));

                vpmovsxbd(w1_lo, ptr[r13 + rax + 4]);
                vpmovsxbd(w1_hi, ptr[r13 + rax + 4 + 16]);
                vcvtdq2ps(w1_lo, w1_lo);
                vcvtdq2ps(w1_hi, w1_hi);
                vmulps(w1_lo, w1_lo, d1);
                vmulps(w1_hi, w1_hi, d1);

                // ───────────────────────────────────────────────────────────────
                // STEP 4: Accumulate context · weight for all queries
                // ───────────────────────────────────────────────────────────────
                // Context base pointer: rsp + context_offset + k_blk * 128
                // (128 bytes = 32 floats × 4 bytes)
                mov(rdx, rcx);
                shl(rdx, 7); // k_blk * 128
                add(rdx, context_offset);
                add(rdx, rsp);

                for (int q = 0; q < tile_size; ++q)
                {
                    Zmm c_lo = Zmm(22); // Context[q, k*32 .. k*32+15]
                    Zmm c_hi = Zmm(23); // Context[q, k*32+16 .. k*32+31]

                    // Context offset = q * d_model * sizeof(float)
                    mov(r10, q);
                    imul(r10, r10, d_model * 4);

                    // Load context block for query q
                    vmovups(c_lo, ptr[rdx + r10]);
                    vmovups(c_hi, ptr[rdx + r10 + 64]);

                    // Accumulate: acc[q, i] += context[q, k] * Wo[i, k]
                    vfmadd231ps(Zmm(q), c_lo, w0_lo);
                    vfmadd231ps(Zmm(q), c_hi, w0_hi);

                    // Accumulate: acc[q, i+1] += context[q, k] * Wo[i+1, k]
                    vfmadd231ps(Zmm(q + 8), c_lo, w1_lo);
                    vfmadd231ps(Zmm(q + 8), c_hi, w1_hi);
                }
            }

            inc(rcx);
            jmp(".blk_loop_2", T_NEAR);
            L(".blk_end_2");

            // ───────────────────────────────────────────────────────────────────
            // STEP 5: Horizontal sum and store results
            // ───────────────────────────────────────────────────────────────────
            // Load actual tile size for bounds checking
            mov(rax, ptr[rsp + tile_size_spill]);

            // Output ptr base computation: output[(tile_start + q) * d_model + i]
            mov(r11, ptr[rsp + tile_start_spill]);

            for (int q = 0; q < tile_size; ++q)
            {
                // Skip if q >= actual_tile_size (dynamic bounds check)
                cmp(rax, q);
                jle(".skip_store_2_" + std::to_string(q), T_NEAR);

                // Reduce ZMM accumulator to scalar: acc[q] for row i
                emit_horizontal_sum_to_scalar(Zmm(q));

                // Reduce ZMM accumulator to scalar: acc[q] for row i+1
                emit_horizontal_sum_to_scalar(Zmm(q + 8));

                // Compute output pointer: output + (tile_start + q) * d_model * 4 + i * 4
                mov(r10, ptr[rsp + tile_start_spill]);
                add(r10, q);
                imul(r10, r10, d_model * 4);
                add(r10, reg_output);

                // Store: output[tile_start + q, i] and output[tile_start + q, i+1]
                vmovss(ptr[r10 + r8 * 4], Xmm(Zmm(q).getIdx()));
                vmovss(ptr[r10 + r8 * 4 + 4], Xmm(Zmm(q + 8).getIdx()));

                L(".skip_store_2_" + std::to_string(q));
            }

            add(r8, 2); // Advance by 2 rows
            jmp(".row_loop", T_NEAR);

            // ═══════════════════════════════════════════════════════════════════
            // SINGLE ROW PROCESSING (fallback for d_model % 2 == 1)
            // ═══════════════════════════════════════════════════════════════════
            L(".single_row");

            // ───────────────────────────────────────────────────────────────────
            // STEP 1: Clear accumulators for all queries (single row only)
            // ───────────────────────────────────────────────────────────────────
            for (int q = 0; q < tile_size; ++q)
            {
                vxorps(Zmm(q), Zmm(q), Zmm(q));
            }

            // ───────────────────────────────────────────────────────────────────
            // STEP 2: Compute Wo row pointer for row i
            // ───────────────────────────────────────────────────────────────────

            mov(r12, r8);
            imul(r12, r12, num_blocks * 36);
            add(r12, reg_Wo);

            xor_(rcx, rcx);
            L(".blk_loop_1");
            cmp(rcx, num_blocks);
            jge(".blk_end_1", T_NEAR);

            // Compute block offset = k_blk * 36
            mov(rax, rcx);
            imul(rax, rax, 36);

            // Load and dequantize Wo[i] block (same as double-row version)
            {
                Zmm d0 = Zmm(16);    // Scale
                Zmm w0_lo = Zmm(17); // Weights [0..15]
                Zmm w0_hi = Zmm(18); // Weights [16..31]

                // Load FP16 scale → broadcast
                vpbroadcastw(Xmm(d0.getIdx()), ptr[r12 + rax]);
                vcvtph2ps(Xmm(d0.getIdx()), Xmm(d0.getIdx()));
                vbroadcastss(d0, Xmm(d0.getIdx()));

                // Load and dequantize INT8 weights
                vpmovsxbd(w0_lo, ptr[r12 + rax + 4]);
                vpmovsxbd(w0_hi, ptr[r12 + rax + 4 + 16]);
                vcvtdq2ps(w0_lo, w0_lo);
                vcvtdq2ps(w0_hi, w0_hi);
                vmulps(w0_lo, w0_lo, d0); // Dequantize
                vmulps(w0_hi, w0_hi, d0);

                // Context base pointer
                mov(rdx, rcx);
                shl(rdx, 7); // k_blk * 128
                add(rdx, context_offset);
                add(rdx, rsp);

                // Accumulate for all queries
                for (int q = 0; q < tile_size; ++q)
                {
                    Zmm c_lo = Zmm(22);
                    Zmm c_hi = Zmm(23);

                    mov(r10, q);
                    imul(r10, r10, d_model * 4);

                    vmovups(c_lo, ptr[rdx + r10]);
                    vmovups(c_hi, ptr[rdx + r10 + 64]);

                    vfmadd231ps(Zmm(q), c_lo, w0_lo);
                    vfmadd231ps(Zmm(q), c_hi, w0_hi);
                }
            }

            inc(rcx);
            jmp(".blk_loop_1", T_NEAR);
            L(".blk_end_1");

            // ───────────────────────────────────────────────────────────────────
            // STEP 5: Horizontal sum and store (single row)
            // ───────────────────────────────────────────────────────────────────
            mov(rax, ptr[rsp + tile_size_spill]);
            mov(r11, ptr[rsp + tile_start_spill]);

            for (int q = 0; q < tile_size; ++q)
            {
                cmp(rax, q);
                jle(".skip_store_1_" + std::to_string(q), T_NEAR);

                emit_horizontal_sum_to_scalar(Zmm(q));

                mov(r10, ptr[rsp + tile_start_spill]);
                add(r10, q);
                imul(r10, r10, d_model * 4);
                add(r10, reg_output);

                vmovss(ptr[r10 + r8 * 4], Xmm(Zmm(q).getIdx()));

                L(".skip_store_1_" + std::to_string(q));
            }

            inc(r8);
            jmp(".row_loop", T_NEAR);

            L(".row_end");
            outLocalLabel();
        }

        /**
         * @brief Single-query Wo projection for prefill mode
         *
         * Dispatches to format-specific implementation based on config_.wo_format.
         * Used as fallback for non-Q8_1 formats in the tile projection loop.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * ALGORITHM
         * ═══════════════════════════════════════════════════════════════════════
         *
         * For each output row i ∈ [0, d_model):
         *   output[q_global, i] = Σ_j context[j] * Wo[i, j]
         *
         * @param reg_Wo Pointer to Wo weight matrix
         * @param reg_output Pointer to output buffer
         * @param reg_q_global Global query index (tile_start + q_local)
         * @param reg_ctx_offset Stack offset to context buffer
         * @param d_model Output dimension
         */
        void emit_prefill_wo_projection(
            const Xbyak::Reg64 &reg_Wo,
            const Xbyak::Reg64 &reg_output,
            const Xbyak::Reg64 &reg_q_global,
            const Xbyak::Reg64 &reg_ctx_offset,
            int d_model)
        {
            using namespace Xbyak;

            // Calculate output pointer: output + q_global * d_model * sizeof(float)
            Reg64 reg_out_ptr = r11;
            mov(reg_out_ptr, reg_q_global);
            imul(reg_out_ptr, reg_out_ptr, d_model * 4);
            add(reg_out_ptr, reg_output);

            // Dispatch by weight format
            switch (config_.wo_format)
            {
            case WoFormat::FP32:
                emit_prefill_wo_fp32(reg_Wo, reg_out_ptr, reg_ctx_offset, d_model);
                break;
            case WoFormat::FP16:
                emit_prefill_wo_fp16(reg_Wo, reg_out_ptr, reg_ctx_offset, d_model);
                break;
            case WoFormat::BF16:
                emit_prefill_wo_bf16(reg_Wo, reg_out_ptr, reg_ctx_offset, d_model);
                break;
            case WoFormat::Q8_1:
                emit_prefill_wo_q8_1(reg_Wo, reg_out_ptr, reg_ctx_offset, d_model);
                break;
            }
        }

        /**
         * @brief FP32 Wo projection for single query
         *
         * Simple row-major matrix-vector multiply with FP32 weights.
         * Loops over output rows, vectorizes over input dimension.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER USAGE
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Accumulators:
         *   zmm_scratch(0): acc = running dot product sum
         *
         * Operands:
         *   zmm_scratch(1): ctx = context[col..col+15]
         *   zmm_scratch(2): wo = Wo[row, col..col+15]
         *
         * Loop counters:
         *   r8: row index
         *   r9: Wo row pointer
         *   rcx: column index
         */
        void emit_prefill_wo_fp32(const Xbyak::Reg64 &reg_Wo, const Xbyak::Reg64 &reg_out,
                                  const Xbyak::Reg64 &reg_ctx_off, int d_model)
        {
            using namespace Xbyak;
            inLocalLabel();

            // Outer loop: rows
            xor_(r8, r8);
            L(".row");
            cmp(r8, d_model);
            jge(".end", T_NEAR);

            // Clear accumulator
            Zmm acc = scratch0().zmm();
            vxorps(acc, acc, acc);

            // Calculate Wo row pointer: Wo + row * d_model * sizeof(float)
            mov(r9, r8);
            imul(r9, r9, d_model * 4);
            add(r9, reg_Wo);

            // Inner loop: columns (vectorized by 16)
            xor_(rcx, rcx);
            L(".col");
            cmp(rcx, d_model);
            jge(".col_end", T_NEAR);

            Zmm ctx = scratch1().zmm();
            Zmm wo = scratch2().zmm();

            // Load context from stack
            lea(rax, ptr[rsp + reg_ctx_off]);
            vmovups(ctx, ptr[rax + rcx * 4]);

            // Load Wo weights
            vmovups(wo, ptr[r9 + rcx * 4]);

            // Accumulate: acc += ctx * wo
            vfmadd231ps(acc, ctx, wo);

            add(rcx, 16);
            jmp(".col", T_NEAR);
            L(".col_end");

            // Horizontal sum and store
            emit_horizontal_sum_to_scalar(acc);
            vmovss(ptr[reg_out + r8 * 4], Xmm(acc.getIdx()));

            inc(r8);
            jmp(".row", T_NEAR);
            L(".end");
            outLocalLabel();
        }

        /**
         * @brief FP16 Wo projection for single query
         *
         * Matrix-vector multiply with FP16 weights (converted to FP32 on-the-fly).
         *
         * ═══════════════════════════════════════════════════════════════════════
         * CONVERSION
         * ═══════════════════════════════════════════════════════════════════════
         *
         * FP16 → FP32 conversion: vcvtph2ps (AVX-512 F16C)
         * Row stride: d_model × 2 bytes (FP16)
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER USAGE
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Same as emit_prefill_wo_fp32 except:
         * - vmovups loads 16 FP16 values (32 bytes) into YMM
         * - vcvtph2ps converts YMM(FP16) → ZMM(FP32)
         */
        void emit_prefill_wo_fp16(const Xbyak::Reg64 &reg_Wo, const Xbyak::Reg64 &reg_out,
                                  const Xbyak::Reg64 &reg_ctx_off, int d_model)
        {
            using namespace Xbyak;
            inLocalLabel();

            xor_(r8, r8);
            L(".row");
            cmp(r8, d_model);
            jge(".end", T_NEAR);

            Zmm acc = scratch0().zmm();
            vxorps(acc, acc, acc);

            // Calculate Wo row pointer: Wo + row * d_model * sizeof(FP16)
            mov(r9, r8);
            imul(r9, r9, d_model * 2); // 2 bytes per FP16
            add(r9, reg_Wo);

            xor_(rcx, rcx);
            L(".col");
            cmp(rcx, d_model);
            jge(".col_end", T_NEAR);

            Zmm ctx = scratch1().zmm();
            Zmm wo = scratch2().zmm();

            // Load context (FP32)
            lea(rax, ptr[rsp + reg_ctx_off]);
            vmovups(ctx, ptr[rax + rcx * 4]);

            // Load FP16 weights and convert to FP32
            vmovups(Ymm(wo.getIdx()), ptr[r9 + rcx * 2]); // Load 16 × FP16
            vcvtph2ps(wo, Ymm(wo.getIdx()));              // Convert to 16 × FP32

            vfmadd231ps(acc, ctx, wo);

            add(rcx, 16);
            jmp(".col", T_NEAR);
            L(".col_end");

            emit_horizontal_sum_to_scalar(acc);
            vmovss(ptr[reg_out + r8 * 4], Xmm(acc.getIdx()));

            inc(r8);
            jmp(".row", T_NEAR);
            L(".end");
            outLocalLabel();
        }

        /**
         * @brief BF16 Wo projection for single query
         *
         * Matrix-vector multiply with BF16 weights (converted to FP32 on-the-fly).
         *
         * ═══════════════════════════════════════════════════════════════════════
         * CONVERSION
         * ═══════════════════════════════════════════════════════════════════════
         *
         * BF16 → FP32 conversion is done by:
         *   1. vpmovzxwd: Zero-extend 16-bit to 32-bit (BF16 → low 16 bits of dword)
         *   2. vpslld: Shift left by 16 (move to high 16 bits = FP32 representation)
         *
         * BF16 format: [S|EEEEEEEE|MMMMMMM] (sign, 8-bit exp, 7-bit mantissa)
         * FP32 format: [S|EEEEEEEE|MMMMMMMMMMMMMMMMMMMMMMM] (same exp, padded mantissa)
         *
         * Row stride: d_model × 2 bytes (BF16)
         */
        void emit_prefill_wo_bf16(const Xbyak::Reg64 &reg_Wo, const Xbyak::Reg64 &reg_out,
                                  const Xbyak::Reg64 &reg_ctx_off, int d_model)
        {
            using namespace Xbyak;
            inLocalLabel();

            xor_(r8, r8);
            L(".row");
            cmp(r8, d_model);
            jge(".end", T_NEAR);

            Zmm acc = scratch0().zmm();
            vxorps(acc, acc, acc);

            // Calculate Wo row pointer
            mov(r9, r8);
            imul(r9, r9, d_model * 2);
            add(r9, reg_Wo);

            xor_(rcx, rcx);
            L(".col");
            cmp(rcx, d_model);
            jge(".col_end", T_NEAR);

            Zmm ctx = scratch1().zmm();
            Zmm wo = scratch2().zmm();

            // Load context (FP32)
            lea(rax, ptr[rsp + reg_ctx_off]);
            vmovups(ctx, ptr[rax + rcx * 4]);

            // Load BF16 weights and convert to FP32
            vpmovzxwd(wo, ptr[r9 + rcx * 2]); // Load 16 × BF16 → 16 × DWORD (zero-extended)
            vpslld(wo, wo, 16);               // Shift to high 16 bits → FP32

            vfmadd231ps(acc, ctx, wo);

            add(rcx, 16);
            jmp(".col", T_NEAR);
            L(".col_end");

            emit_horizontal_sum_to_scalar(acc);
            vmovss(ptr[reg_out + r8 * 4], Xmm(acc.getIdx()));

            inc(r8);
            jmp(".row", T_NEAR);
            L(".end");
            outLocalLabel();
        }

        /**
         * @brief Q8_1 Wo projection for single query
         *
         * Matrix-vector multiply with Q8_1 quantized weights.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * Q8_1 BLOCK LAYOUT (36 bytes)
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Offset | Size | Content
         * -------|------|--------
         *   0    |  2   | FP16 scale (d)
         *   2    |  2   | INT16 sum_qs (unused for FP32 input)
         *   4    | 32   | 32 × INT8 quantized weights (qs)
         *
         * Dequantization: w_fp32[i] = qs[i] * float(d)
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER USAGE
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Accumulator:
         *   zmm_scratch(0): acc = running dot product
         *
         * Context:
         *   zmm_scratch(1): ctx_lo = context[blk*32 .. blk*32+15]
         *   zmm_scratch(2): ctx_hi = context[blk*32+16 .. blk*32+31]
         *
         * Weights:
         *   zmm_scratch(3): d = scale (broadcast)
         *   zmm_scratch(4): wo_lo = dequantized weights [0..15]
         *   zmm_scratch(5): wo_hi = dequantized weights [16..31]
         *
         * Loop counters:
         *   r8: row index
         *   r9: Wo row base pointer
         *   rcx: block index within row
         */
        void emit_prefill_wo_q8_1(const Xbyak::Reg64 &reg_Wo, const Xbyak::Reg64 &reg_out,
                                  const Xbyak::Reg64 &reg_ctx_off, int d_model)
        {
            using namespace Xbyak;
            inLocalLabel();

            int num_blocks = d_model / 32;

            // Outer loop: rows
            xor_(r8, r8);
            L(".row");
            cmp(r8, d_model);
            jge(".end", T_NEAR);

            // Clear accumulator
            Zmm acc = scratch0().zmm();
            vxorps(acc, acc, acc);

            // Calculate Wo row pointer: Wo + row * num_blocks * 36
            mov(r9, r8);
            imul(r9, r9, num_blocks * 36);
            add(r9, reg_Wo);

            // Inner loop: blocks
            xor_(rcx, rcx);
            L(".blk");
            cmp(rcx, num_blocks);
            jge(".blk_end", T_NEAR);

            // Calculate context block pointer: ctx_offset + blk * 128
            mov(rdx, rcx);
            shl(rdx, 7); // blk * 128 (32 floats × 4 bytes)
            lea(rax, ptr[rsp + reg_ctx_off]);
            add(rdx, rax);

            // Load context block (32 floats = 2 × ZMM)
            Zmm ctx_lo = scratch1().zmm();
            Zmm ctx_hi = scratch2().zmm();
            vmovups(ctx_lo, ptr[rdx]);
            vmovups(ctx_hi, ptr[rdx + 64]);

            // Calculate Wo block offset: blk * 36
            mov(rax, rcx);
            imul(rax, rax, 36);
            add(rax, r9);

            // Load and broadcast FP16 scale
            Zmm d = scratch3().zmm();
            vpbroadcastw(Xmm(d.getIdx()), ptr[rax]);     // Load FP16
            vcvtph2ps(Xmm(d.getIdx()), Xmm(d.getIdx())); // FP16 → FP32
            vbroadcastss(d, Xmm(d.getIdx()));            // Broadcast to all lanes

            // Load and dequantize INT8 weights
            Zmm wo_lo = scratch4().zmm();
            Zmm wo_hi = scratch5().zmm();
            vpmovsxbd(wo_lo, ptr[rax + 4]);      // qs[0..15] INT8 → INT32
            vpmovsxbd(wo_hi, ptr[rax + 4 + 16]); // qs[16..31] INT8 → INT32
            vcvtdq2ps(wo_lo, wo_lo);             // INT32 → FP32
            vcvtdq2ps(wo_hi, wo_hi);
            vmulps(wo_lo, wo_lo, d); // Dequantize: qs * scale
            vmulps(wo_hi, wo_hi, d);

            // Accumulate: acc += context · weights
            vfmadd231ps(acc, ctx_lo, wo_lo);
            vfmadd231ps(acc, ctx_hi, wo_hi);

            inc(rcx);
            jmp(".blk", T_NEAR);
            L(".blk_end");

            // Horizontal sum and store
            emit_horizontal_sum_to_scalar(acc);
            vmovss(ptr[reg_out + r8 * 4], Xmm(acc.getIdx()));

            inc(r8);
            jmp(".row", T_NEAR);
            L(".end");
            outLocalLabel();
        }

        /**
         * @brief Emit attention computation for a single query position (DECODE mode)
         *
         * This is the core decode-mode attention routine. Unlike prefill mode which
         * processes tiles of queries, decode mode processes one query at a time
         * (the newly generated token) against the entire KV cache.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * ALGORITHM OVERVIEW
         * ═══════════════════════════════════════════════════════════════════════
         *
         * For each head h ∈ [0, num_heads):
         *   1. Initialize context accumulators = 0
         *   2. Initialize softmax state (max = -∞, sum = 0)
         *   3. Load Q[q, h] into ZMM registers (kept for entire KV loop)
         *   4. For each KV position k ∈ [0, max_kv_pos):
         *      - score = Q[q,h] · K[k,kv_h] / sqrt(head_dim)
         *      - Update (max, sum, context) with online softmax
         *   5. Normalize context by 1/sum
         *   6. Store to context buffer
         *
         * After all heads:
         *   7. Apply Wo projection: output = context × Wo^T
         *
         * ═══════════════════════════════════════════════════════════════════════
         * CAUSAL MASKING
         * ═══════════════════════════════════════════════════════════════════════
         *
         * When config_.causal is true:
         *   max_kv_pos = min(q_idx + position_offset + 1, seq_len_kv)
         *
         * This ensures each query can only attend to past tokens (including itself).
         *
         * ═══════════════════════════════════════════════════════════════════════
         * GQA (Grouped Query Attention)
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Supports GQA where multiple query heads share the same KV head.
         * kv_head = h / heads_per_kv where heads_per_kv = num_heads / num_kv_heads
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER ALLOCATION
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Preserved callee-saved registers (set up in prologue):
         *   r12: reg_Q (Q pointer)
         *   r13: reg_K (K pointer)
         *   r14: reg_V (V pointer)
         *   r15: reg_seq_len_kv
         *   rbx: reg_output
         *   rbp: reg_Wo
         *
         * Volatile registers (clobbered per head):
         *   rdi, rsi: Q/K/V base pointers
         *   rcx: KV position loop counter
         *   rax: scratch for computations
         *
         * @param reg_Q Q matrix pointer [seq_len_q, num_heads, head_dim] Q8_1
         * @param reg_K K cache pointer [seq_len_kv, num_kv_heads, head_dim] Q8_1
         * @param reg_V V cache pointer [seq_len_kv, num_kv_heads, head_dim] Q8_1
         * @param reg_Wo Wo projection weights
         * @param reg_output Output buffer
         * @param reg_q_idx Query position index
         * @param reg_seq_len_kv Length of KV cache
         * @param num_blocks Number of Q8_1 blocks per head (head_dim / 32)
         * @param spill_base_offset Stack offset for V accumulator spill
         * @param q_stack_offset Stack offset for Q blocks (unused in decode)
         * @param context_buffer_offset Stack offset for context buffer
         * @param q_idx_spill_offset Stack offset to spill q_idx
         * @param position_offset_spill_offset Stack offset for position_offset
         */
        void emit_single_query_attention(
            const Xbyak::Reg64 &reg_Q,
            const Xbyak::Reg64 &reg_K,
            const Xbyak::Reg64 &reg_V,
            const Xbyak::Reg64 &reg_Wo,
            const Xbyak::Reg64 &reg_output,
            const Xbyak::Reg64 &reg_q_idx,
            const Xbyak::Reg64 &reg_seq_len_kv,
            int num_blocks,
            int spill_base_offset,
            int q_stack_offset,
            int context_buffer_offset,
            int q_idx_spill_offset,
            int position_offset_spill_offset,
            int fa2_scores_spill_offset,
            int fa2_max1_spill_offset)
        {
            using namespace Xbyak;

            debug_emit("emit_single_query_attention");

            // Save reg_q_idx to stack - it gets clobbered by emitters that use eax
            mov(ptr[rsp + q_idx_spill_offset], reg_q_idx);

            // For causal attention, compute max_kv_pos = min(q_idx + position_offset + 1, seq_len_kv)
            // We store max_kv_pos in a register for the KV loop comparison
            Reg64 reg_max_kv_pos = r11; // Reuse r11 temporarily before Q base calculation

            if (config_.causal)
            {
                // max_kv_pos = q_idx + position_offset + 1
                mov(reg_max_kv_pos, reg_q_idx);
                add(reg_max_kv_pos, ptr[rsp + position_offset_spill_offset]);
                inc(reg_max_kv_pos); // +1 because we want to include position [q + offset]

                // max_kv_pos = min(max_kv_pos, seq_len_kv)
                cmp(reg_max_kv_pos, reg_seq_len_kv);
                cmovg(reg_max_kv_pos, reg_seq_len_kv); // if max_kv_pos > seq_len_kv, use seq_len_kv
            }
            else
            {
                // Non-causal: attend to all KV positions
                mov(reg_max_kv_pos, reg_seq_len_kv);
            }

            // For GQA, compute KV head index
            int heads_per_kv = config_.num_heads / config_.num_kv_heads;

            // Q layout: [seq_len_q, num_heads, head_dim] in Q8_1 blocks
            int q_stride = config_.num_heads * num_blocks * 36; // bytes per query position

            // Phase 1: Compute attention context for all heads
            // Store to context buffer on stack (not output yet)
            for (int h = 0; h < config_.num_heads; ++h)
            {
                int kv_head = h / heads_per_kv;
                std::string head_label = "q" + std::to_string(config_.batch_size) + "_h" + std::to_string(h);

                // Initialize context accumulators for this head
                v_accum_emitter_.emit_init_context(*this, num_blocks, spill_base_offset);

                // Initialize softmax state for this head
                // max = -FLT_MAX, sum = 0
                load_constant_f32(state_max().zmm(), -3.4028235e+38f); // -FLT_MAX
                vxorps(state_sum().zmm(), state_sum().zmm(), state_sum().zmm());

                // Initialize constant for unsigned Q conversion
                // NOTE: Use rsi as scratch (rdi is used for Q_base below)
                emit_broadcast_i32_const(const_128().zmm(), 0x80808080, rsi);

                // Calculate Q base pointer for this query position
                // NOTE: Must be computed inside head loop because rdi gets clobbered
                //       by emit_single_head_attention (used for K/V pointers)
                // NOTE: Load q_idx from spill slot since rax is clobbered by emitters
                // Use rsi for Q base (rdi is used for K/V pointers in single_head_attention)
                Reg64 reg_Q_base = rsi;
                mov(reg_Q_base, ptr[rsp + q_idx_spill_offset]); // Load saved q_idx
                imul(reg_Q_base, reg_Q_base, q_stride);
                add(reg_Q_base, reg_Q); // reg_Q = r12, preserved callee-saved

                // Load Q[q,h] blocks to registers for this head
                // Optimized for Decode: Keep Q in ZMM registers across KV loop
                emit_load_q_head_to_registers(reg_Q_base, h, num_blocks);

                // Loop over KV positions (up to max_kv_pos for causal, seq_len_kv for non-causal)
                // Use unique labels for each head to avoid conflicts
                std::string loop_label = head_label + "_kv_loop";
                std::string end_label = head_label + "_kv_end";

                Reg64 reg_kv_idx = rcx; // Reuse rcx (Wo moved to rbp)
                xor_(reg_kv_idx, reg_kv_idx);

                if (config_.use_fa2_tiling)
                {
                    // ═══════════════════════════════════════════════════════════════
                    // FA2-STYLE KV LOOP: Process 4 positions at a time
                    // ═══════════════════════════════════════════════════════════════
                    // Benefits:
                    //   - Context rescale once per tile (not 4x)
                    //   - Single softmax state update per tile
                    //   - Interleaved V loads hide memory latency

                    std::string fa2_tile_loop = head_label + "_fa2_tile_loop";
                    std::string fa2_tile_end = head_label + "_fa2_tile_end";
                    std::string fa2_remainder_loop = head_label + "_fa2_remainder";
                    std::string fa2_remainder_end = head_label + "_fa2_remainder_end";

                    const int fa2_kv_step =
                        (config_.fa2_tile_width == JitAttentionConfig::Fa2TileWidth::KV8) ? 8 : 4;

                    Reg64 reg_tile_bound = r8;
                    mov(reg_tile_bound, reg_max_kv_pos);

                    // Round down to multiple of the selected tile width.
                    if (fa2_kv_step == 8)
                    {
                        and_(reg_tile_bound, ~7);
                    }
                    else
                    {
                        and_(reg_tile_bound, ~3);
                    }

                    // Tile loop: Process KV in steps of 4 or 8.
                    L(fa2_tile_loop.c_str());
                    cmp(reg_kv_idx, reg_tile_bound);
                    jge(fa2_tile_end.c_str(), T_NEAR);

                    if (fa2_kv_step == 8)
                    {
                        emit_fa2_tile_attention_8x(
                            reg_Q_base, reg_K, reg_V,
                            reg_kv_idx, h, kv_head,
                            num_blocks, spill_base_offset,
                            head_label,
                            fa2_scores_spill_offset, fa2_max1_spill_offset);
                        add(reg_kv_idx, 8);
                    }
                    else
                    {
                        emit_fa2_tile_attention_4x(
                            reg_Q_base, reg_K, reg_V,
                            reg_kv_idx, h, kv_head,
                            num_blocks, spill_base_offset,
                            head_label);
                        add(reg_kv_idx, 4);
                    }
                    jmp(fa2_tile_loop.c_str(), T_NEAR);
                    L(fa2_tile_end.c_str());

                    // Remainder Loop: Process remaining 0-3 positions one at a time
                    L(fa2_remainder_loop.c_str());
                    cmp(reg_kv_idx, reg_max_kv_pos);
                    jge(fa2_remainder_end.c_str(), T_NEAR);

                    emit_single_head_attention(
                        reg_Q_base, reg_K, reg_V,
                        reg_kv_idx, h, kv_head,
                        num_blocks, spill_base_offset,
                        q_stack_offset, head_label);

                    inc(reg_kv_idx);
                    jmp(fa2_remainder_loop.c_str(), T_NEAR);

                    L(fa2_remainder_end.c_str());
                }
                else
                {
                    // ═══════════════════════════════════════════════════════════════
                    // ORIGINAL KV LOOP: One position at a time (FA1 style)
                    // ═══════════════════════════════════════════════════════════════
                    L(loop_label.c_str());
                    cmp(reg_kv_idx, reg_max_kv_pos); // Compare against max_kv_pos (causal-aware)
                    jge(end_label.c_str(), T_NEAR);

                    emit_single_head_attention(
                        reg_Q_base, reg_K, reg_V,
                        reg_kv_idx, h, kv_head,
                        num_blocks, spill_base_offset,
                        q_stack_offset, head_label);

                    inc(reg_kv_idx);
                    jmp(loop_label.c_str(), T_NEAR);

                    L(end_label.c_str());
                }

                // Normalize context by 1/sum for this head
                Zmm zmm_inv_sum = scratch4().zmm();
                load_constant_f32(scratch5().zmm(), 1.0f);
                vdivps(zmm_inv_sum, scratch5().zmm(), state_sum().zmm());
                v_accum_emitter_.emit_normalize_context(*this, zmm_inv_sum, num_blocks, spill_base_offset);

                // Store this head's context to context buffer (not output)
                emit_store_head_to_context_buffer(h, num_blocks, spill_base_offset, context_buffer_offset);
            }

            // Phase 2: Apply Wo projection
            // context buffer: [num_heads * head_dim] FP32
            // Wo: [d_model, num_heads * head_dim] (row-major)
            // output: [d_model] FP32
            // output[i] = sum_j(context[j] * Wo[i, j])

            // Restore reg_q_idx before output calculation
            mov(reg_q_idx, ptr[rsp + q_idx_spill_offset]);

            emit_wo_projection(reg_Wo, reg_output, reg_q_idx, context_buffer_offset);
        }

        /**
         * @brief Copy Q blocks for one head from input to stack buffer
         *
         * Copies Q8_1 blocks to a stack-aligned buffer for repeated access.
         * Used in prefill mode where Q values are reused across KV positions.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * MEMORY LAYOUT
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Source (Q8_1 packed):
         *   reg_Q_base + head_idx * num_blocks * 36 + block * 36
         *   Block layout: [scale:2, sum_qs:2, qs:32] = 36 bytes
         *
         * Destination (stack, padded):
         *   rsp + q_stack_offset + block * 64
         *   Padded to 64-byte alignment for AVX-512 loads
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER USAGE
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Scratch:
         *   zmm_scratch(0): 256-bit copy buffer
         *   esi: 32-bit copy for scale + sum_qs header
         *
         * @param reg_Q_base Q pointer + q_idx offset
         * @param head_idx Head index to copy
         * @param num_blocks Blocks per head (head_dim / 32)
         * @param q_stack_offset Stack offset for destination buffer
         */
        void emit_copy_q_head_to_stack(
            const Xbyak::Reg64 &reg_Q_base,
            int head_idx,
            int num_blocks,
            int q_stack_offset)
        {
            using namespace Xbyak;

            debug_emit("emit_copy_q_head_to_stack (head " + std::to_string(head_idx) + ")");

            int q_head_offset = head_idx * num_blocks * 36;

            // Use scratch zone ZMM to avoid clobbering accumulators
            // scratch0().zmm() = zmm20, we'll use the lower 256 bits
            Zmm zmm_tmp = scratch0().zmm();

            for (int b = 0; b < num_blocks; ++b)
            {
                int src_offset = q_head_offset + b * 36;
                int dst_offset = q_stack_offset + b * 64; // Padded to 64 bytes

                // Copy 32 bytes (data portion) using AVX-512 load/store (256-bit)
                // vmovdqu32 is the AVX-512 32-bit version
                vmovdqu32(Ymm(zmm_tmp.getIdx()), ptr[reg_Q_base + src_offset + 4]); // Data at offset 4

                vmovdqu32(ptr[rsp + dst_offset + 4], Ymm(zmm_tmp.getIdx()));

                // Copy 4 bytes (scale + sum_qs) using esi as scratch
                // NOTE: Use esi (not edi) since rdi might be reg_Q_base
                mov(esi, ptr[reg_Q_base + src_offset]);
                mov(ptr[rsp + dst_offset], esi);
            }
        }

        /**
         * @brief Load Q blocks for one head into ZMM registers
         *
         * Optimized for DECODE mode: Loads Q directly into ZMM registers and
         * keeps them resident across the entire KV loop to minimize memory
         * traffic. Also applies unsigned conversion (XOR with 0x80) for VNNI.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER ALLOCATION
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Output (persistent across KV loop):
         *   zmm10 + b: Q data for block b (converted to unsigned INT8)
         *
         * Uses zmm10-13 for Q data (up to 4 blocks = head_dim 128).
         * This is safe because:
         *   - emit_fast_exp uses zmm8, zmm14, zmm15 (not zmm10-13)
         *   - V accumulation uses zmm8-9 (not zmm10+)
         *
         * The Q scale values are stored in memory and loaded dynamically
         * during dot product computation.
         *
         * @param reg_Q_base Q pointer for current query position
         * @param head_idx Head index to load
         * @param num_blocks Blocks per head (typically 2 for head_dim=64)
         */
        void emit_load_q_head_to_registers(
            const Xbyak::Reg64 &reg_Q_base,
            int head_idx,
            int num_blocks)
        {
            using namespace Xbyak;

            debug_emit("emit_load_q_head_to_registers (head " + std::to_string(head_idx) + ")");

            int q_head_offset = head_idx * num_blocks * 36;

            for (int b = 0; b < num_blocks; ++b)
            {
                int src_offset = q_head_offset + b * 36;

                // Load Q data (32 int8) -> ZMM (10 + b)
                // Use zmm10-13 for Q data, avoiding:
                //   - zmm8-9: V accumulation scratch
                //   - zmm14-15: emit_fast_exp scratch
                //   - zmm16-19: StateRegs
                //   - zmm26-31: ConstRegs
                Ymm ymm_q(10 + b);
                vmovdqu8(ymm_q, ptr[reg_Q_base + src_offset + 4]);

                // Convert signed INT8 to unsigned (XOR 0x80) for VNNI compatibility
                // After this, values are in [0, 255] instead of [-128, 127]
                vpxord(ymm_q, ymm_q, Ymm(const_128().zmm().getIdx()));
            }
        }

        /**
         * @brief Store head context directly to output buffer
         *
         * Simplified output path for testing: stores FP32 context directly
         * without Wo projection. Used for debugging context computation.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * OUTPUT LAYOUT
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Output: [seq_len_q, num_heads × head_dim] FP32
         *
         * For query q and head h:
         *   output[q * d_model + h * head_dim ... + head_dim - 1]
         *
         * ═══════════════════════════════════════════════════════════════════════
         * ACCUMULATOR LAYOUT
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Register-resident (first 2 blocks = 64 floats):
         *   zmm_accum(0): floats [0..15]
         *   zmm_accum(1): floats [16..31]
         *   zmm_accum(2): floats [32..47]
         *   zmm_accum(3): floats [48..63]
         *
         * Stack-spilled (blocks 2+):
         *   spill_base_offset + (b-2) * 128: floats [b*32 .. b*32+31]
         *
         * @param reg_output Output buffer pointer
         * @param reg_q_idx Query index
         * @param head_idx Head index
         * @param num_blocks Blocks per head
         * @param spill_base_offset Stack offset for spilled blocks
         */
        void emit_store_head_context(
            const Xbyak::Reg64 &reg_output,
            const Xbyak::Reg64 &reg_q_idx,
            int head_idx,
            int num_blocks,
            int spill_base_offset)
        {
            using namespace Xbyak;

            debug_emit("emit_store_head_context (head " + std::to_string(head_idx) + ")");

            // Output layout: [seq_len_q, num_heads * head_dim] FP32
            // Each head outputs head_dim floats = num_blocks * 32 floats
            int head_dim = num_blocks * 32;
            int d_model = config_.num_heads * head_dim;
            int out_offset_per_q = d_model * 4;        // FP32 output stride per query
            int head_offset = head_idx * head_dim * 4; // Offset for this head

            // Calculate output pointer: output + q_idx * d_model * 4 + head_idx * head_dim * 4
            Reg64 reg_out_ptr = rdi;
            mov(reg_out_ptr, reg_q_idx);
            imul(reg_out_ptr, reg_out_ptr, out_offset_per_q);
            add(reg_out_ptr, head_offset);
            add(reg_out_ptr, reg_output);

            // Store register-resident accumulators
            vmovups(ptr[reg_out_ptr], accum0().zmm());      // floats 0-15
            vmovups(ptr[reg_out_ptr + 64], accum1().zmm()); // floats 16-31

            if (num_blocks >= 2)
            {
                vmovups(ptr[reg_out_ptr + 128], accum2().zmm()); // floats 32-47
                vmovups(ptr[reg_out_ptr + 192], accum3().zmm()); // floats 48-63
            }

            // Store spilled accumulators
            if (num_blocks > 2)
            {
                for (int b = 2; b < num_blocks; ++b)
                {
                    int spill_lo = spill_base_offset + (b - 2) * 128;
                    int spill_hi = spill_lo + 64;
                    int out_lo = b * 64 * 2; // b * 32 floats * 2 halves * 4 bytes
                    int out_hi = out_lo + 64;

                    vmovups(scratch0().zmm(), ptr[rsp + spill_lo]);
                    vmovups(ptr[reg_out_ptr + out_lo], scratch0().zmm());

                    vmovups(scratch0().zmm(), ptr[rsp + spill_hi]);
                    vmovups(ptr[reg_out_ptr + out_hi], scratch0().zmm());
                }
            }
        }

        /**
         * @brief Store head context to stack buffer for Wo projection
         *
         * Collects normalized context from ZMM accumulators and stack spill
         * slots into a contiguous buffer for efficient Wo matrix multiplication.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * CONTEXT BUFFER LAYOUT
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Buffer: [num_heads × head_dim] FP32 contiguous on stack
         *
         * For head h:
         *   context_buffer_offset + h * head_dim * 4 ... + (h+1) * head_dim * 4
         *
         * ═══════════════════════════════════════════════════════════════════════
         * ACCUMULATOR LAYOUT
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Same as emit_store_head_context - first 2 blocks in registers,
         * remaining blocks spilled to stack at spill_base_offset.
         *
         * @param head_idx Head index
         * @param num_blocks Blocks per head (head_dim / 32)
         * @param spill_base_offset Stack offset for spilled accumulators
         * @param context_buffer_offset Stack offset for output context buffer
         */
        void emit_store_head_to_context_buffer(
            int head_idx,
            int num_blocks,
            int spill_base_offset,
            int context_buffer_offset)
        {
            using namespace Xbyak;

            debug_emit("emit_store_head_to_context_buffer (head " + std::to_string(head_idx) + ")");

            // Context buffer layout: [num_heads * head_dim] FP32 on stack
            int head_dim = num_blocks * 32;
            int head_offset = head_idx * head_dim * 4; // FP32 bytes for this head
            int ctx_ptr = context_buffer_offset + head_offset;

            // Store register-resident accumulators (first 2 blocks = 64 floats in zmm0-3)
            vmovups(ptr[rsp + ctx_ptr], accum0().zmm());      // floats 0-15
            vmovups(ptr[rsp + ctx_ptr + 64], accum1().zmm()); // floats 16-31

            if (num_blocks >= 2)
            {
                vmovups(ptr[rsp + ctx_ptr + 128], accum2().zmm()); // floats 32-47
                vmovups(ptr[rsp + ctx_ptr + 192], accum3().zmm()); // floats 48-63
            }

            // Store spilled accumulators (blocks 2+)
            if (num_blocks > 2)
            {
                for (int b = 2; b < num_blocks; ++b)
                {
                    int spill_lo = spill_base_offset + (b - 2) * 128;
                    int spill_hi = spill_lo + 64;
                    int out_lo = ctx_ptr + b * 64 * 2; // offset for this block's low half
                    int out_hi = out_lo + 64;

                    // Load from spill slot and store to context buffer
                    vmovups(scratch0().zmm(), ptr[rsp + spill_lo]);
                    vmovups(ptr[rsp + out_lo], scratch0().zmm());

                    vmovups(scratch0().zmm(), ptr[rsp + spill_hi]);
                    vmovups(ptr[rsp + out_hi], scratch0().zmm());
                }
            }
        }

        /**
         * @brief Dispatch Wo projection for DECODE mode
         *
         * Computes output = context × Wo^T where context is stored in the
         * context buffer on stack.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * MATRIX DIMENSIONS
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Context buffer: [d_model] = [num_heads × head_dim] FP32
         * Wo weights: [d_model, d_model] (row-major, format-dependent)
         * Output: [d_model] FP32
         *
         * For each output row i ∈ [0, d_model):
         *   output[i] = Σ_j context[j] × Wo[i, j]
         *
         * ═══════════════════════════════════════════════════════════════════════
         * FORMAT DISPATCH
         * ═══════════════════════════════════════════════════════════════════════
         *
         * WoFormat::FP32: Direct FP32 dot products
         * WoFormat::FP16: FP16 → FP32 conversion via vcvtph2ps
         * WoFormat::BF16: BF16 → FP32 conversion via shift-left-16
         * WoFormat::Q8_1: Dequantize blocks then dot product
         *
         * @param reg_Wo Wo weight matrix pointer
         * @param reg_output Output buffer pointer
         * @param reg_q_idx Query index for output offset calculation
         * @param context_buffer_offset Stack offset for context buffer
         */
        void emit_wo_projection(
            const Xbyak::Reg64 &reg_Wo,
            const Xbyak::Reg64 &reg_output,
            const Xbyak::Reg64 &reg_q_idx,
            int context_buffer_offset)
        {
            using namespace Xbyak;

            debug_emit("emit_wo_projection");

            int d_model = config_.num_heads * config_.head_dim;

            // Calculate output pointer: output + q_idx * d_model * 4
            // Use r11 since rdi is used as scratch inside emit_wo_projection_* methods
            Reg64 reg_out_ptr = r11;
            mov(reg_out_ptr, reg_q_idx);
            imul(reg_out_ptr, reg_out_ptr, d_model * 4);
            add(reg_out_ptr, reg_output);

            // Dispatch based on Wo format (and optimization flag for FP32)
            switch (config_.wo_format)
            {
            case WoFormat::FP32:
                if (config_.use_optimized_wo)
                {
                    emit_wo_projection_fp32_optimized(reg_Wo, reg_out_ptr, context_buffer_offset, d_model);
                }
                else
                {
                    emit_wo_projection_fp32(reg_Wo, reg_out_ptr, context_buffer_offset, d_model);
                }
                break;
            case WoFormat::FP16:
                emit_wo_projection_fp16(reg_Wo, reg_out_ptr, context_buffer_offset, d_model);
                break;
            case WoFormat::BF16:
                emit_wo_projection_bf16(reg_Wo, reg_out_ptr, context_buffer_offset, d_model);
                break;
            case WoFormat::Q8_1:
                emit_wo_projection_q8_1(reg_Wo, reg_out_ptr, context_buffer_offset, d_model);
                break;
            case WoFormat::Q8_1_VNNI_PACKED:
            {
                // A = context buffer on stack
                lea(rsi, ptr[rsp + context_buffer_offset]);

                // SysV args: rdi=WoPacked, rsi=A, rdx=C, rcx=m, r8=n, r9=k
                mov(rdi, reg_Wo);
                mov(rdx, reg_out_ptr);
                mov(ecx, 1);
                mov(r8d, d_model);
                mov(r9d, d_model);

                mov(rax, (size_t)llaminar2_wo_q8_1_vnni_packed_gemm);
                call(rax);
            }
            break;
            }
        }

        /**
         * @brief FP32 Wo projection for DECODE mode
         *
         * Simple row-major matrix-vector multiply with FP32 weights.
         * Uses runtime loops to avoid code bloat for large d_model values.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * ALGORITHM
         * ═══════════════════════════════════════════════════════════════════════
         *
         * For each output row i ∈ [0, d_model):
         *   acc = 0
         *   For j = 0, 16, 32, ... < d_model:
         *     acc += context[j:j+16] · Wo[i, j:j+16]  (vectorized 16-wide)
         *   output[i] = horizontal_sum(acc)
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER USAGE
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Loop counters (GPR):
         *   r8:  Output row index (outer loop)
         *   r9:  Inner loop index (columns, stride 16)
         *   r10: Wo row pointer
         *   rdi: Scratch for context pointer
         *
         * ZMM registers:
         *   zmm_scratch(0): Accumulator
         *   zmm_scratch(1): Context vector chunk
         *   zmm_scratch(2): Wo weight chunk
         */
        void emit_wo_projection_fp32(
            const Xbyak::Reg64 &reg_Wo,
            const Xbyak::Reg64 &reg_out_ptr,
            int context_buffer_offset,
            int d_model)
        {
            using namespace Xbyak;

            debug_emit("emit_wo_projection_fp32 (d_model=" + std::to_string(d_model) + ")");

            // Use local labels for loops to avoid conflicts when called multiple times
            inLocalLabel();

            // Use runtime loop for output rows
            // Save scratch GPRs we'll use
            Reg64 reg_out_idx = r8; // Output row index
            Reg64 reg_j = r9;       // Inner loop index
            Reg64 reg_wo_row = r10; // Wo row pointer

            xor_(reg_out_idx, reg_out_idx);

            L(".outer_loop");
            cmp(reg_out_idx, d_model);
            jge(".outer_end", T_NEAR);

            // Zero accumulator
            Zmm zmm_acc = scratch0().zmm();
            vxorps(zmm_acc, zmm_acc, zmm_acc);

            // Calculate Wo row pointer: Wo + out_idx * d_model * 4
            mov(reg_wo_row, reg_out_idx);
            imul(reg_wo_row, reg_wo_row, d_model * 4);
            add(reg_wo_row, reg_Wo);

            // Inner loop: dot product context * Wo_row
            xor_(reg_j, reg_j);

            L(".inner_loop");
            cmp(reg_j, d_model);
            jge(".inner_end", T_NEAR);

            // Load context chunk from stack
            Zmm zmm_ctx = scratch1().zmm();
            lea(rdi, ptr[rsp + context_buffer_offset]);
            vmovups(zmm_ctx, ptr[rdi + reg_j * 4]);

            // Load Wo row chunk
            Zmm zmm_wo = scratch2().zmm();
            vmovups(zmm_wo, ptr[reg_wo_row + reg_j * 4]);

            // FMA: acc += ctx * wo
            vfmadd231ps(zmm_acc, zmm_ctx, zmm_wo);

            add(reg_j, 16);
            jmp(".inner_loop", T_NEAR);

            L(".inner_end");

            // Horizontal sum to get final output value
            emit_horizontal_sum_to_scalar(zmm_acc);

            // Store result
            lea(rdi, ptr[reg_out_ptr + reg_out_idx * 4]);
            vmovss(ptr[rdi], Xmm(zmm_acc.getIdx()));

            inc(reg_out_idx);
            jmp(".outer_loop", T_NEAR);

            L(".outer_end");

            outLocalLabel();
        }

        /**
         * @brief Optimized FP32 Wo projection using BLAS-style 4×64 GEMV
         *
         * Uses JitWoProjectionOptimizedEmitter for 4× ILP over output columns.
         * This achieves ~12 GFLOP/s single-threaded vs ~8 GFLOP/s for naive.
         *
         * @param reg_Wo Pointer to Wo weights [d_model × d_model]
         * @param reg_out_ptr Pointer to output buffer for current query
         * @param context_buffer_offset Stack offset to context buffer
         * @param d_model Model dimension (K and N for GEMV)
         */
        void emit_wo_projection_fp32_optimized(
            const Xbyak::Reg64 &reg_Wo,
            const Xbyak::Reg64 &reg_out_ptr,
            int context_buffer_offset,
            int d_model)
        {
            using namespace Xbyak;

            debug_emit("emit_wo_projection_fp32_optimized (d_model=" + std::to_string(d_model) + ")");

            // Load context pointer from stack into a register
            // We use r13 which is callee-saved but we're inside JIT so it's fine
            Reg64 reg_context = r13;
            lea(reg_context, ptr[rsp + context_buffer_offset]);

            // Call the optimized GEMV emitter (row-major Wo semantics)
            // The emitter generates code for: output[rows] = Wo[rows, cols] × context[cols]
            JitWoProjectionOptimizedEmitter emitter;
            emitter.emit_gemv_wox_rowmajor_fp32(*this, reg_context, reg_Wo, reg_out_ptr, d_model, d_model);
        }

        /**
         * @brief FP16 Wo projection for DECODE mode
         *
         * Same algorithm as FP32 but loads FP16 weights and converts on-the-fly.
         * Row stride is d_model × 2 bytes.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * CONVERSION
         * ═══════════════════════════════════════════════════════════════════════
         *
         * 1. vmovups(ymm, [ptr]) - Load 16 × FP16 (32 bytes) into YMM
         * 2. vcvtph2ps(zmm, ymm) - Convert 16 × FP16 → 16 × FP32
         */
        void emit_wo_projection_fp16(
            const Xbyak::Reg64 &reg_Wo,
            const Xbyak::Reg64 &reg_out_ptr,
            int context_buffer_offset,
            int d_model)
        {
            using namespace Xbyak;

            debug_emit("emit_wo_projection_fp16 (d_model=" + std::to_string(d_model) + ")");

            inLocalLabel();

            Reg64 reg_out_idx = r8;
            Reg64 reg_j = r9;
            Reg64 reg_wo_row = r10;

            xor_(reg_out_idx, reg_out_idx);

            L(".outer_loop");
            cmp(reg_out_idx, d_model);
            jge(".outer_end", T_NEAR);

            Zmm zmm_acc = scratch0().zmm();
            vxorps(zmm_acc, zmm_acc, zmm_acc);

            // Wo row pointer: Wo + out_idx * d_model * 2 (FP16)
            mov(reg_wo_row, reg_out_idx);
            imul(reg_wo_row, reg_wo_row, d_model * 2);
            add(reg_wo_row, reg_Wo);

            xor_(reg_j, reg_j);

            L(".inner_loop");
            cmp(reg_j, d_model);
            jge(".inner_end", T_NEAR);

            Zmm zmm_ctx = scratch1().zmm();
            Zmm zmm_wo = scratch2().zmm();
            Ymm ymm_fp16 = Ymm(zmm_wo.getIdx());

            // Load context (FP32)
            lea(rdi, ptr[rsp + context_buffer_offset]);
            vmovups(zmm_ctx, ptr[rdi + reg_j * 4]);

            // Load FP16 weights and convert to FP32
            vmovups(ymm_fp16, ptr[reg_wo_row + reg_j * 2]);
            vcvtph2ps(zmm_wo, ymm_fp16);

            vfmadd231ps(zmm_acc, zmm_ctx, zmm_wo);

            add(reg_j, 16);
            jmp(".inner_loop", T_NEAR);

            L(".inner_end");

            emit_horizontal_sum_to_scalar(zmm_acc);
            lea(rdi, ptr[reg_out_ptr + reg_out_idx * 4]);
            vmovss(ptr[rdi], Xmm(zmm_acc.getIdx()));

            inc(reg_out_idx);
            jmp(".outer_loop", T_NEAR);

            L(".outer_end");

            outLocalLabel();
        }

        /**
         * @brief BF16 Wo projection for DECODE mode
         *
         * Same algorithm as FP32 but loads BF16 weights and converts on-the-fly.
         * Row stride is d_model × 2 bytes.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * CONVERSION
         * ═══════════════════════════════════════════════════════════════════════
         *
         * BF16 is the upper 16 bits of FP32 (same exponent, truncated mantissa).
         * Conversion: zero-extend to 32-bit, then shift left by 16.
         *
         * 1. vpmovzxwd(zmm, [ptr]) - Load 16 × WORD, zero-extend to 16 × DWORD
         * 2. vpslld(zmm, zmm, 16) - Shift left 16 → upper 16 bits = FP32
         */
        void emit_wo_projection_bf16(
            const Xbyak::Reg64 &reg_Wo,
            const Xbyak::Reg64 &reg_out_ptr,
            int context_buffer_offset,
            int d_model)
        {
            using namespace Xbyak;

            debug_emit("emit_wo_projection_bf16 (d_model=" + std::to_string(d_model) + ")");

            inLocalLabel();

            Reg64 reg_out_idx = r8;
            Reg64 reg_j = r9;
            Reg64 reg_wo_row = r10;

            xor_(reg_out_idx, reg_out_idx);

            L(".outer_loop");
            cmp(reg_out_idx, d_model);
            jge(".outer_end", T_NEAR);

            Zmm zmm_acc = scratch0().zmm();
            vxorps(zmm_acc, zmm_acc, zmm_acc);

            // Wo row pointer: Wo + out_idx * d_model * 2 (BF16)
            mov(reg_wo_row, reg_out_idx);
            imul(reg_wo_row, reg_wo_row, d_model * 2);
            add(reg_wo_row, reg_Wo);

            xor_(reg_j, reg_j);

            L(".inner_loop");
            cmp(reg_j, d_model);
            jge(".inner_end", T_NEAR);

            Zmm zmm_ctx = scratch1().zmm();
            Zmm zmm_wo = scratch2().zmm();

            // Load context (FP32)
            lea(rdi, ptr[rsp + context_buffer_offset]);
            vmovups(zmm_ctx, ptr[rdi + reg_j * 4]);

            // Load BF16 weights and convert to FP32
            vpmovzxwd(zmm_wo, ptr[reg_wo_row + reg_j * 2]); // Zero-extend 16-bit to 32-bit
            vpslld(zmm_wo, zmm_wo, 16);                     // Shift to high 16 bits

            vfmadd231ps(zmm_acc, zmm_ctx, zmm_wo);

            add(reg_j, 16);
            jmp(".inner_loop", T_NEAR);

            L(".inner_end");

            emit_horizontal_sum_to_scalar(zmm_acc);
            lea(rdi, ptr[reg_out_ptr + reg_out_idx * 4]);
            vmovss(ptr[rdi], Xmm(zmm_acc.getIdx()));

            inc(reg_out_idx);
            jmp(".outer_loop", T_NEAR);

            L(".outer_end");

            outLocalLabel();
        }

        /**
         * @brief Q8_1 Wo projection for DECODE mode
         *
         * Processes Wo weights in Q8_1 block format, dequantizing on-the-fly.
         * Uses block-oriented loop instead of element-oriented.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * ALGORITHM
         * ═══════════════════════════════════════════════════════════════════════
         *
         * For each output row i ∈ [0, d_model):
         *   acc = 0
         *   For each block b ∈ [0, num_blocks_per_row):
         *     Load context[b * 32 .. (b+1) * 32 - 1] (2 × ZMM)
         *     Load and dequantize Wo[i, b] block
         *     acc += context · Wo_dequant
         *   output[i] = horizontal_sum(acc)
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER USAGE
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Loop counters (GPR):
         *   r8:  Output row index (outer loop)
         *   r9:  Block index (inner loop)
         *   r10: Wo row base pointer
         *   rdi: Context and Wo block pointers (scratch)
         *
         * ZMM registers:
         *   zmm_scratch(0): Accumulator
         *   zmm_scratch(1): ctx_lo (context[k*32 .. k*32+15])
         *   zmm_scratch(2): ctx_hi (context[k*32+16 .. k*32+31])
         *   zmm_scratch(3): d (scale, broadcast)
         *   zmm_scratch(4): wo_lo (dequantized weights [0..15])
         *   zmm_scratch(5): wo_hi (dequantized weights [16..31])
         */
        void emit_wo_projection_q8_1(
            const Xbyak::Reg64 &reg_Wo,
            const Xbyak::Reg64 &reg_out_ptr,
            int context_buffer_offset,
            int d_model)
        {
            using namespace Xbyak;

            debug_emit("emit_wo_projection_q8_1 (d_model=" + std::to_string(d_model) + ")");

            inLocalLabel();

            int num_blocks_per_row = d_model / 32;

            Reg64 reg_out_idx = r8;
            Reg64 reg_b = r9;
            Reg64 reg_wo_row = r10;
            Reg64 reg_ctx_ptr = rdi;

            // Outer loop: output rows
            xor_(reg_out_idx, reg_out_idx);

            L(".outer_loop");
            cmp(reg_out_idx, d_model);
            jge(".outer_end", T_NEAR);

            // Clear accumulator
            Zmm zmm_acc = scratch0().zmm();
            vxorps(zmm_acc, zmm_acc, zmm_acc);

            // Wo row pointer: Wo + out_idx * num_blocks_per_row * 36
            mov(reg_wo_row, reg_out_idx);
            imul(reg_wo_row, reg_wo_row, num_blocks_per_row * 36);
            add(reg_wo_row, reg_Wo);

            // Inner loop: blocks
            xor_(reg_b, reg_b);

            L(".inner_loop");
            cmp(reg_b, num_blocks_per_row);
            jge(".inner_end", T_NEAR);

            // ───────────────────────────────────────────────────────────────────
            // STEP 1: Calculate context buffer pointer for block b
            //         Context is FP32 at: rsp + context_buffer_offset + b * 128
            // ───────────────────────────────────────────────────────────────────
            mov(reg_ctx_ptr, reg_b);
            shl(reg_ctx_ptr, 7); // * 128 (32 floats × 4 bytes)
            add(reg_ctx_ptr, context_buffer_offset);
            add(reg_ctx_ptr, rsp);

            // ───────────────────────────────────────────────────────────────────
            // STEP 2: Load context block (32 FP32 values in 2 ZMM registers)
            //         ctx_lo = context[b*32 + 0..15]
            //         ctx_hi = context[b*32 + 16..31]
            // ───────────────────────────────────────────────────────────────────
            Zmm zmm_ctx_lo = scratch1().zmm();
            Zmm zmm_ctx_hi = scratch2().zmm();
            vmovups(zmm_ctx_lo, ptr[reg_ctx_ptr]);
            vmovups(zmm_ctx_hi, ptr[reg_ctx_ptr + 64]);

            // ───────────────────────────────────────────────────────────────────
            // STEP 3: Calculate Wo block pointer
            //         Block layout: 36 bytes (2 FP16 d + 2 INT16 sum + 32 INT8)
            //         Wo block ptr = reg_wo_row + b * 36
            // ───────────────────────────────────────────────────────────────────
            Reg64 reg_wo_block = rdi;
            mov(reg_wo_block, reg_b);
            imul(reg_wo_block, reg_wo_block, 36);
            add(reg_wo_block, reg_wo_row);

            // ───────────────────────────────────────────────────────────────────
            // STEP 4: Load Q8_1 scale and broadcast
            //         d is FP16 at offset 0 in the block
            //         Convert FP16 → FP32 → broadcast to all 16 lanes
            // ───────────────────────────────────────────────────────────────────
            Zmm zmm_d = scratch3().zmm();
            vpbroadcastw(Xmm(zmm_d.getIdx()), ptr[reg_wo_block]);
            vcvtph2ps(Xmm(zmm_d.getIdx()), Xmm(zmm_d.getIdx()));
            vbroadcastss(zmm_d, Xmm(zmm_d.getIdx()));

            // ───────────────────────────────────────────────────────────────────
            // STEP 5: Load and sign-extend Q8_1 weight data
            //         INT8 data at offset 4: 32 bytes (16 + 16)
            //         vpmovsxbd: sign-extend 16 INT8 → 16 INT32
            // ───────────────────────────────────────────────────────────────────
            Zmm zmm_wo_lo = scratch4().zmm();
            Zmm zmm_wo_hi = scratch5().zmm();
            vpmovsxbd(zmm_wo_lo, ptr[reg_wo_block + 4]);
            vpmovsxbd(zmm_wo_hi, ptr[reg_wo_block + 4 + 16]);

            // ───────────────────────────────────────────────────────────────────
            // STEP 6: Dequantize: convert INT32 → FP32, multiply by scale
            //         result = (float)qs[i] * d
            // ───────────────────────────────────────────────────────────────────
            vcvtdq2ps(zmm_wo_lo, zmm_wo_lo);
            vcvtdq2ps(zmm_wo_hi, zmm_wo_hi);
            vmulps(zmm_wo_lo, zmm_wo_lo, zmm_d);
            vmulps(zmm_wo_hi, zmm_wo_hi, zmm_d);

            // ───────────────────────────────────────────────────────────────────
            // STEP 7: Accumulate dot product: acc += ctx · wo
            //         Two FMAs for the two halves of the block
            // ───────────────────────────────────────────────────────────────────
            vfmadd231ps(zmm_acc, zmm_ctx_lo, zmm_wo_lo);
            vfmadd231ps(zmm_acc, zmm_ctx_hi, zmm_wo_hi);

            inc(reg_b);
            jmp(".inner_loop", T_NEAR);

            L(".inner_end");

            // ───────────────────────────────────────────────────────────────────
            // STEP 8: Reduce ZMM accumulator to scalar and store
            // ───────────────────────────────────────────────────────────────────
            emit_horizontal_sum_to_scalar(zmm_acc);
            lea(rdi, ptr[reg_out_ptr + reg_out_idx * 4]);
            vmovss(ptr[rdi], Xmm(zmm_acc.getIdx()));

            inc(reg_out_idx);
            jmp(".outer_loop", T_NEAR);

            L(".outer_end");

            outLocalLabel();
        }

        /**
         * @brief Horizontal sum of ZMM register to scalar in xmm[0]
         *
         * Reduces 16 FP32 elements in a ZMM register to a single scalar value.
         * The result is stored in element 0 of the input ZMM's corresponding XMM.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * ALGORITHM
         * ═══════════════════════════════════════════════════════════════════════
         *
         * Binary tree reduction:
         *
         *   ZMM (512-bit): [a0..a15]
         *     └─ vextractf32x8 + vaddps → YMM: [a0+a8, a1+a9, ..., a7+a15]
         *
         *   YMM (256-bit): [b0..b7]
         *     └─ vextractf32x4 + vaddps → XMM: [b0+b4, b1+b5, b2+b6, b3+b7]
         *
         *   XMM (128-bit): [c0, c1, c2, c3]
         *     └─ vshufps 0x4E + vaddps → [c0+c2, c1+c3, ...]
         *     └─ vshufps 0xB1 + vaddss → scalar sum in element 0
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER USAGE
         * ═══════════════════════════════════════════════════════════════════════
         *
         *   Input:  zmm (passed by reference, also used as output)
         *   Scratch: zmm25 (safe to use - not part of state or accumulator zones)
         *   Output: xmm[0] contains scalar sum
         *
         * @param zmm ZMM register containing 16 FP32 values to sum
         */
        void emit_horizontal_sum_to_scalar(const Xbyak::Zmm &zmm)
        {
            using namespace Xbyak;

            // ───────────────────────────────────────────────────────────────────
            // Use zmm25 (scratch) as temporary to avoid clobbering:
            // - zmm16 (StateRegs::ZMM_MAX)
            // - zmm20 (often the accumulator passed as zmm)
            // - zmm21 (used as scratch in some contexts)
            // ───────────────────────────────────────────────────────────────────
            Zmm zmm_tmp = Zmm(25);
            Ymm ymm_tmp = Ymm(zmm_tmp.getIdx());
            Xmm xmm_tmp = Xmm(zmm_tmp.getIdx());

            // ───────────────────────────────────────────────────────────────────
            // STEP 1: ZMM → YMM (add upper 256-bit half to lower)
            // ───────────────────────────────────────────────────────────────────
            vextractf32x8(ymm_tmp, zmm, 1); // ymm_tmp = zmm[8..15]
            vaddps(Ymm(zmm.getIdx()), Ymm(zmm.getIdx()), ymm_tmp);

            // ───────────────────────────────────────────────────────────────────
            // STEP 2: YMM → XMM (add upper 128-bit half to lower)
            //         vextractf32x4 supports EVEX encoding for all registers
            // ───────────────────────────────────────────────────────────────────
            vextractf32x4(xmm_tmp, Zmm(zmm.getIdx()), 1); // xmm_tmp = ymm[4..7]
            vaddps(Xmm(zmm.getIdx()), Xmm(zmm.getIdx()), xmm_tmp);

            // ───────────────────────────────────────────────────────────────────
            // STEP 3: XMM → scalar (two-stage horizontal sum)
            //         First: swap high/low 64 bits and add
            //         Then:  swap high/low 32 bits and add scalar
            // ───────────────────────────────────────────────────────────────────
            vshufps(xmm_tmp, Xmm(zmm.getIdx()), Xmm(zmm.getIdx()), 0x4E); // Swap 64-bit halves
            vaddps(Xmm(zmm.getIdx()), Xmm(zmm.getIdx()), xmm_tmp);
            vshufps(xmm_tmp, Xmm(zmm.getIdx()), Xmm(zmm.getIdx()), 0xB1); // Swap 32-bit halves
            vaddss(Xmm(zmm.getIdx()), Xmm(zmm.getIdx()), xmm_tmp);        // Final scalar add
        }

        /**
         * @brief Emit attention for single head at single KV position
         *
         * Computes one step of the attention loop for DECODE mode:
         *   1. Q[h] · K[kv, kv_h] dot product → score
         *   2. Online softmax update (max tracking, rescale, weight)
         *   3. Weighted V accumulation
         *
         * ═══════════════════════════════════════════════════════════════════════
         * ALGORITHM
         * ═══════════════════════════════════════════════════════════════════════
         *
         * For each KV position, this method is called once per head:
         *
         *   score = dot(Q[head], K[kv_pos, kv_head])
         *   (max_new, sum_new, weight, corr) = online_softmax_update(score, max, sum)
         *   context = context * corr + weight * V[kv_pos, kv_head]
         *
         * The online softmax tracks running max and sum to avoid two-pass softmax.
         * When max increases, previous accumulators are rescaled by corr = exp(max_old - max_new).
         *
         * ═══════════════════════════════════════════════════════════════════════
         * MEMORY LAYOUT
         * ═══════════════════════════════════════════════════════════════════════
         *
         * K layout: [seq_len_kv, num_kv_heads, head_dim] in Q8_1 blocks
         *   K[kv_idx, kv_head] at: K_base + kv_idx * (num_kv_heads * num_blocks * 36)
         *                                + kv_head * num_blocks * 36
         *
         * V layout: Same as K
         *
         * Q blocks: Pre-copied to stack at q_stack_offset (for register-based dot)
         *
         * ═══════════════════════════════════════════════════════════════════════
         * REGISTER USAGE
         * ═══════════════════════════════════════════════════════════════════════
         *
         *   GPR inputs:
         *     reg_Q_ptr:  Pointer to Q tensor (for accessing scales)
         *     reg_K:      Base pointer to K tensor
         *     reg_V:      Base pointer to V tensor
         *     reg_kv_idx: Current KV position index
         *
         *   Scratch GPR:
         *     rdi:        K/V block pointer calculation
         *
         *   ZMM state (managed by softmax_emitter_ and v_accum_emitter_):
         *     zmm16: Running max
         *     zmm17: Running sum
         *     zmm18: Current weight (exp(score - max))
         *     zmm19: Correction factor
         *     zmm0-7: Context accumulators (may be spilled to stack)
         *
         * @param reg_Q_ptr      Register holding Q tensor base pointer
         * @param reg_K          Register holding K tensor base pointer
         * @param reg_V          Register holding V tensor base pointer
         * @param reg_kv_idx     Register holding current KV position index
         * @param head_idx       Query head index (determines which Q head to use)
         * @param kv_head_idx    KV head index (for GQA, multiple Q heads share one KV head)
         * @param num_blocks     Number of Q8_1 blocks per head (head_dim / 32)
         * @param spill_base_offset Stack offset for spilled accumulators
         * @param q_stack_offset Stack offset where Q blocks are copied
         * @param label_prefix   Unique prefix for internal jump labels
         */
        void emit_single_head_attention(
            const Xbyak::Reg64 &reg_Q_ptr,
            const Xbyak::Reg64 &reg_K,
            const Xbyak::Reg64 &reg_V,
            const Xbyak::Reg64 &reg_kv_idx,
            int head_idx,
            int kv_head_idx,
            int num_blocks,
            int spill_base_offset,
            int q_stack_offset,
            const std::string &label_prefix)
        {
            using namespace Xbyak;

            // ───────────────────────────────────────────────────────────────────
            // STEP 1: Calculate K pointer for this KV position and head
            //         K_ptr = K_base + kv_idx * k_stride + kv_head * head_size
            //         k_stride = num_kv_heads × num_blocks × 36 bytes
            // ───────────────────────────────────────────────────────────────────
            Reg64 reg_K_ptr = rdi; // Reuse rdi as scratch
            int k_stride = config_.num_kv_heads * num_blocks * 36;

            mov(reg_K_ptr, reg_kv_idx);
            imul(reg_K_ptr, reg_K_ptr, k_stride);
            add(reg_K_ptr, kv_head_idx * num_blocks * 36);
            add(reg_K_ptr, reg_K);

            // ───────────────────────────────────────────────────────────────────
            // STEP 2: Compute Q · K dot product
            //         score = Q[head] · K[kv_pos, kv_head]
            //         Q blocks are pre-loaded in ZMM 10+ by emit_load_q_head_to_registers
            //         K blocks are loaded from memory via reg_K_ptr
            // ───────────────────────────────────────────────────────────────────
            Xmm xmm_score_result(scratch1().zmm().getIdx()); // xmm21 for score
            Xmm xmm_scale_local(scratch2().zmm().getIdx());  // xmm22 for scale

            // Load attention scale (1/sqrt(head_dim)) from const_scale() element 0
            vmovss(xmm_scale_local, Xmm(const_scale().zmm().getIdx()));

            // Emit register-based dot product (Q in ZMM regs, K from memory)
            int q_head_offset = head_idx * num_blocks * 36;
            dot_emitter_.emit_dot_product_register_q_mem_dq(*this, xmm_score_result, reg_K_ptr, reg_Q_ptr,
                                                            q_head_offset, num_blocks, xmm_scale_local, 10);

            // ───────────────────────────────────────────────────────────────────
            // STEP 3: Online softmax update
            //         Updates running max and sum, computes attention weight
            //         Result:
            //           zmm_weight() = exp(score - max_new) / sum_new (approx)
            //           zmm_corr() = exp(max_old - max_new) if max changed, else 1.0
            // ───────────────────────────────────────────────────────────────────
            std::string softmax_label = label_prefix + "_h" + std::to_string(head_idx) + "_softmax";
            softmax_emitter_.emit_update(*this, xmm_score_result, softmax_label);

            // ───────────────────────────────────────────────────────────────────
            // STEP 4: Rescale previous context accumulators
            //         If max increased, all previous weights need rescaling:
            //           context *= corr (where corr = exp(max_old - max_new))
            //         This maintains the invariant: context = Σ (softmax_i × V_i)
            // ───────────────────────────────────────────────────────────────────
            v_accum_emitter_.emit_rescale_context(*this, num_blocks, spill_base_offset);

            // ───────────────────────────────────────────────────────────────────
            // STEP 5: Calculate V pointer (same layout as K)
            //         V_ptr = V_base + kv_idx * v_stride + kv_head * head_size
            // ───────────────────────────────────────────────────────────────────
            Reg64 reg_V_ptr = rdi;
            int v_stride = config_.num_kv_heads * num_blocks * 36;

            mov(reg_V_ptr, reg_kv_idx);
            imul(reg_V_ptr, reg_V_ptr, v_stride);
            add(reg_V_ptr, kv_head_idx * num_blocks * 36);
            add(reg_V_ptr, reg_V);

            // ───────────────────────────────────────────────────────────────────
            // STEP 6: Weighted V accumulation
            //         context += weight × V[kv_pos, kv_head]
            //         Weight is in zmm_weight() from softmax update
            //         V is dequantized from Q8_1 blocks at reg_V_ptr
            // ───────────────────────────────────────────────────────────────────
            v_accum_emitter_.emit_weighted_accum(*this, reg_V_ptr, num_blocks, spill_base_offset);
        }

        /**
         * @brief FA2-style tile processing: 4 KV positions at once
         *
         * FlashAttention-2 optimization: Process 4 KV positions in a single call,
         * computing scores, finding tile max, updating softmax state ONCE, and
         * accumulating weighted V with interleaved loads.
         *
         * ═══════════════════════════════════════════════════════════════════════
         * ALGORITHM (FA2 Tile Processing)
         * ═══════════════════════════════════════════════════════════════════════
         *
         * For a tile of 4 KV positions [kv, kv+1, kv+2, kv+3]:
         *
         *   1. Compute 4 scores simultaneously:
         *      score[0..3] = Q · K[kv+0..3] * scale
         *
         *   2. Find tile max:
         *      tile_max = max(score[0], score[1], score[2], score[3])
         *
         *   3. Update softmax state ONCE per tile:
         *      if tile_max > running_max:
         *          correction = exp(running_max - tile_max)
         *          context *= correction     // Rescale once, not 4 times!
         *          running_sum *= correction
         *          running_max = tile_max
         *
         *   4. Compute weights:
         *      weight[i] = exp(score[i] - running_max)
         *      running_sum += weight[0] + weight[1] + weight[2] + weight[3]
         *
         *   5. Interleaved V accumulation:
         *      context += weight[0] * V[kv+0]
         *      context += weight[1] * V[kv+1]
         *      context += weight[2] * V[kv+2]
         *      context += weight[3] * V[kv+3]
         *      (Loads are interleaved with FMAs to hide memory latency)
         *
         * ═══════════════════════════════════════════════════════════════════════
         * PERFORMANCE BENEFITS
         * ═══════════════════════════════════════════════════════════════════════
         *
         * vs. calling emit_single_head_attention 4 times:
         *   - Context rescale: 1× instead of up to 4×
         *   - Softmax branches: 1× instead of 4×
         *   - V load/compute overlap: pipelined across 4 rows
         *
         * @param reg_Q_ptr      Register holding Q tensor base pointer
         * @param reg_K          Register holding K tensor base pointer
         * @param reg_V          Register holding V tensor base pointer
         * @param reg_kv_idx     Register holding current KV position index (start of tile)
         * @param head_idx       Query head index
         * @param kv_head_idx    KV head index (for GQA)
         * @param num_blocks     Number of Q8_1 blocks per head (head_dim / 32)
         * @param spill_base_offset Stack offset for spilled accumulators
         * @param label_prefix   Unique prefix for internal jump labels
         */
        void emit_fa2_tile_attention_4x(
            const Xbyak::Reg64 &reg_Q_ptr,
            const Xbyak::Reg64 &reg_K,
            const Xbyak::Reg64 &reg_V,
            const Xbyak::Reg64 &reg_kv_idx,
            int head_idx,
            int kv_head_idx,
            int num_blocks,
            int spill_base_offset,
            const std::string &label_prefix)
        {
            using namespace Xbyak;
            using namespace llaminar2::jit;

            debug_emit("emit_fa2_tile_attention_4x (FA2 tile processing)");

            // ═══════════════════════════════════════════════════════════════════
            // REGISTER TRACKING: If enabled, track all register borrows
            // ═══════════════════════════════════════════════════════════════════
            if (tracking_enabled())
            {
                reset_borrows(); // Start fresh for this tile
                debug_emit("  [TRACKING] Register tracking enabled for FA2 tile");
            }

            // ───────────────────────────────────────────────────────────────────
            // STEP 1: Calculate K pointer for this KV tile
            //         K_ptr = K_base + kv_idx * k_stride + kv_head * head_size
            // ───────────────────────────────────────────────────────────────────
            Reg64 reg_K_ptr = rdi;
            int k_stride = config_.num_kv_heads * num_blocks * 36;
            int kv_head_offset = kv_head_idx * num_blocks * 36;

            mov(reg_K_ptr, reg_kv_idx);
            imul(reg_K_ptr, reg_K_ptr, k_stride);
            add(reg_K_ptr, kv_head_offset);
            add(reg_K_ptr, reg_K);

            // ───────────────────────────────────────────────────────────────────
            // SOFTWARE PREFETCH (DECODE / FA2): specialize by work size.
            //
            // Heuristic intent:
            // - SMALL: keep hot, short KV in L1 (low distance)
            // - LARGE: reduce L1 pollution, prefetch into L2 (medium distance)
            // - XL: streaming KV, prefetch into L3 (long distance)
            //
            // NOTE: This is compile-time (codegen-time) specialization via
            // JitAttentionConfig::work_size.
            // ───────────────────────────────────────────────────────────────────
            {
                const int kv_prefetch_positions =
                    (config_.work_size == WorkSizeClass::SMALL) ? 4 : (config_.work_size == WorkSizeClass::LARGE) ? 16
                                                                                                                  : 64;
                const int pf_bytes = kv_prefetch_positions * k_stride;

                if (config_.work_size == WorkSizeClass::LARGE)
                {
                    prefetcht1(ptr[reg_K_ptr + pf_bytes]);
                    if (num_blocks > 1)
                    {
                        prefetcht1(ptr[reg_K_ptr + pf_bytes + 64]);
                    }
                }
                else if (config_.work_size == WorkSizeClass::XL)
                {
                    prefetcht2(ptr[reg_K_ptr + pf_bytes]);
                    if (num_blocks > 1)
                    {
                        prefetcht2(ptr[reg_K_ptr + pf_bytes + 64]);
                    }
                }
                else
                {
                    prefetcht0(ptr[reg_K_ptr + pf_bytes]);
                    if (num_blocks > 1)
                    {
                        prefetcht0(ptr[reg_K_ptr + pf_bytes + 64]);
                    }
                }
            }

            // ───────────────────────────────────────────────────────────────────
            // STEP 2: Compute 4 Q·K scores simultaneously
            //         Using FA2 vectorized dot product emitter
            //         Q is pre-loaded in zmm10-13 by emit_load_q_head_to_registers
            // ───────────────────────────────────────────────────────────────────
            // BORROW: Score0-3 (xmm20-23) for dot product outputs
            auto guard_score0 = borrow<Score0>();
            auto guard_score1 = borrow<Score1>();
            auto guard_score2 = borrow<Score2>();
            auto guard_score3 = borrow<Score3>();

            // BORROW: Scratch4 (xmm24) for local scale copy
            auto guard_scale = borrow<Scratch4>();
            Xmm xmm_scale_local = guard_scale.xmm();

            // BORROW: Q data registers (Input2-5 = zmm10-13) - these were loaded
            //         by emit_load_q_head_to_registers and are read during dot products.
            //         We borrow them here so we know when they can be released.
            auto guard_q0 = borrow<Input2>(); // zmm10 - Q block 0 low
            auto guard_q1 = borrow<Input3>(); // zmm11 - Q block 0 high
            auto guard_q2 = borrow<Input4>(); // zmm12 - Q block 1 low (if num_blocks >= 2)
            auto guard_q3 = borrow<Input5>(); // zmm13 - Q block 1 high (if num_blocks >= 2)

            // Load attention scale (1/sqrt(head_dim))
            vmovss(xmm_scale_local, Xmm(const_scale().zmm().getIdx()));

            // Q data is pre-loaded in zmm10+ by emit_load_q_head_to_registers
            // d_Q scales are loaded on-demand inside emit_dot_product_4x to avoid register pressure.

            int q_head_offset = head_idx * num_blocks * 36;

            // emit_dot_product_4x: Q data in zmm10+, d_Q loaded from memory
            dot_emitter_.emit_dot_product_4x(
                *this,
                guard_score0.xmm(), guard_score1.xmm(), guard_score2.xmm(), guard_score3.xmm(),
                reg_K_ptr, k_stride,
                num_blocks, xmm_scale_local, 10, reg_Q_ptr, q_head_offset);

            // ═══════════════════════════════════════════════════════════════════
            // Q DATA NO LONGER NEEDED - RELEASE Q REGISTERS
            // After dot products, Q is not used again. Release these registers
            // so emit_interleaved_v_accum_4x can use Input2-3 (zmm10-11) for V loading.
            // ═══════════════════════════════════════════════════════════════════
            guard_q0.release(); // zmm10 now available
            guard_q1.release(); // zmm11 now available
            guard_q2.release(); // zmm12 now available
            guard_q3.release(); // zmm13 now available

            // ───────────────────────────────────────────────────────────────────
            // STEP 3: Find tile max (max of 4 scores)
            //         Using typed registers: Scratch4 (zmm24) is SAFE because it
            //         does NOT overlap ScoreZone (xmm20-23).
            //
            //         NOTE: We already borrowed Scratch4 for xmm_scale_local above.
            //         After vmovss, xmm24 has the scale in element 0, but the upper
            //         bits of zmm24 are undefined. We can reuse it for tile_max
            //         because emit_tile_max_reduction_4_typed will overwrite it.
            // ───────────────────────────────────────────────────────────────────
            // Scratch4 already borrowed above as guard_scale
            Zmm tile_max = guard_scale.zmm(); // Reuse zmm24

            softmax_emitter_.emit_tile_max_reduction_4_typed(
                *this, Scratch4{}, score0(), score1(), score2(), score3());

            // ───────────────────────────────────────────────────────────────────
            // STEP 4: Update softmax state ONCE for the tile
            //         Compare tile_max with running_max, apply correction if needed
            // ───────────────────────────────────────────────────────────────────
            std::string tile_label = label_prefix + "_fa2_tile";
            softmax_emitter_.emit_tile_state_update(*this, tile_max, tile_label);

            // ───────────────────────────────────────────────────────────────────
            // STEP 5: Rescale context accumulators ONCE
            //         correction factor is in zmm_corr() after emit_tile_state_update
            // ───────────────────────────────────────────────────────────────────
            v_accum_emitter_.emit_rescale_context(*this, num_blocks, spill_base_offset);

            // ───────────────────────────────────────────────────────────────────
            // STEP 6: Compute weights and accumulate sum for all 4 positions
            //         weight[i] = exp(score[i] - running_max)
            //         running_sum += weight[0] + weight[1] + weight[2] + weight[3]
            //
            //         NOTE: Weights are output to Scratch0-3 (zmm20-23), which
            //         ALIAS Score0-3 (xmm20-23). We RELEASE the score guards first
            //         and then BORROW the scratch registers for weights.
            // ───────────────────────────────────────────────────────────────────
            // RELEASE Score0-3 before reusing as Scratch0-3
            guard_score0.release();
            guard_score1.release();
            guard_score2.release();
            guard_score3.release();

            // BORROW: Scratch0-3 (zmm20-23) for weights
            // These alias the same physical registers as Score0-3, but now
            // we're using them as ZMM (full 512-bit) instead of XMM (128-bit)
            auto guard_weight0 = borrow<Scratch0>();
            auto guard_weight1 = borrow<Scratch1>();
            auto guard_weight2 = borrow<Scratch2>();
            auto guard_weight3 = borrow<Scratch3>();

            // Read the scalar scores (still in xmm20-23) and write full ZMM weights
            softmax_emitter_.emit_compute_weight_and_accumulate(*this, score0().xmm(), guard_weight0.zmm());
            softmax_emitter_.emit_compute_weight_and_accumulate(*this, score1().xmm(), guard_weight1.zmm());
            softmax_emitter_.emit_compute_weight_and_accumulate(*this, score2().xmm(), guard_weight2.zmm());
            softmax_emitter_.emit_compute_weight_and_accumulate(*this, score3().xmm(), guard_weight3.zmm());

            // ───────────────────────────────────────────────────────────────────
            // STEP 7: Calculate V pointer for this KV tile
            // ───────────────────────────────────────────────────────────────────
            Reg64 reg_V_ptr = rdi;
            int v_stride = config_.num_kv_heads * num_blocks * 36;

            mov(reg_V_ptr, reg_kv_idx);
            imul(reg_V_ptr, reg_V_ptr, v_stride);
            add(reg_V_ptr, kv_head_offset);
            add(reg_V_ptr, reg_V);

            // Prefetch V for upcoming tiles (same policy as K)
            {
                const int kv_prefetch_positions =
                    (config_.work_size == WorkSizeClass::SMALL) ? 4 : (config_.work_size == WorkSizeClass::LARGE) ? 16
                                                                                                                  : 64;
                const int pf_bytes = kv_prefetch_positions * v_stride;

                if (config_.work_size == WorkSizeClass::LARGE)
                {
                    prefetcht1(ptr[reg_V_ptr + pf_bytes]);
                    if (num_blocks > 1)
                    {
                        prefetcht1(ptr[reg_V_ptr + pf_bytes + 64]);
                    }
                }
                else if (config_.work_size == WorkSizeClass::XL)
                {
                    prefetcht2(ptr[reg_V_ptr + pf_bytes]);
                    if (num_blocks > 1)
                    {
                        prefetcht2(ptr[reg_V_ptr + pf_bytes + 64]);
                    }
                }
                else
                {
                    prefetcht0(ptr[reg_V_ptr + pf_bytes]);
                    if (num_blocks > 1)
                    {
                        prefetcht0(ptr[reg_V_ptr + pf_bytes + 64]);
                    }
                }
            }

            // ───────────────────────────────────────────────────────────────────
            // STEP 8: Interleaved V accumulation for 4 positions
            //         Overlaps V loads with FMA to hide memory latency
            //         The weights are broadcast in weight0-3 (zmm20-23), so the
            //         low element (accessible via xmm) is the scalar weight.
            // ───────────────────────────────────────────────────────────────────
            v_accum_emitter_.emit_interleaved_v_accum_4x(
                *this,
                reg_V_ptr, v_stride,
                num_blocks,
                guard_weight0.xmm(), guard_weight1.xmm(), guard_weight2.xmm(), guard_weight3.xmm(),
                spill_base_offset);

            // Guards released automatically at end of scope
        }

        /**
         * @brief FA2-style tile processing: 8 KV positions at once
         *
         * Implemented as 2× 4-wide dot products, but with a single tile max/state
         * update + a single context rescale for the combined 8 scores.
         *
         * This is decode-focused and avoids changing the existing 4x microkernel.
         *
         * @param fa2_scores_spill_offset Stack offset for 8 scalar scores (8 floats)
         * @param fa2_max1_spill_offset   Stack offset for scalar max of first 4 scores
         */
        void emit_fa2_tile_attention_8x(
            const Xbyak::Reg64 &reg_Q_ptr,
            const Xbyak::Reg64 &reg_K,
            const Xbyak::Reg64 &reg_V,
            const Xbyak::Reg64 &reg_kv_idx,
            int head_idx,
            int kv_head_idx,
            int num_blocks,
            int spill_base_offset,
            const std::string &label_prefix,
            int fa2_scores_spill_offset,
            int fa2_max1_spill_offset)
        {
            using namespace Xbyak;
            using namespace llaminar2::jit;

            debug_emit("emit_fa2_tile_attention_8x (FA2 tile processing)");

            if (tracking_enabled())
            {
                reset_borrows();
            }

            // K/V strides and head offsets
            const int k_stride = config_.num_kv_heads * num_blocks * 36;
            const int v_stride = config_.num_kv_heads * num_blocks * 36;
            const int kv_head_offset = kv_head_idx * num_blocks * 36;

            // STEP 1: K pointer for first 4 (kv..kv+3)
            Reg64 reg_K_ptr = rdi;
            mov(reg_K_ptr, reg_kv_idx);
            imul(reg_K_ptr, reg_K_ptr, k_stride);
            add(reg_K_ptr, kv_head_offset);
            add(reg_K_ptr, reg_K);

            // Prefetch K (same policy as 4x)
            {
                const int kv_prefetch_positions =
                    (config_.work_size == WorkSizeClass::SMALL) ? 4 : (config_.work_size == WorkSizeClass::LARGE) ? 16
                                                                                                                  : 64;
                const int pf_bytes = kv_prefetch_positions * k_stride;
                if (config_.work_size == WorkSizeClass::LARGE)
                {
                    prefetcht1(ptr[reg_K_ptr + pf_bytes]);
                    if (num_blocks > 1)
                        prefetcht1(ptr[reg_K_ptr + pf_bytes + 64]);
                }
                else if (config_.work_size == WorkSizeClass::XL)
                {
                    prefetcht2(ptr[reg_K_ptr + pf_bytes]);
                    if (num_blocks > 1)
                        prefetcht2(ptr[reg_K_ptr + pf_bytes + 64]);
                }
                else
                {
                    prefetcht0(ptr[reg_K_ptr + pf_bytes]);
                    if (num_blocks > 1)
                        prefetcht0(ptr[reg_K_ptr + pf_bytes + 64]);
                }
            }

            // BORROW: Score0-3 for dot outputs
            auto guard_score0 = borrow<Score0>();
            auto guard_score1 = borrow<Score1>();
            auto guard_score2 = borrow<Score2>();
            auto guard_score3 = borrow<Score3>();

            // BORROW: Scratch4 for local scale copy (xmm24/zmm24)
            auto guard_scale = borrow<Scratch4>();
            Xmm xmm_scale_local = guard_scale.xmm();

            // BORROW: Q data registers (Input2-5 = zmm10-13)
            auto guard_q0 = borrow<Input2>();
            auto guard_q1 = borrow<Input3>();
            auto guard_q2 = borrow<Input4>();
            auto guard_q3 = borrow<Input5>();

            vmovss(xmm_scale_local, Xmm(const_scale().zmm().getIdx()));

            const int q_head_offset = head_idx * num_blocks * 36;

            // STEP 2a: scores0..3
            dot_emitter_.emit_dot_product_4x(
                *this,
                guard_score0.xmm(), guard_score1.xmm(), guard_score2.xmm(), guard_score3.xmm(),
                reg_K_ptr, k_stride,
                num_blocks, xmm_scale_local, 10, reg_Q_ptr, q_head_offset);

            // Spill scores0..3
            vmovss(ptr[rsp + fa2_scores_spill_offset + 0], guard_score0.xmm());
            vmovss(ptr[rsp + fa2_scores_spill_offset + 4], guard_score1.xmm());
            vmovss(ptr[rsp + fa2_scores_spill_offset + 8], guard_score2.xmm());
            vmovss(ptr[rsp + fa2_scores_spill_offset + 12], guard_score3.xmm());

            // tile_max1 in zmm24
            softmax_emitter_.emit_tile_max_reduction_4_typed(
                *this, Scratch4{}, score0(), score1(), score2(), score3());
            vmovss(ptr[rsp + fa2_max1_spill_offset], Xmm(guard_scale.zmm().getIdx()));

            // STEP 2b: scores4..7 (advance K by 4 rows)
            add(reg_K_ptr, 4 * k_stride);
            dot_emitter_.emit_dot_product_4x(
                *this,
                guard_score0.xmm(), guard_score1.xmm(), guard_score2.xmm(), guard_score3.xmm(),
                reg_K_ptr, k_stride,
                num_blocks, xmm_scale_local, 10, reg_Q_ptr, q_head_offset);

            // Spill scores4..7
            vmovss(ptr[rsp + fa2_scores_spill_offset + 16], guard_score0.xmm());
            vmovss(ptr[rsp + fa2_scores_spill_offset + 20], guard_score1.xmm());
            vmovss(ptr[rsp + fa2_scores_spill_offset + 24], guard_score2.xmm());
            vmovss(ptr[rsp + fa2_scores_spill_offset + 28], guard_score3.xmm());

            // tile_max2 in zmm24
            softmax_emitter_.emit_tile_max_reduction_4_typed(
                *this, Scratch4{}, score0(), score1(), score2(), score3());

            // Q no longer needed after dot products
            guard_q0.release();
            guard_q1.release();
            guard_q2.release();
            guard_q3.release();

            // Combine tile_max = max(tile_max1, tile_max2)
            Xmm xmm_max1(scratch5().zmm().getIdx());
            vmovss(xmm_max1, ptr[rsp + fa2_max1_spill_offset]);
            vmaxss(Xmm(guard_scale.zmm().getIdx()), Xmm(guard_scale.zmm().getIdx()), xmm_max1);
            vbroadcastss(guard_scale.zmm(), Xmm(guard_scale.zmm().getIdx()));
            Zmm tile_max = guard_scale.zmm();

            // STEP 4: Update softmax state ONCE for the 8-wide tile
            std::string tile_label = label_prefix + "_fa2_tile8";
            softmax_emitter_.emit_tile_state_update(*this, tile_max, tile_label);

            // STEP 5: Rescale context accumulators ONCE
            v_accum_emitter_.emit_rescale_context(*this, num_blocks, spill_base_offset);

            // RELEASE Score0-3 before reusing as Scratch0-3
            guard_score0.release();
            guard_score1.release();
            guard_score2.release();
            guard_score3.release();

            // BORROW: Scratch0-3 (zmm20-23) for weights
            auto guard_weight0 = borrow<Scratch0>();
            auto guard_weight1 = borrow<Scratch1>();
            auto guard_weight2 = borrow<Scratch2>();
            auto guard_weight3 = borrow<Scratch3>();

            // STEP 7: V pointer for first 4 (kv..kv+3)
            Reg64 reg_V_ptr = rdi;
            mov(reg_V_ptr, reg_kv_idx);
            imul(reg_V_ptr, reg_V_ptr, v_stride);
            add(reg_V_ptr, kv_head_offset);
            add(reg_V_ptr, reg_V);

            // Prefetch V (same policy as 4x)
            {
                const int kv_prefetch_positions =
                    (config_.work_size == WorkSizeClass::SMALL) ? 4 : (config_.work_size == WorkSizeClass::LARGE) ? 16
                                                                                                                  : 64;
                const int pf_bytes = kv_prefetch_positions * v_stride;
                if (config_.work_size == WorkSizeClass::LARGE)
                {
                    prefetcht1(ptr[reg_V_ptr + pf_bytes]);
                    if (num_blocks > 1)
                        prefetcht1(ptr[reg_V_ptr + pf_bytes + 64]);
                }
                else if (config_.work_size == WorkSizeClass::XL)
                {
                    prefetcht2(ptr[reg_V_ptr + pf_bytes]);
                    if (num_blocks > 1)
                        prefetcht2(ptr[reg_V_ptr + pf_bytes + 64]);
                }
                else
                {
                    prefetcht0(ptr[reg_V_ptr + pf_bytes]);
                    if (num_blocks > 1)
                        prefetcht0(ptr[reg_V_ptr + pf_bytes + 64]);
                }
            }

            // GROUP 0: scores0..3 -> weights0..3 -> V accum (kv..kv+3)
            vmovss(score0().xmm(), ptr[rsp + fa2_scores_spill_offset + 0]);
            vmovss(score1().xmm(), ptr[rsp + fa2_scores_spill_offset + 4]);
            vmovss(score2().xmm(), ptr[rsp + fa2_scores_spill_offset + 8]);
            vmovss(score3().xmm(), ptr[rsp + fa2_scores_spill_offset + 12]);

            softmax_emitter_.emit_compute_weight_and_accumulate(*this, score0().xmm(), guard_weight0.zmm());
            softmax_emitter_.emit_compute_weight_and_accumulate(*this, score1().xmm(), guard_weight1.zmm());
            softmax_emitter_.emit_compute_weight_and_accumulate(*this, score2().xmm(), guard_weight2.zmm());
            softmax_emitter_.emit_compute_weight_and_accumulate(*this, score3().xmm(), guard_weight3.zmm());

            v_accum_emitter_.emit_interleaved_v_accum_4x(
                *this,
                reg_V_ptr, v_stride,
                num_blocks,
                guard_weight0.xmm(), guard_weight1.xmm(), guard_weight2.xmm(), guard_weight3.xmm(),
                spill_base_offset);

            // GROUP 1: scores4..7 -> weights0..3 -> V accum (kv+4..kv+7)
            add(reg_V_ptr, 4 * v_stride);

            vmovss(score0().xmm(), ptr[rsp + fa2_scores_spill_offset + 16]);
            vmovss(score1().xmm(), ptr[rsp + fa2_scores_spill_offset + 20]);
            vmovss(score2().xmm(), ptr[rsp + fa2_scores_spill_offset + 24]);
            vmovss(score3().xmm(), ptr[rsp + fa2_scores_spill_offset + 28]);

            softmax_emitter_.emit_compute_weight_and_accumulate(*this, score0().xmm(), guard_weight0.zmm());
            softmax_emitter_.emit_compute_weight_and_accumulate(*this, score1().xmm(), guard_weight1.zmm());
            softmax_emitter_.emit_compute_weight_and_accumulate(*this, score2().xmm(), guard_weight2.zmm());
            softmax_emitter_.emit_compute_weight_and_accumulate(*this, score3().xmm(), guard_weight3.zmm());

            v_accum_emitter_.emit_interleaved_v_accum_4x(
                *this,
                reg_V_ptr, v_stride,
                num_blocks,
                guard_weight0.xmm(), guard_weight1.xmm(), guard_weight2.xmm(), guard_weight3.xmm(),
                spill_base_offset);
        }
    };

    /**
     * @brief Thread-safe cache for JIT-generated attention kernels
     *
     * Caches generated machine code indexed by JitAttentionConfig to avoid
     * regenerating the same kernel multiple times. Thread-safe via mutex.
     *
     * ═══════════════════════════════════════════════════════════════════════════
     * DESIGN
     * ═══════════════════════════════════════════════════════════════════════════
     *
     * - Singleton pattern (instance() returns static instance)
     * - Kernels are generated on first request for a given config
     * - Generated code persists for lifetime of the cache
     * - Config comparison uses all fields: head_dim, num_heads, wo_format, etc.
     *
     * ═══════════════════════════════════════════════════════════════════════════
     * THREAD SAFETY
     * ═══════════════════════════════════════════════════════════════════════════
     *
     * - getKernel() locks mutex before cache lookup/generation
     * - Multiple threads may safely request kernels concurrently
     * - Generated code is read-only after creation (safe to execute from threads)
     *
     * ═══════════════════════════════════════════════════════════════════════════
     * USAGE
     * ═══════════════════════════════════════════════════════════════════════════
     *
     *   JitAttentionConfig config;
     *   config.head_dim = 64;
     *   config.num_heads = 14;
     *   // ... set other fields ...
     *
     *   auto fn = JitAttentionKernelCache::instance().getKernel(config);
     *   fn(Q, K, V, Wo, output, seq_len_q, seq_len_kv, scale, 0);
     */
    class JitAttentionKernelCache
    {
    public:
        /**
         * @brief Get singleton instance
         * @return Reference to the global cache instance
         */
        static JitAttentionKernelCache &instance()
        {
            static JitAttentionKernelCache inst;
            return inst;
        }

        /**
         * @brief Get or generate kernel for config
         *
         * Thread-safe: locks mutex before accessing cache.
         * If kernel doesn't exist, generates and caches it.
         *
         * @param config Kernel configuration (determines code generation)
         * @return Function pointer to generated kernel (valid for cache lifetime)
         */
        JitAttentionKernelFn getKernel(const JitAttentionConfig &config)
        {
            std::lock_guard<std::mutex> lock(mutex_);

            auto it = cache_.find(config);
            if (it != cache_.end())
            {
                return it->second->getKernel();
            }

            // Generate new kernel on cache miss
            auto generator = std::make_unique<JitFusedAttentionWoGenerator>(config);
            auto fn = generator->getKernel();
            cache_[config] = std::move(generator);
            return fn;
        }

        /**
         * @brief Clear all cached kernels
         *
         * Releases all generated code. Useful for testing or memory reclamation.
         * After clear(), subsequent getKernel() calls will regenerate code.
         */
        void clear()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cache_.clear();
        }

        /**
         * @brief Get number of cached kernels
         * @return Number of unique configurations currently cached
         */
        size_t size() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return cache_.size();
        }

    private:
        JitAttentionKernelCache() = default;

        mutable std::mutex mutex_; ///< Protects cache_ for thread-safe access
        std::unordered_map<JitAttentionConfig, std::unique_ptr<JitFusedAttentionWoGenerator>> cache_;
    };

    /**
     * @brief High-level interface for JIT fused attention + Wo projection
     *
     * Wraps the JIT kernel cache to provide a simple object-oriented API.
     * Kernel is fetched from cache on construction (generated if needed).
     *
     * ═══════════════════════════════════════════════════════════════════════════
     * USAGE
     * ═══════════════════════════════════════════════════════════════════════════
     *
     *   // Create once (kernel generated/cached at construction time)
     *   JitAttentionConfig config;
     *   config.head_dim = 64;
     *   config.num_heads = 14;
     *   config.num_kv_heads = 2;
     *   config.wo_format = WoFormat::Q8_1;
     *
     *   JitFusedAttentionWo attn(config);
     *
     *   // Execute many times
     *   attn.compute(Q, K, V, Wo, output, seq_len_q, seq_len_kv, scale, 0);
     *
     * ═══════════════════════════════════════════════════════════════════════════
     * TENSOR LAYOUTS
     * ═══════════════════════════════════════════════════════════════════════════
     *
     *   Q: [seq_len_q, num_heads, head_dim] as Q8_1 blocks
     *   K: [seq_len_kv, num_kv_heads, head_dim] as Q8_1 blocks
     *   V: [seq_len_kv, num_kv_heads, head_dim] as Q8_1 blocks
     *   Wo: [d_model, d_model] in format specified by config.wo_format
     *   output: [seq_len_q, d_model] as FP32
     *
     * ═══════════════════════════════════════════════════════════════════════════
     * KERNEL MODES
     * ═══════════════════════════════════════════════════════════════════════════
     *
     *   DECODE (seq_len_q = 1):
     *     Single query attention, optimized for token-by-token generation.
     *     Uses register-resident Q for fast repeated K/V access.
     *
     *   PREFILL (seq_len_q > 1):
     *     Tiled attention for initial context processing.
     *     Processes Q in tiles of 4 queries at a time.
     */
    class JitFusedAttentionWo
    {
    public:
        /**
         * @brief Construct with given configuration
         *
         * Fetches or generates kernel from global cache.
         * Thread-safe: multiple JitFusedAttentionWo instances can be
         * created concurrently with the same or different configs.
         *
         * @param config Kernel configuration (head_dim, num_heads, wo_format, etc.)
         */
        explicit JitFusedAttentionWo(const JitAttentionConfig &config)
            : config_(config), kernel_(JitAttentionKernelCache::instance().getKernel(config))
        {
        }

        /**
         * @brief Execute fused attention + Wo projection
         *
         * Computes: output = softmax(Q @ K^T / scale) @ V @ Wo
         *
         * Automatically selects DECODE or PREFILL kernel based on seq_len_q.
         * Applies causal masking based on position_offset.
         *
         * @param Q          Q tensor data (Q8_1 format)
         * @param K          K tensor data (Q8_1 format)
         * @param V          V tensor data (Q8_1 format)
         * @param Wo         Wo weights (format per config.wo_format)
         * @param output     Output buffer (FP32, size = seq_len_q × d_model)
         * @param seq_len_q  Number of query positions (1 = decode, >1 = prefill)
         * @param seq_len_kv Number of KV positions (includes KV cache)
         * @param scale      Attention scale, typically 1/sqrt(head_dim)
         * @param position_offset Causal mask offset (0 for prefill, >0 for decode)
         */
        void compute(
            const void *Q,
            const void *K,
            const void *V,
            const void *Wo,
            float *output,
            int seq_len_q,
            int seq_len_kv,
            float scale,
            int position_offset = 0)
        {
            kernel_(Q, K, V, Wo, output, seq_len_q, seq_len_kv, scale, position_offset);
        }

        /**
         * @brief Get the underlying kernel function pointer
         */
        JitAttentionKernelFn getKernel() const { return kernel_; }

    private:
        JitAttentionConfig config_;
        JitAttentionKernelFn kernel_;
    };

} // namespace llaminar::v2::kernels::jit
