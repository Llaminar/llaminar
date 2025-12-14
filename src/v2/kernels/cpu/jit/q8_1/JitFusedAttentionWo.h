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
 * Architecture:
 *   1. For each query position q:
 *      a. For each KV position kv:
 *         - Score = Q8DotProduct(Q[q], K[kv]) * scale
 *         - OnlineSoftmax update (max, sum, correction)
 *         - Context[q] += softmax_weight * V[kv]
 *      b. Normalize context by 1/sum
 *      c. Output[q] = Context[q] * Wo
 *
 * JIT Strategy:
 *   - Generate code at runtime based on model dimensions
 *   - Specialize for head_dim, num_heads, batch_size
 *   - Cache generated code for reuse
 */

#pragma once

#include "JitMicrokernelBase.h"
#include "JitQ8DotProduct.h"
#include "JitOnlineSoftmax.h"
#include "JitVWeightedAccum.h"
#include "JitWoProjection.h"

#include <memory>
#include <unordered_map>
#include <mutex>
#include <functional>

namespace llaminar::v2::kernels::jit
{

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
                   effectiveMode() == other.effectiveMode();
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
            // Calculate optimal Q_TILE_SIZE for register blocking
            // We reserve zmm20-zmm31 for constants and scratch (zmm20-25 scratch, zmm26-31 consts).
            // We have 4 ZMM registers (zmm16-zmm19) available for Q data.
            // Each query requires num_blocks registers.
            // num_blocks = head_dim / 32.
            // Constraint: q_tile_size * num_blocks <= 4.
            int num_blocks = config_.head_dim / 32;
            int available_q_regs = 4;

            if (num_blocks > 0)
            {
                q_tile_size_ = available_q_regs / num_blocks;
            }
            else
            {
                q_tile_size_ = 8;
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
         * Loop order: Q → H → KV
         * Optimized for single-token generation with minimal memory overhead.
         */
        void generate_decode_kernel()
        {
            using namespace Xbyak;

            // Function prologue
            // Calling convention: System V AMD64
            // Integer args: rdi = Q, rsi = K, rdx = V, rcx = Wo, r8 = output, r9 = seq_len_q
            // Stack: [rsp+8] = seq_len_kv (7th integer arg), [rsp+16] = position_offset (8th integer arg)
            // Float args: xmm0 = scale (1st float arg)

            push_callee_saved();

            // Save parameters to callee-saved registers
            Reg64 reg_Q = r12;
            Reg64 reg_K = r13;
            Reg64 reg_V = rbx;
            Reg64 reg_Wo = rbp;
            Reg64 reg_output = r14;
            Reg64 reg_seq_len_q = r15;

            mov(reg_Q, rdi);
            mov(reg_K, rsi);
            mov(reg_V, rdx);
            mov(reg_Wo, rcx);
            mov(reg_output, r8);
            mov(reg_seq_len_q, r9);

            // Load stack parameters
            // Note: r10 is caller-saved and may be clobbered during emit_wo_projection,
            // so we'll save it to our own stack frame
            Reg64 reg_seq_len_kv = r10;
            mov(reg_seq_len_kv, ptr[rsp + stack_frame_size() + 8]); // 7th integer arg

            // Load scale from xmm0 (1st float argument in System V AMD64)
            // xmm0 already contains scale, broadcast to zmm_scale()
            vbroadcastss(zmm_scale(), xmm0);

            // Initialize constants for exp
            load_constant_f32(zmm_log2e(), 1.44269504089f);
            load_constant_f32(zmm_exp_min(), -87.0f);
            load_constant_f32(zmm_one(), 1.0f);

            // Initialize constant for Q8_1 unsigned conversion (0x80808080)
            emit_broadcast_i32_const(zmm_128(), 0x80808080, rax);

            // Calculate working space needed on stack:
            // 1. Q blocks for one head: num_blocks * 64 bytes (padded)
            // 2. Context accumulator spill: (num_blocks - 2) * 128 bytes
            // 3. Context buffer for Wo projection: num_heads * head_dim * 4 bytes
            // 4. q_idx spill slot: 8 bytes
            // 5. seq_len_kv spill slot: 8 bytes
            // 6. position_offset spill slot: 8 bytes
            // 7. Extra for alignment and temps
            int num_blocks = config_.head_dim / 32;
            int d_model = config_.num_heads * config_.head_dim;
            int q_stack_size = num_blocks * 64; // Padded Q blocks for one head
            int spill_bytes = (num_blocks > 2) ? (num_blocks - 2) * 128 : 0;
            int context_buffer_size = d_model * 4;                                   // FP32 context for all heads
            int stack_size = q_stack_size + spill_bytes + context_buffer_size + 256; // Extra for alignment
            stack_size = (stack_size + 63) & ~63;                                    // Align to 64 bytes

            // Stack layout:
            // [rsp + 0]                     : Q blocks for current head (q_stack_size bytes)
            // [rsp + q_stack_size]          : Spill area for context accumulators
            // [rsp + q_stack_size + spill]  : Context buffer for Wo projection (d_model * 4 bytes)
            // [rsp + context_end]           : q_idx spill (8 bytes)
            // [rsp + context_end + 8]       : seq_len_kv spill (8 bytes)
            // [rsp + context_end + 16]      : position_offset spill (8 bytes)
            // [rsp + ... ]                  : Temp/alignment padding
            int q_stack_offset = 0;
            int spill_base_offset = q_stack_size;
            int context_buffer_offset = q_stack_size + spill_bytes;
            int q_idx_spill_offset = context_buffer_offset + d_model * 4;
            int seq_len_kv_spill_offset = q_idx_spill_offset + 8;
            int position_offset_spill_offset = seq_len_kv_spill_offset + 8;

            sub(rsp, stack_size);

            // Save seq_len_kv to stack (it gets clobbered by emit_wo_projection)
            mov(ptr[rsp + seq_len_kv_spill_offset], reg_seq_len_kv);

            // Load and save position_offset to stack
            // position_offset is 8th integer arg, at [old_rsp + 16] = [rsp + stack_size + stack_frame_size() + 16]
            Reg64 reg_tmp = rdi; // Temporarily reuse rdi
            mov(reg_tmp, ptr[rsp + stack_size + stack_frame_size() + 16]);
            mov(ptr[rsp + position_offset_spill_offset], reg_tmp);

            // Main loop over query positions
            Reg64 reg_q_idx = rax;
            xor_(reg_q_idx, reg_q_idx);

            L("main_loop_q");
            cmp(reg_q_idx, reg_seq_len_q);
            jge("main_loop_q_end", T_NEAR);

            // Restore reg_seq_len_kv from stack (it may have been clobbered in previous iteration)
            mov(reg_seq_len_kv, ptr[rsp + seq_len_kv_spill_offset]);

            // Emit attention for one query position (includes Wo projection)
            emit_single_query_attention(
                reg_Q, reg_K, reg_V, reg_Wo, reg_output,
                reg_q_idx, reg_seq_len_kv,
                num_blocks, spill_base_offset, q_stack_offset, context_buffer_offset,
                q_idx_spill_offset, position_offset_spill_offset);

            inc(reg_q_idx);
            jmp("main_loop_q", T_NEAR);

            L("main_loop_q_end");

            // Cleanup
            add(rsp, stack_size);
            pop_callee_saved();
            ret();

            ready();
        }

        /**
         * @brief Generate prefill-optimized kernel (seq_len_q > 1)
         *
         * Loop order: Q_tile → H → KV → Q_inner
         *
         * This processes queries in tiles to bound stack usage. Within each tile,
         * we iterate over heads, then for each head we iterate over KV positions
         * and for each KV position we process all queries in the tile that pass
         * the causal mask.
         *
         * This achieves K/V cache reuse: each K/V block is loaded once per head
         * and reused across all queries in the tile.
         *
         * Register allocation strategy:
         * - Callee-saved (r12-r15, rbx, rbp): Input pointers, seq_len_q
         * - Stack spills: All loop indices and values that must survive emitter calls
         * - Caller-saved: Freely used within each emitter, reloaded from spills as needed
         *
         * Stack layout (Q_TILE_SIZE = 8):
         * [0]: Q blocks for tile (Q_TILE_SIZE * num_blocks * 64 bytes)
         * [+q_blocks]: Softmax state (Q_TILE_SIZE * 8 bytes: max,sum pairs)
         * [+softmax]: Context buffer (Q_TILE_SIZE * d_model * 4 bytes)
         * [+context]: Spill slots for loop variables
         */
        void generate_prefill_kernel()
        {
            using namespace Xbyak;

            // Function prologue
            push_callee_saved();

            // Callee-saved register allocation (persist across all emitter calls)
            Reg64 reg_Q = r12;
            Reg64 reg_K = r13;
            Reg64 reg_V = rbx;
            Reg64 reg_Wo = rbp;
            Reg64 reg_output = r14;
            Reg64 reg_seq_len_q = r15;

            mov(reg_Q, rdi);
            mov(reg_K, rsi);
            mov(reg_V, rdx);
            mov(reg_Wo, rcx);
            mov(reg_output, r8);
            mov(reg_seq_len_q, r9);

            // Broadcast scale
            vbroadcastss(zmm_scale(), xmm0);

            // Initialize constants for exp
            load_constant_f32(zmm_log2e(), 1.44269504089f);
            load_constant_f32(zmm_one(), 1.0f);
            load_constant_f32(zmm_exp_min(), -87.0f);

            // Initialize constant for unsigned Q conversion
            emit_broadcast_i32_const(zmm_128(), 0x80808080, rax);

            // Calculate dimensions
            int num_blocks = config_.head_dim / 32;
            int d_model = config_.num_heads * config_.head_dim;
            int head_dim = config_.head_dim;
            int heads_per_kv = config_.num_heads / config_.num_kv_heads;

            // Stack layout calculation
            int q_blocks_size = q_tile_size_ * num_blocks * 64;
            int softmax_state_size = q_tile_size_ * config_.num_heads * 8; // (max, sum) per query per head
            int context_size = q_tile_size_ * d_model * 4;                 // FP32 context per query

            int q_blocks_offset = 0;
            int softmax_offset = q_blocks_size;
            int context_offset = softmax_offset + softmax_state_size;
            int spill_vars_offset = context_offset + context_size;

            // Spill slots
            int tile_start_spill = spill_vars_offset;
            int tile_size_spill = tile_start_spill + 8;
            int seq_len_kv_spill = tile_size_spill + 8;
            int position_offset_spill = seq_len_kv_spill + 8;
            int kv_idx_spill = position_offset_spill + 8;
            int q_local_spill = kv_idx_spill + 8;
            int k_ptr_spill = q_local_spill + 8;
            int v_ptr_spill = k_ptr_spill + 8;

            // Debug: print spill offsets
            printf("DEBUG spill offsets: tile_start=%d, tile_size=%d, seq_len_kv=%d, position_offset=%d, kv_idx=%d\n",
                   tile_start_spill, tile_size_spill, seq_len_kv_spill, position_offset_spill, kv_idx_spill);

            // New spill slots for assembly head loop
            int head_idx_spill = v_ptr_spill + 8;
            int head_offset_spill = head_idx_spill + 8;
            int softmax_head_offset_spill = head_offset_spill + 8;
            int context_head_offset_spill = softmax_head_offset_spill + 8;
            int q_loop_spill = context_head_offset_spill + 8;

            int stack_size = q_loop_spill + 8 + 64;
            stack_size = (stack_size + 63) & ~63;

            sub(rsp, stack_size);

            // Load and spill seq_len_kv (7th arg)
            mov(rax, ptr[rsp + stack_size + stack_frame_size() + 8]);
            mov(ptr[rsp + seq_len_kv_spill], rax);

            // Load and spill position_offset (8th arg)
            mov(rax, ptr[rsp + stack_size + stack_frame_size() + 16]);
            mov(ptr[rsp + position_offset_spill], rax);

            // Initialize zmm_128 constant for Q8_1 unsigned conversion
            emit_broadcast_i32_const(zmm_128(), 0x80808080, rax);

            // ===============================================================
            // TILE LOOP: Process queries in tiles of Q_TILE_SIZE
            // ===============================================================
            mov(qword[rsp + tile_start_spill], 0);

            L("prefill_tile_loop");

            mov(rax, ptr[rsp + tile_start_spill]);
            cmp(rax, reg_seq_len_q);
            jge("prefill_tile_end", T_NEAR);

            // Calculate tile_size = min(q_tile_size_, seq_len_q - tile_start)
            mov(rcx, reg_seq_len_q);
            sub(rcx, rax);
            cmp(rcx, q_tile_size_);
            jle("prefill_tile_size_ok", T_NEAR);
            mov(rcx, q_tile_size_);
            L("prefill_tile_size_ok");
            mov(ptr[rsp + tile_size_spill], rcx);

            // Initialize softmax state: max=-inf, sum=0
            // Initialize for ALL heads in the tile
            emit_prefill_init_softmax(q_tile_size_ * config_.num_heads, softmax_offset);

            // Initialize context buffer to zeros
            emit_prefill_init_context(q_tile_size_, d_model, context_offset);

            // ===============================================================
            // HEAD LOOP: Process each attention head (Assembly Loop)
            // ===============================================================

            // Initialize loop variables
            mov(qword[rsp + head_idx_spill], 0);
            mov(qword[rsp + head_offset_spill], 0);
            mov(rax, softmax_offset);
            mov(ptr[rsp + softmax_head_offset_spill], rax);
            mov(rax, context_offset);
            mov(ptr[rsp + context_head_offset_spill], rax);

            L("prefill_head_loop");
            mov(rax, ptr[rsp + head_idx_spill]);
            cmp(rax, config_.num_heads);
            jge("prefill_head_end", T_NEAR);

            // Copy Q blocks for all queries in tile for this head
            mov(r11, ptr[rsp + head_offset_spill]);
            emit_prefill_copy_q_tile(reg_Q, r11, num_blocks, q_blocks_offset,
                                     tile_start_spill, tile_size_spill);

            // ===============================================================
            // KV LOOP: Iterate over K/V cache positions (runtime)
            // ===============================================================
            mov(qword[rsp + kv_idx_spill], 0);

            L(".kv_loop");
            mov(r8, ptr[rsp + kv_idx_spill]);
            mov(r9, ptr[rsp + seq_len_kv_spill]);
            cmp(r8, r9);
            jge(".kv_end", T_NEAR);

            // Calculate kv_head = head_idx / heads_per_kv
            xor_(rdx, rdx);
            mov(rax, ptr[rsp + head_idx_spill]);
            mov(rcx, heads_per_kv);
            div(rcx); // rax = kv_head

            // Calculate K/V pointers
            int kv_stride = config_.num_kv_heads * num_blocks * 36;
            int kv_head_stride = num_blocks * 36;

            // K_ptr = reg_K + kv_idx * kv_stride + kv_head * kv_head_stride
            mov(rdi, ptr[rsp + kv_idx_spill]);
            imul(rdi, rdi, kv_stride);

            mov(rsi, rax); // kv_head
            imul(rsi, rsi, kv_head_stride);
            add(rdi, rsi);

            mov(rsi, rdi); // Copy offset for V

            add(rdi, reg_K);
            mov(ptr[rsp + k_ptr_spill], rdi);

            add(rsi, reg_V);
            mov(ptr[rsp + v_ptr_spill], rsi);

            // ===============================================================
            // Q INNER LOOP: Process queries within tile that pass causal mask
            // ===============================================================
            // For causal: q_start = max(0, kv_idx - position_offset)
            mov(r10, ptr[rsp + kv_idx_spill]);
            if (config_.causal)
            {
                sub(r10, ptr[rsp + position_offset_spill]);
                xor_(r11, r11);
                cmp(r10, 0);
                cmovl(r10, r11);
            }
            else
            {
                xor_(r10, r10);
            }

            // r10 now holds q_start - save it for debug
            mov(ptr[rsp + q_local_spill], r10); // Temporarily save q_start

            // Now calculate q_local_start = max(0, q_start - tile_start)
            mov(r10, ptr[rsp + q_local_spill]); // Reload q_start
            mov(rax, ptr[rsp + tile_start_spill]);
            sub(r10, rax);
            xor_(r11, r11);
            cmp(r10, 0);
            cmovl(r10, r11);
            mov(ptr[rsp + q_local_spill], r10); // Now store final q_local_start

            // Process attention for all valid queries in tile
            // Reload K/V pointers from spill (emitters may clobber registers)
            mov(rdi, ptr[rsp + k_ptr_spill]);
            mov(rsi, ptr[rsp + v_ptr_spill]);

            // Load dynamic offsets
            mov(rdx, ptr[rsp + softmax_head_offset_spill]);
            mov(rcx, ptr[rsp + context_head_offset_spill]);

            emit_prefill_tile_attention(
                rdi, rsi, r10,              // K_ptr, V_ptr, q_local_start
                ptr[rsp + tile_size_spill], // tile_size
                num_blocks,
                q_blocks_offset,
                d_model,
                rdx, rcx);

            // Increment kv_idx
            mov(r8, ptr[rsp + kv_idx_spill]);
            inc(r8);
            mov(ptr[rsp + kv_idx_spill], r8);
            jmp(".kv_loop", T_NEAR);

            L(".kv_end");

            // Increment offsets
            mov(rax, ptr[rsp + head_offset_spill]);
            add(rax, num_blocks * 36);
            mov(ptr[rsp + head_offset_spill], rax);

            mov(rax, ptr[rsp + softmax_head_offset_spill]);
            add(rax, q_tile_size_ * 8);
            mov(ptr[rsp + softmax_head_offset_spill], rax);

            mov(rax, ptr[rsp + context_head_offset_spill]);
            add(rax, head_dim * 4);
            mov(ptr[rsp + context_head_offset_spill], rax);

            // Increment head_idx
            mov(rax, ptr[rsp + head_idx_spill]);
            inc(rax);
            mov(ptr[rsp + head_idx_spill], rax);

            jmp("prefill_head_loop", T_NEAR);

            L("prefill_head_end");

            // ===============================================================
            // NORMALIZE AND PROJECT: 1/sum normalization + Wo projection
            // ===============================================================
            emit_prefill_normalize_project(
                reg_Wo, reg_output,
                num_blocks, q_tile_size_,
                softmax_offset, context_offset,
                tile_start_spill, tile_size_spill, d_model, q_local_spill, q_loop_spill);

            // Advance tile_start
            mov(rax, ptr[rsp + tile_start_spill]);
            add(rax, q_tile_size_);
            mov(ptr[rsp + tile_start_spill], rax);
            jmp("prefill_tile_loop", T_NEAR);

            L("prefill_tile_end");

            // Epilogue
            add(rsp, stack_size);
            pop_callee_saved();
            ret();

            ready();
        }

        // =================================================================
        // Prefill helper methods
        // =================================================================

        /**
         * @brief Initialize softmax state for prefill tile
         */
        void emit_prefill_init_softmax(int tile_size, int softmax_offset)
        {
            using namespace Xbyak;

            // max = -FLT_MAX, sum = 0 for each query
            Zmm zmm_neg_inf = zmm_scratch(0);
            load_constant_f32(zmm_neg_inf, -3.4028235e+38f);

            Zmm zmm_zero = zmm_scratch(1);
            vxorps(zmm_zero, zmm_zero, zmm_zero);

            for (int q = 0; q < tile_size; ++q)
            {
                int offset = softmax_offset + q * 8;
                vmovss(ptr[rsp + offset], Xmm(zmm_neg_inf.getIdx()));
                vmovss(ptr[rsp + offset + 4], Xmm(zmm_zero.getIdx()));
            }
        }

        /**
         * @brief Initialize context buffer for prefill tile
         */
        void emit_prefill_init_context(int tile_size, int d_model, int context_offset)
        {
            using namespace Xbyak;

            Zmm zmm_zero = zmm_scratch(0);
            vxorps(zmm_zero, zmm_zero, zmm_zero);

            int total_floats = tile_size * d_model;
            // DEBUG: Reduce loop limit to check for overflow
            for (int i = 0; i < total_floats - 16; i += 16)
            {
                vmovups(ptr[rsp + context_offset + i * 4], zmm_zero);
            }
        }

        /**
         * @brief Vectorized exponential for ZMM
         * Computes exp(x) for each element in src
         */
        void emit_vec_exp(const Xbyak::Zmm &dst, const Xbyak::Zmm &src)
        {
            using namespace Xbyak;

            // Constants
            Xmm xmm_log2e = Xmm(zmm_log2e().getIdx());

            Zmm zmm_n = zmm_scratch(0);
            Zmm zmm_f = zmm_scratch(1);
            Zmm zmm_p = zmm_scratch(2);

            // x * log2e
            vmulps(zmm_f, src, zmm_log2e());

            // n = floor(x * log2e)
            vrndscaleps(zmm_n, zmm_f, 0x01); // Round down

            // f = x * log2e - n
            vsubps(zmm_f, zmm_f, zmm_n);

            // Polynomial p = c5 * f + c4 ...
            // c5 = 0.0013334f (0x3aaec3b4)
            mov(eax, 0x3aaec3b4);
            vpbroadcastd(zmm_p, eax);
            vmulps(zmm_p, zmm_p, zmm_f);

            // c4 = 0.00961812f (0x3c1d955b)
            mov(eax, 0x3c1d955b);
            vpbroadcastd(dst, eax); // use dst as temp
            vaddps(zmm_p, zmm_p, dst);
            vmulps(zmm_p, zmm_p, zmm_f);

            // c3 = 0.05550411f (0x3d6356eb)
            mov(eax, 0x3d6356eb);
            vpbroadcastd(dst, eax);
            vaddps(zmm_p, zmm_p, dst);
            vmulps(zmm_p, zmm_p, zmm_f);

            // c2 = 0.24022650f (0x3e75fdf0)
            mov(eax, 0x3e75fdf0);
            vpbroadcastd(dst, eax);
            vaddps(zmm_p, zmm_p, dst);
            vmulps(zmm_p, zmm_p, zmm_f);

            // c1 = 0.69314718f (0x3f317218)
            mov(eax, 0x3f317218);
            vpbroadcastd(dst, eax);
            vaddps(zmm_p, zmm_p, dst);
            vmulps(zmm_p, zmm_p, zmm_f);

            // c0 = 1.0f (0x3f800000)
            mov(eax, 0x3f800000);
            vpbroadcastd(dst, eax);
            vaddps(zmm_p, zmm_p, dst);

            // Final scaling: p * 2^n
            vscalefps(dst, zmm_p, zmm_n);
        }

        /**
         * @brief Process attention for all valid queries in prefill tile
         *
         * Optimized to load K/V once and reuse across all queries in the tile.
         * Uses register blocking for Q data if it fits (num_blocks <= 2).
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

            // CRITICAL: Use local labels to avoid collisions across multiple calls
            // This function is called in a KV loop, so labels must be scoped!
            inLocalLabel();

            // 1. Compute Scores (Q * K^T)
            // Accumulators: zmm0..zmm(q_tile_size_-1)
            // Initialize accumulators to 0
            for (int q = 0; q < q_tile_size_; ++q)
            {
                vxorps(Zmm(q), Zmm(q), Zmm(q));
            }

            // Optimization: Pre-load Q data into registers if it fits
            // We have zmm16-31 available (16 registers).
            // Each query needs num_blocks registers for data.
            // bool use_q_registers = (q_tile_size_ * num_blocks <= 16);
            bool use_q_registers = false; // Disabled due to conflict with JitMicrokernelBase register allocation

            if (use_q_registers)
            {
                for (int b = 0; b < num_blocks; ++b)
                {
                    int q_off_base = q_blocks_offset + b * 64;
                    for (int q = 0; q < q_tile_size_; ++q)
                    {
                        int q_off = q_off_base + q * num_blocks * 64;
                        int reg_idx = 16 + b * q_tile_size_ + q;
                        vmovdqu8(Ymm(reg_idx), ptr[rsp + q_off + 4]);
                        vpxord(Ymm(reg_idx), Ymm(reg_idx), Ymm(zmm_128().getIdx())); // unsigned to signed
                    }
                }
            }

            // Loop over blocks
            for (int b = 0; b < num_blocks; ++b)
            {
                int k_off = b * 36;
                int q_off_base = q_blocks_offset + b * 64; // Base offset for block b of Q0

                // Use scratch registers to avoid clobbering accumulators (zmm0-15)
                // K scale: zmm20
                // K sum_qs: zmm21
                // K data: zmm22
                Xmm xmm_d_k(20);
                Ymm ymm_sum_qs_k(21);     // Use YMM for 8 copies
                Xmm xmm_sum_qs_k_low(21); // Access low 128 bits
                Ymm ymm_k(22);

                vpbroadcastw(xmm_d_k, ptr[reg_K_ptr + k_off]);
                vcvtph2ps(xmm_d_k, xmm_d_k); // d_k

                vpbroadcastw(xmm_sum_qs_k_low, ptr[reg_K_ptr + k_off + 2]);
                vpmovsxwd(ymm_sum_qs_k, xmm_sum_qs_k_low); // Convert 8 words to 8 dwords
                vcvtdq2ps(ymm_sum_qs_k, ymm_sum_qs_k);     // sum_qs_k (8 floats)

                vmovdqu8(ymm_k, ptr[reg_K_ptr + k_off + 4]); // K data

                // Loop over queries in tile (unrolled)
                for (int q = 0; q < q_tile_size_; ++q)
                {
                    int q_off = q_off_base + q * num_blocks * 64;

                    // d_q: zmm23
                    Xmm xmm_d_q(23);
                    vpbroadcastw(xmm_d_q, ptr[rsp + q_off]);
                    vcvtph2ps(xmm_d_q, xmm_d_q); // d_q

                    // Dot product: zmm25
                    Ymm ymm_dot(25);
                    vxorps(ymm_dot, ymm_dot, ymm_dot);

                    if (use_q_registers)
                    {
                        int reg_idx = 16 + b * q_tile_size_ + q;
                        vpdpbusd(ymm_dot, Ymm(reg_idx), ymm_k);
                    }
                    else
                    {
                        // Q data: zmm24
                        Ymm ymm_q(24);
                        vmovdqu8(ymm_q, ptr[rsp + q_off + 4]);
                        vpxord(ymm_q, ymm_q, Ymm(zmm_128().getIdx())); // unsigned to signed
                        vpdpbusd(ymm_dot, ymm_q, ymm_k);
                    }

                    vcvtdq2ps(ymm_dot, ymm_dot);

                    // Scale: d_q * d_k -> zmm23 (reuse d_q)
                    Xmm xmm_scale(23);
                    Ymm ymm_scale(23);
                    vmulps(xmm_scale, xmm_d_q, xmm_d_k);
                    vbroadcastss(ymm_scale, xmm_scale); // Broadcast to YMM
                    vmulps(ymm_dot, ymm_dot, ymm_scale);

                    // Correction: 16 * sum_qs_k * scale -> zmm24 (reuse Q data)
                    // 16.0f = 0x41800000 (distributed correction for 8 lanes)
                    Xmm xmm_corr_low(24);
                    Ymm ymm_corr(24);
                    mov(eax, 0x41800000); // 16.0f
                    vmovd(xmm_corr_low, eax);
                    vbroadcastss(ymm_corr, xmm_corr_low); // Broadcast to YMM
                    vmulps(ymm_corr, ymm_corr, ymm_sum_qs_k);
                    vmulps(ymm_corr, ymm_corr, ymm_scale);

                    vsubps(ymm_dot, ymm_dot, ymm_corr);

                    // Accumulate
                    vaddps(Zmm(q), Zmm(q), Zmm(ymm_dot.getIdx()));
                }
            }

            // Horizontal sum and scale scores
            vmovss(xmm8, Xmm(zmm_scale().getIdx())); // scale
            for (int q = 0; q < q_tile_size_; ++q)
            {
                emit_horizontal_sum_to_scalar(Zmm(q));
                vmulss(Xmm(Zmm(q).getIdx()), Xmm(Zmm(q).getIdx()), xmm8);
            }

            // 2. Update Softmax (Scalar Loop - Optimized for sparsity)
            mov(r8, op_tile_size);

            for (int q = 0; q < q_tile_size_; ++q)
            {
                std::string skip_label = ".skip_softmax_" + std::to_string(q);

                cmp(reg_q_local_start, q);
                jg(skip_label, T_NEAR); // q_local_start > q

                cmp(r8, q);
                jle(skip_label, T_NEAR); // tile_size <= q

                // Valid query
                mov(r9, q);
                shl(r9, 3);
                add(r9, reg_softmax_head_offset);
                add(r9, rsp);

                Xmm xmm_score = Xmm(Zmm(q).getIdx());
                Xmm xmm_max = xmm16;
                Xmm xmm_sum = xmm17;

                vmovss(xmm_max, ptr[r9]);
                vmovss(xmm_sum, ptr[r9 + 4]);

                Xmm xmm_new_max = xmm18;
                vmaxss(xmm_new_max, xmm_score, xmm_max);

                // corr = exp(old_max - new_max)
                Xmm xmm_corr = xmm19;
                vsubss(xmm_corr, xmm_max, xmm_new_max);
                emit_prefill_exp_scalar(xmm_corr, xmm_corr);

                // weight = exp(score - new_max)
                Xmm xmm_weight = xmm20;
                vsubss(xmm_weight, xmm_score, xmm_new_max);
                emit_prefill_exp_scalar(xmm_weight, xmm_weight);

                // new_sum = old_sum * corr + weight
                vmulss(xmm_sum, xmm_sum, xmm_corr);
                vaddss(xmm_sum, xmm_sum, xmm_weight);

                // Store updated state
                vmovss(ptr[r9], xmm_new_max);
                vmovss(ptr[r9 + 4], xmm_sum);

                // Save corr/weight for V accumulation
                vbroadcastss(Zmm(q), xmm_corr);                  // zmm0..7 = corr
                vbroadcastss(Zmm(q + q_tile_size_), xmm_weight); // zmm8..15 = weight

                jmp(".done_softmax_" + std::to_string(q), T_NEAR);

                L(skip_label);
                // Zero out corr/weight for invalid queries (safety)
                // corr = 1.0 (preserve existing context), weight = 0.0 (no new contribution)
                vmovups(Zmm(q), zmm_one());
                vxorps(Zmm(q + q_tile_size_), Zmm(q + q_tile_size_), Zmm(q + q_tile_size_));

                L(".done_softmax_" + std::to_string(q));
            }

            // 3. Accumulate V
            for (int b = 0; b < num_blocks; ++b)
            {
                int v_off = b * 36;
                int ctx_off_base = b * 32 * 4;

                // Load V block
                vpbroadcastw(xmm16, ptr[reg_V_ptr + v_off]);
                vcvtph2ps(xmm16, xmm16);
                vbroadcastss(zmm16, xmm16); // V scale

                vpmovsxbd(zmm17, ptr[reg_V_ptr + v_off + 4]);
                vpmovsxbd(zmm18, ptr[reg_V_ptr + v_off + 4 + 16]);
                vcvtdq2ps(zmm17, zmm17);
                vcvtdq2ps(zmm18, zmm18);

                // Loop over queries
                for (int q = 0; q < q_tile_size_; ++q)
                {
                    std::string skip_v = ".skip_v_" + std::to_string(b) + "_" + std::to_string(q);

                    cmp(reg_q_local_start, q);
                    jg(skip_v, T_NEAR);
                    cmp(r8, q);
                    jle(skip_v, T_NEAR);

                    // Valid query
                    mov(r9, q);
                    imul(r9, r9, d_model * 4);
                    add(r9, reg_context_head_offset);
                    add(r9, rsp);

                    // Load Context
                    Zmm zmm_ctx_lo = zmm19;
                    Zmm zmm_ctx_hi = zmm20;

                    vmovups(zmm_ctx_lo, ptr[r9 + ctx_off_base]);
                    vmovups(zmm_ctx_hi, ptr[r9 + ctx_off_base + 64]);

                    // weighted_v = V * scale * weight[q]
                    Zmm zmm_wv_lo = zmm21;
                    Zmm zmm_wv_hi = zmm22;

                    vmulps(zmm_wv_lo, zmm17, zmm16);                     // V_lo * scale
                    vmulps(zmm_wv_lo, zmm_wv_lo, Zmm(q + q_tile_size_)); // * weight[q]

                    vmulps(zmm_wv_hi, zmm18, zmm16);                     // V_hi * scale
                    vmulps(zmm_wv_hi, zmm_wv_hi, Zmm(q + q_tile_size_)); // * weight[q]

                    // Update context
                    vmulps(zmm_ctx_lo, zmm_ctx_lo, Zmm(q)); // * corr[q]
                    vaddps(zmm_ctx_lo, zmm_ctx_lo, zmm_wv_lo);

                    vmulps(zmm_ctx_hi, zmm_ctx_hi, Zmm(q)); // * corr[q]
                    vaddps(zmm_ctx_hi, zmm_ctx_hi, zmm_wv_hi);

                    // Store
                    vmovups(ptr[r9 + ctx_off_base], zmm_ctx_lo);
                    vmovups(ptr[r9 + ctx_off_base + 64], zmm_ctx_hi);

                    L(skip_v);
                }
            }

            outLocalLabel();
        }

        /**
         * @brief Copy Q blocks for tile for one head (dynamic head offset)
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

            int q_stride = config_.num_heads * num_blocks * 36;
            // head_offset is passed in register

            inLocalLabel();

            xor_(r8, r8); // q_local = 0

            L(".copy_loop");
            cmp(r8, ptr[rsp + tile_size_spill]);
            jge(".copy_end", T_NEAR);

            // Q source: reg_Q + (tile_start + q_local) * q_stride + head_offset
            mov(r9, ptr[rsp + tile_start_spill]);
            add(r9, r8);
            imul(r9, r9, q_stride);
            add(r9, reg_head_offset); // Add dynamic head offset
            add(r9, reg_Q);

            // Q dest on stack: q_blocks_offset + q_local * num_blocks * 64
            mov(r10, r8);
            imul(r10, r10, num_blocks * 64);
            add(r10, q_blocks_offset);

            // Copy blocks
            for (int b = 0; b < num_blocks; ++b)
            {
                int src_b = b * 36;
                int dst_b = b * 64;

                // Copy scale+sum_qs (4 bytes)
                mov(eax, ptr[r9 + src_b]);
                mov(ptr[rsp + r10 + dst_b], eax);

                // Copy data (32 bytes)
                Ymm ymm_tmp = Ymm(zmm_scratch(0).getIdx());
                vmovdqu8(ymm_tmp, ptr[r9 + src_b + 4]);

                vmovdqu8(ptr[rsp + r10 + dst_b + 4], ymm_tmp);
            }

            inc(r8);
            jmp(".copy_loop", T_NEAR);

            L(".copy_end");

            outLocalLabel();
        }

        /**
         * @brief Process attention for one query in prefill tile (dynamic offsets)
         *
         * K/V pointers are passed in registers, q_local indicates which query.
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
            // head_ctx_offset is passed in reg_context_head_offset

            // Calculate Q base on stack: rsp + q_blocks_offset + q_local * num_blocks * 64
            Reg64 reg_q_base = r11;
            mov(reg_q_base, reg_q_local);
            imul(reg_q_base, reg_q_base, num_blocks * 64);
            add(reg_q_base, q_blocks_offset);
            add(reg_q_base, rsp);

            // Q·K dot product
            Xmm xmm_score = Xmm(zmm_scratch(0).getIdx());
            Xmm xmm_scale_tmp = Xmm(zmm_scratch(1).getIdx());
            vmovss(xmm_scale_tmp, Xmm(zmm_scale().getIdx()));

            emit_prefill_dot_product(xmm_score, reg_K_ptr, reg_q_base, num_blocks, xmm_scale_tmp);

            // Load softmax state
            Reg64 reg_sm = r11; // Use r11 (reused later for reg_ctx)
            mov(reg_sm, reg_q_local);
            shl(reg_sm, 3);
            add(reg_sm, reg_softmax_head_offset); // Use dynamic offset
            add(reg_sm, rsp);

            // Use StateRegs for max/sum to avoid clobbering constants
            Xmm xmm_max = Xmm(zmm_max().getIdx()); // zmm16
            Xmm xmm_sum = Xmm(zmm_sum().getIdx()); // zmm17
            vmovss(xmm_max, ptr[reg_sm]);
            vmovss(xmm_sum, ptr[reg_sm + 4]);

            // Online softmax update
            Xmm xmm_new_max = Xmm(zmm_scratch(2).getIdx()); // zmm22
            Xmm xmm_corr = Xmm(zmm_corr().getIdx());        // zmm19
            Xmm xmm_weight = Xmm(zmm_weight().getIdx());    // zmm18

            vmaxss(xmm_new_max, xmm_score, xmm_max);

            // corr = exp(old_max - new_max)
            vsubss(xmm_corr, xmm_max, xmm_new_max);
            emit_prefill_exp_scalar(xmm_corr, xmm_corr);

            // weight = exp(score - new_max)
            vsubss(xmm_weight, xmm_score, xmm_new_max);
            emit_prefill_exp_scalar(xmm_weight, xmm_weight);

            // new_sum = old_sum * corr + weight
            vmulss(xmm_sum, xmm_sum, xmm_corr);
            vaddss(xmm_sum, xmm_sum, xmm_weight);

            // Store updated state
            vmovss(ptr[reg_sm], xmm_new_max);
            vmovss(ptr[reg_sm + 4], xmm_sum);

            // Broadcast for V accumulation
            vbroadcastss(zmm_weight(), xmm_weight);
            vbroadcastss(zmm_corr(), xmm_corr);

            // Calculate context pointer
            Reg64 reg_ctx = r11;
            mov(reg_ctx, reg_q_local);
            imul(reg_ctx, reg_ctx, d_model * 4);
            add(reg_ctx, reg_context_head_offset); // Use dynamic offset
            add(reg_ctx, rsp);

            // Accumulate weighted V
            emit_prefill_v_accum(reg_V_ptr, reg_ctx, num_blocks);
        }

        /**
         * @brief Q8_1 dot product for prefill (inline version)
         * Optimized to accumulate across blocks before horizontal sum.
         */
        void emit_prefill_dot_product(
            const Xbyak::Xmm &dst,
            const Xbyak::Reg64 &reg_K,
            const Xbyak::Reg64 &reg_Q,
            int num_blocks,
            const Xbyak::Xmm &scale)
        {
            using namespace Xbyak;

            // Use ymm7 as accumulator for scaled floats
            Ymm ymm_acc(7);
            vxorps(ymm_acc, ymm_acc, ymm_acc);

            Ymm ymm_q(4);
            Ymm ymm_k(5);
            Ymm ymm_dot(6);
            Xmm xmm_d_q(8);
            Xmm xmm_d_k(9);
            Xmm xmm_sum_qs_k(15);
            Xmm xmm_correction(14);
            Xmm xmm_scale_block(10); // d_q * d_k

            for (int b = 0; b < num_blocks; ++b)
            {
                int q_off = b * 36;
                int k_off = b * 36;

                // Load Q scale
                vpbroadcastw(xmm_d_q, ptr[reg_Q + q_off]);
                vcvtph2ps(xmm_d_q, xmm_d_q);

                // Load Q data
                vmovdqu8(ymm_q, ptr[reg_Q + q_off + 4]);
                vpxord(ymm_q, ymm_q, Ymm(zmm_128().getIdx()));

                // Load K scale
                vpbroadcastw(xmm_d_k, ptr[reg_K + k_off]);
                vcvtph2ps(xmm_d_k, xmm_d_k);

                // Load K sum_qs
                vpbroadcastw(xmm_sum_qs_k, ptr[reg_K + k_off + 2]);
                vpmovsxwd(xmm_sum_qs_k, xmm_sum_qs_k);
                vcvtdq2ps(xmm_sum_qs_k, xmm_sum_qs_k);

                // Load K data
                vmovdqu8(ymm_k, ptr[reg_K + k_off + 4]);

                // vpdpbusd -> 8 int32 sums
                vxorps(ymm_dot, ymm_dot, ymm_dot);
                vpdpbusd(ymm_dot, ymm_q, ymm_k);

                // Convert to float
                vcvtdq2ps(ymm_dot, ymm_dot);

                // Scale factor: d_q * d_k
                vmulps(xmm_scale_block, xmm_d_q, xmm_d_k);
                vbroadcastss(Ymm(xmm_scale_block.getIdx()), xmm_scale_block);

                // Scale the dot product
                vmulps(ymm_dot, ymm_dot, Ymm(xmm_scale_block.getIdx()));

                // Calculate correction: 16 * sum_qs * (d_q * d_k)
                // 16.0f = 0x41800000 (distributed correction for 8 lanes)
                mov(eax, 0x41800000);
                vmovd(xmm_correction, eax);
                vbroadcastss(Ymm(xmm_correction.getIdx()), xmm_correction);

                // Broadcast sum_qs to YMM
                vbroadcastss(Ymm(xmm_sum_qs_k.getIdx()), xmm_sum_qs_k);

                vmulps(Ymm(xmm_correction.getIdx()), Ymm(xmm_correction.getIdx()), Ymm(xmm_sum_qs_k.getIdx()));
                vmulps(Ymm(xmm_correction.getIdx()), Ymm(xmm_correction.getIdx()), Ymm(xmm_scale_block.getIdx()));

                // Subtract correction
                vsubps(ymm_dot, ymm_dot, Ymm(xmm_correction.getIdx()));

                // Accumulate
                vaddps(ymm_acc, ymm_acc, ymm_dot);
            }

            // Horizontal sum of ymm_acc (8 floats) -> dst (scalar)
            Xmm xmm_tmp = Xmm(ymm_dot.getIdx()); // Reuse ymm_dot register as temp
            vextractf128(xmm_tmp, ymm_acc, 1);
            vaddps(dst, Xmm(ymm_acc.getIdx()), xmm_tmp); // dst = low + high (FLOAT add)
            vshufps(xmm_tmp, dst, dst, 0x4E);
            vaddps(dst, dst, xmm_tmp);
            vshufps(xmm_tmp, dst, dst, 0xB1);
            vaddps(dst, dst, xmm_tmp);

            // Final scale
            vmulss(dst, dst, scale);
        }

        /**
         * @brief Fast scalar exp for prefill (using range reduction)
         */
        void emit_prefill_exp_scalar(const Xbyak::Xmm &dst, const Xbyak::Xmm &src)
        {
            using namespace Xbyak;

            // Scratch registers
            // Avoid zmm20 (xmm_score) and zmm22 (xmm_new_max) which are live
            Xmm xmm_n = Xmm(zmm_scratch(3).getIdx()); // zmm23
            Xmm xmm_f = Xmm(zmm_scratch(4).getIdx()); // zmm24
            Xmm xmm_p = Xmm(zmm_scratch(5).getIdx()); // zmm25

            // Load log2e
            Xmm xmm_log2e = Xmm(zmm_log2e().getIdx());

            // x * log2e
            vmulss(xmm_f, src, xmm_log2e);

            // n = floor(x * log2e)
            vrndscaless(xmm_n, xmm_n, xmm_f, 0x01); // Round down (floor)

            // f = x * log2e - n
            vsubss(xmm_f, xmm_f, xmm_n);

            // Polynomial for 2^f (f in [0, 1))
            // p = c0 + f * (c1 + f * (c2 + f * (c3 + f * (c4 + f * c5))))
            // Coefficients from JitFastExp.h (Taylor for 2^x)

            mov(eax, 0x3aaec3b4); // c5 = 0.0013334f
            vmovd(xmm_p, eax);

            mov(eax, 0x3c1d9250);           // c4 = 0.0096181f
            vmovd(dst, eax);                // reuse dst as temp
            vfmadd213ss(xmm_p, xmm_f, dst); // p = p*f + c4

            mov(eax, 0x3d63570a); // c3 = 0.0555041f
            vmovd(dst, eax);
            vfmadd213ss(xmm_p, xmm_f, dst); // p = p*f + c3

            mov(eax, 0x3e75fdf0); // c2 = 0.2402265f
            vmovd(dst, eax);
            vfmadd213ss(xmm_p, xmm_f, dst); // p = p*f + c2

            mov(eax, 0x3f317218); // c1 = 0.6931472f
            vmovd(dst, eax);
            vfmadd213ss(xmm_p, xmm_f, dst); // p = p*f + c1

            // p = p*f + 1.0
            Xmm xmm_one = Xmm(zmm_one().getIdx());
            vfmadd213ss(xmm_p, xmm_f, xmm_one);

            // Scale by 2^n
            // dst = p * 2^n
            vscalefss(dst, xmm_p, xmm_n);
        }

        /**
         * @brief Accumulate weighted V for prefill
         */
        void emit_prefill_v_accum(
            const Xbyak::Reg64 &reg_V,
            const Xbyak::Reg64 &reg_ctx,
            int num_blocks)
        {
            using namespace Xbyak;

            for (int b = 0; b < num_blocks; ++b)
            {
                int v_off = b * 36;
                int ctx_off = b * 32 * 4;

                // Load V scale
                Zmm zmm_v_scale = zmm_scratch(0);
                vpbroadcastw(Xmm(zmm_v_scale.getIdx()), ptr[reg_V + v_off]);
                vcvtph2ps(Xmm(zmm_v_scale.getIdx()), Xmm(zmm_v_scale.getIdx()));
                vbroadcastss(zmm_v_scale, Xmm(zmm_v_scale.getIdx()));

                // combined = v_scale * weight
                Zmm zmm_comb = zmm_scratch(1);
                vmulps(zmm_comb, zmm_v_scale, zmm_weight());

                // Load V data
                Zmm zmm_v_lo = zmm_scratch(2);
                Zmm zmm_v_hi = zmm_scratch(3);
                vpmovsxbd(zmm_v_lo, ptr[reg_V + v_off + 4]);
                vpmovsxbd(zmm_v_hi, ptr[reg_V + v_off + 4 + 16]);
                vcvtdq2ps(zmm_v_lo, zmm_v_lo);
                vcvtdq2ps(zmm_v_hi, zmm_v_hi);

                vmulps(zmm_v_lo, zmm_v_lo, zmm_comb);
                vmulps(zmm_v_hi, zmm_v_hi, zmm_comb);

                // Load context
                Zmm zmm_ctx_lo = zmm_scratch(4);
                Zmm zmm_ctx_hi = zmm_scratch(5);
                vmovups(zmm_ctx_lo, ptr[reg_ctx + ctx_off]);
                vmovups(zmm_ctx_hi, ptr[reg_ctx + ctx_off + 64]);

                // context = context * corr + weighted_v
                vmulps(zmm_ctx_lo, zmm_ctx_lo, zmm_corr());
                vmulps(zmm_ctx_hi, zmm_ctx_hi, zmm_corr());
                vaddps(zmm_ctx_lo, zmm_ctx_lo, zmm_v_lo);
                vaddps(zmm_ctx_hi, zmm_ctx_hi, zmm_v_hi);

                // Store
                vmovups(ptr[reg_ctx + ctx_off], zmm_ctx_lo);
                vmovups(ptr[reg_ctx + ctx_off + 64], zmm_ctx_hi);
            }
        }

        /**
         * @brief Normalize and project for prefill tile
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

            xor_(r8, r8); // q_local

            L(".proj_loop");
            cmp(r8, ptr[rsp + tile_size_spill]);
            jge(".proj_end", T_NEAR);

            mov(ptr[rsp + q_local_spill], r8); // Save q_local (clobbered by emit_prefill_wo_projection)

            // Normalize all heads for this query
            // r10 = context_ptr_offset = context_offset + q_local * d_model * 4
            mov(r10, r8);
            imul(r10, r10, d_model * 4);
            add(r10, context_offset);

            // r11 = softmax_ptr_offset = softmax_offset + q_local * 8
            mov(r11, r8);
            shl(r11, 3);
            add(r11, softmax_offset);

            Zmm zmm_inv_sum = zmm_scratch(0);
            Zmm zmm_one = zmm_scratch(1);
            load_constant_f32(zmm_one, 1.0f);
            Zmm zmm_epsilon = zmm_scratch(2);
            load_constant_f32(zmm_epsilon, 1e-20f);

            // Head Loop (Assembly)
            xor_(r9, r9); // head_idx (Use r9 instead of r12)

            L(".head_loop");
            cmp(r9, config_.num_heads);
            jge(".head_end", T_NEAR);

            // Load 1/sum for this head
            // r11 is offset from rsp
            vbroadcastss(zmm_inv_sum, ptr[rsp + r11 + 4]);
            vmaxps(zmm_inv_sum, zmm_inv_sum, zmm_epsilon);
            vdivps(zmm_inv_sum, zmm_one, zmm_inv_sum);

            for (int b = 0; b < num_blocks; ++b)
            {
                int b_off = b * 32 * 4;
                Zmm zmm_lo_safe = zmm_scratch(4);
                Zmm zmm_hi_safe = zmm_scratch(3);

                // r10 is offset from rsp
                vmovups(zmm_lo_safe, ptr[rsp + r10 + b_off]);
                vmovups(zmm_hi_safe, ptr[rsp + r10 + b_off + 64]);
                vmulps(zmm_lo_safe, zmm_lo_safe, zmm_inv_sum);
                vmulps(zmm_hi_safe, zmm_hi_safe, zmm_inv_sum);
                vmovups(ptr[rsp + r10 + b_off], zmm_lo_safe);
                vmovups(ptr[rsp + r10 + b_off + 64], zmm_hi_safe);
            }

            // Advance pointers
            add(r11, tile_size * 8); // Next head's softmax
            add(r10, head_dim * 4);  // Next head's context

            inc(r9);
            jmp(".head_loop", T_NEAR);

            L(".head_end");

            mov(r8, ptr[rsp + q_local_spill]); // Restore q_local
            inc(r8);
            jmp(".proj_loop", T_NEAR);

            L(".proj_end");

            // Tile-wide projection
            emit_prefill_wo_projection_tile(
                reg_Wo, reg_output, tile_size, context_offset,
                tile_start_spill, tile_size_spill, d_model, q_loop_spill);

            outLocalLabel();
        }

        /**
         * @brief Wo projection for a whole tile in prefill
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

            // Dispatch by format
            switch (config_.wo_format)
            {
            case WoFormat::Q8_1:
                emit_prefill_wo_q8_1_tile(reg_Wo, reg_output, tile_size, context_offset, tile_start_spill, tile_size_spill, d_model);
                break;
            default:
                // Fallback to sequential for other formats (or implement them later)
                // For now, just loop and call single projection
                // NOTE: emit_prefill_wo_projection uses r8 as row counter, so save/restore it
                // Use a stack spill for the query loop counter to avoid register conflicts
                {
                    // q_loop_spill is defined in the prologue
                    mov(qword[rsp + q_loop_spill], 0);
                    L(".fallback_loop");
                    mov(r8, ptr[rsp + q_loop_spill]);
                    cmp(r8, ptr[rsp + tile_size_spill]);
                    jge(".fallback_end", T_NEAR);

                    mov(r11, ptr[rsp + tile_start_spill]);
                    add(r11, r8); // q_global

                    // r10 = context_ptr_offset
                    mov(r10, r8);
                    imul(r10, r10, d_model * 4);
                    add(r10, context_offset);

                    emit_prefill_wo_projection(reg_Wo, reg_output, r11, r10, d_model);

                    // Increment and store loop counter
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

            // We unroll i (output rows) by 2
            // We unroll q (tile queries) by tile_size (up to 8)
            // Registers:
            // acc[q] for i: zmm0..zmm7
            // acc[q] for i+1: zmm8..zmm15
            // W[i]: zmm16, zmm17, zmm18 (scale, lo, hi)
            // W[i+1]: zmm19, zmm20, zmm21
            // C[q]: zmm22, zmm23 (lo, hi)

            // Outer loop over i (output rows)
            xor_(r8, r8); // i

            L(".row_loop");
            cmp(r8, d_model);
            jge(".row_end", T_NEAR);

            // Check if we can do 2 rows
            mov(r9, d_model);
            sub(r9, r8);
            cmp(r9, 2);
            jl(".single_row", T_NEAR);

            // --- Double row processing ---

            // Clear accumulators
            for (int q = 0; q < tile_size; ++q)
            {
                vxorps(Zmm(q), Zmm(q), Zmm(q));             // acc[q] for i
                vxorps(Zmm(q + 8), Zmm(q + 8), Zmm(q + 8)); // acc[q] for i+1
            }

            // Pointers to W rows
            // r12 = W + i * row_size
            // r13 = W + (i+1) * row_size
            mov(r12, r8);
            imul(r12, r12, num_blocks * 36);
            add(r12, reg_Wo);

            mov(r13, r12);
            add(r13, num_blocks * 36);

            // Inner loop over blocks
            xor_(rcx, rcx); // k_blk

            L(".blk_loop_2");
            cmp(rcx, num_blocks);
            jge(".blk_end_2", T_NEAR);

            // Load W[i] block
            // Offset = rcx * 36
            mov(rax, rcx);
            imul(rax, rax, 36);

            // Load W[i]
            {
                Zmm d0 = Zmm(16);
                Zmm w0_lo = Zmm(17);
                Zmm w0_hi = Zmm(18);

                vpbroadcastw(Xmm(d0.getIdx()), ptr[r12 + rax]);
                vcvtph2ps(Xmm(d0.getIdx()), Xmm(d0.getIdx()));
                vbroadcastss(d0, Xmm(d0.getIdx()));

                vpmovsxbd(w0_lo, ptr[r12 + rax + 4]);
                vpmovsxbd(w0_hi, ptr[r12 + rax + 4 + 16]);
                vcvtdq2ps(w0_lo, w0_lo);
                vcvtdq2ps(w0_hi, w0_hi);
                vmulps(w0_lo, w0_lo, d0);
                vmulps(w0_hi, w0_hi, d0);

                // Load W[i+1]
                Zmm d1 = Zmm(19);
                Zmm w1_lo = Zmm(20);
                Zmm w1_hi = Zmm(21);

                vpbroadcastw(Xmm(d1.getIdx()), ptr[r13 + rax]);
                vcvtph2ps(Xmm(d1.getIdx()), Xmm(d1.getIdx()));
                vbroadcastss(d1, Xmm(d1.getIdx()));

                vpmovsxbd(w1_lo, ptr[r13 + rax + 4]);
                vpmovsxbd(w1_hi, ptr[r13 + rax + 4 + 16]);
                vcvtdq2ps(w1_lo, w1_lo);
                vcvtdq2ps(w1_hi, w1_hi);
                vmulps(w1_lo, w1_lo, d1);
                vmulps(w1_hi, w1_hi, d1);

                // Loop over q
                // Context offset: context_offset + q * d_model * 4 + k_blk * 128
                // We can compute base pointer for k_blk once
                // base = rsp + context_offset + k_blk * 128
                mov(rdx, rcx);
                shl(rdx, 7); // * 128
                add(rdx, context_offset);
                add(rdx, rsp);

                for (int q = 0; q < tile_size; ++q)
                {
                    // Load C[q]
                    // Offset = q * d_model * 4
                    // But d_model is variable.
                    // We can precompute offsets? Or just compute on fly.
                    // q * d_model * 4
                    // Since tile_size is small (8), we can just do it.

                    Zmm c_lo = Zmm(22);
                    Zmm c_hi = Zmm(23);

                    // r14 = q * d_model * 4
                    // We can use a scratch reg. r14 is callee-save? Xbyak handles it?
                    // No, we should use caller-save. r14 is callee-save.
                    // r10, r11 are scratch.

                    mov(r10, q);
                    imul(r10, r10, d_model * 4);

                    vmovups(c_lo, ptr[rdx + r10]);
                    vmovups(c_hi, ptr[rdx + r10 + 64]);

                    // Accumulate i
                    vfmadd231ps(Zmm(q), c_lo, w0_lo);
                    vfmadd231ps(Zmm(q), c_hi, w0_hi);

                    // Accumulate i+1
                    vfmadd231ps(Zmm(q + 8), c_lo, w1_lo);
                    vfmadd231ps(Zmm(q + 8), c_hi, w1_hi);
                }
            }

            inc(rcx);
            jmp(".blk_loop_2", T_NEAR);
            L(".blk_end_2");

            // Horizontal sum and store
            // We need to check actual_tile_size
            mov(rax, ptr[rsp + tile_size_spill]); // actual size (use rax instead of r14/reg_output)

            // Output ptr base
            mov(r11, ptr[rsp + tile_start_spill]); // tile_start
            // output[q_global][i]
            // ptr = reg_output + (tile_start + q) * d_model * 4 + i * 4

            for (int q = 0; q < tile_size; ++q)
            {
                // Check if q < actual_tile_size
                cmp(rax, q);
                jle(".skip_store_2_" + std::to_string(q), T_NEAR);

                // Sum i
                emit_horizontal_sum_to_scalar(Zmm(q));

                // Sum i+1
                emit_horizontal_sum_to_scalar(Zmm(q + 8));

                // Store
                // ptr = reg_output + (tile_start + q) * d_model * 4 + r8 * 4
                mov(r10, ptr[rsp + tile_start_spill]);
                add(r10, q);
                imul(r10, r10, d_model * 4);
                add(r10, reg_output);

                vmovss(ptr[r10 + r8 * 4], Xmm(Zmm(q).getIdx()));
                vmovss(ptr[r10 + r8 * 4 + 4], Xmm(Zmm(q + 8).getIdx()));

                L(".skip_store_2_" + std::to_string(q));
            }

            add(r8, 2);
            jmp(".row_loop", T_NEAR);

            L(".single_row");
            // --- Single row processing (fallback for odd d_model) ---
            // (Simplified version of above, only 1 row)

            // Clear accumulators
            for (int q = 0; q < tile_size; ++q)
            {
                vxorps(Zmm(q), Zmm(q), Zmm(q));
            }

            mov(r12, r8);
            imul(r12, r12, num_blocks * 36);
            add(r12, reg_Wo);

            xor_(rcx, rcx);
            L(".blk_loop_1");
            cmp(rcx, num_blocks);
            jge(".blk_end_1", T_NEAR);

            // Load W[i]
            mov(rax, rcx);
            imul(rax, rax, 36);

            // Load W[i]
            {
                Zmm d0 = Zmm(16);
                Zmm w0_lo = Zmm(17);
                Zmm w0_hi = Zmm(18);

                vpbroadcastw(Xmm(d0.getIdx()), ptr[r12 + rax]);
                vcvtph2ps(Xmm(d0.getIdx()), Xmm(d0.getIdx()));
                vbroadcastss(d0, Xmm(d0.getIdx()));

                vpmovsxbd(w0_lo, ptr[r12 + rax + 4]);
                vpmovsxbd(w0_hi, ptr[r12 + rax + 4 + 16]);
                vcvtdq2ps(w0_lo, w0_lo);
                vcvtdq2ps(w0_hi, w0_hi);
                vmulps(w0_lo, w0_lo, d0);
                vmulps(w0_hi, w0_hi, d0);

                mov(rdx, rcx);
                shl(rdx, 7);
                add(rdx, context_offset);
                add(rdx, rsp);

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

            mov(rax, ptr[rsp + tile_size_spill]); // use rax instead of r14
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
         * @brief Wo projection for one query in prefill
         */
        void emit_prefill_wo_projection(
            const Xbyak::Reg64 &reg_Wo,
            const Xbyak::Reg64 &reg_output,
            const Xbyak::Reg64 &reg_q_global,
            const Xbyak::Reg64 &reg_ctx_offset,
            int d_model)
        {
            using namespace Xbyak;

            // Output pointer
            Reg64 reg_out_ptr = r11; // Use r11 since rdi is used as scratch inside emit_wo_projection_* methods
            mov(reg_out_ptr, reg_q_global);
            imul(reg_out_ptr, reg_out_ptr, d_model * 4);
            add(reg_out_ptr, reg_output);

            // Dispatch by format
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

        void emit_prefill_wo_fp32(const Xbyak::Reg64 &reg_Wo, const Xbyak::Reg64 &reg_out,
                                  const Xbyak::Reg64 &reg_ctx_off, int d_model)
        {
            using namespace Xbyak;
            inLocalLabel();

            xor_(r8, r8);
            L(".row");
            cmp(r8, d_model);
            jge(".end", T_NEAR);

            Zmm acc = zmm_scratch(0);
            vxorps(acc, acc, acc);

            mov(r9, r8);
            imul(r9, r9, d_model * 4);
            add(r9, reg_Wo);

            xor_(rcx, rcx);
            L(".col");
            cmp(rcx, d_model);
            jge(".col_end", T_NEAR);

            Zmm ctx = zmm_scratch(1);
            Zmm wo = zmm_scratch(2);
            lea(rax, ptr[rsp + reg_ctx_off]);
            vmovups(ctx, ptr[rax + rcx * 4]);
            vmovups(wo, ptr[r9 + rcx * 4]);
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

        void emit_prefill_wo_fp16(const Xbyak::Reg64 &reg_Wo, const Xbyak::Reg64 &reg_out,
                                  const Xbyak::Reg64 &reg_ctx_off, int d_model)
        {
            using namespace Xbyak;
            inLocalLabel();

            xor_(r8, r8);
            L(".row");
            cmp(r8, d_model);
            jge(".end", T_NEAR);

            Zmm acc = zmm_scratch(0);
            vxorps(acc, acc, acc);

            mov(r9, r8);
            imul(r9, r9, d_model * 2);
            add(r9, reg_Wo);

            xor_(rcx, rcx);
            L(".col");
            cmp(rcx, d_model);
            jge(".col_end", T_NEAR);

            Zmm ctx = zmm_scratch(1);
            Zmm wo = zmm_scratch(2);
            lea(rax, ptr[rsp + reg_ctx_off]);
            vmovups(ctx, ptr[rax + rcx * 4]);
            vmovups(Ymm(wo.getIdx()), ptr[r9 + rcx * 2]);
            vcvtph2ps(wo, Ymm(wo.getIdx()));
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

        void emit_prefill_wo_bf16(const Xbyak::Reg64 &reg_Wo, const Xbyak::Reg64 &reg_out,
                                  const Xbyak::Reg64 &reg_ctx_off, int d_model)
        {
            using namespace Xbyak;
            inLocalLabel();

            xor_(r8, r8);
            L(".row");
            cmp(r8, d_model);
            jge(".end", T_NEAR);

            Zmm acc = zmm_scratch(0);
            vxorps(acc, acc, acc);

            mov(r9, r8);
            imul(r9, r9, d_model * 2);
            add(r9, reg_Wo);

            xor_(rcx, rcx);
            L(".col");
            cmp(rcx, d_model);
            jge(".col_end", T_NEAR);

            Zmm ctx = zmm_scratch(1);
            Zmm wo = zmm_scratch(2);
            lea(rax, ptr[rsp + reg_ctx_off]);
            vmovups(ctx, ptr[rax + rcx * 4]);
            vpmovzxwd(wo, ptr[r9 + rcx * 2]);
            vpslld(wo, wo, 16);
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

        void emit_prefill_wo_q8_1(const Xbyak::Reg64 &reg_Wo, const Xbyak::Reg64 &reg_out,
                                  const Xbyak::Reg64 &reg_ctx_off, int d_model)
        {
            using namespace Xbyak;
            inLocalLabel();

            int num_blocks = d_model / 32;

            xor_(r8, r8);
            L(".row");
            cmp(r8, d_model);
            jge(".end", T_NEAR);

            Zmm acc = zmm_scratch(0);
            vxorps(acc, acc, acc);

            mov(r9, r8);
            imul(r9, r9, num_blocks * 36);
            add(r9, reg_Wo);

            xor_(rcx, rcx);
            L(".blk");
            cmp(rcx, num_blocks);
            jge(".blk_end", T_NEAR);

            // Context block
            mov(rdx, rcx);
            shl(rdx, 7);
            lea(rax, ptr[rsp + reg_ctx_off]);
            add(rdx, rax);

            Zmm ctx_lo = zmm_scratch(1);
            Zmm ctx_hi = zmm_scratch(2);
            vmovups(ctx_lo, ptr[rdx]);
            vmovups(ctx_hi, ptr[rdx + 64]);

            // Wo block
            mov(rax, rcx);
            imul(rax, rax, 36);
            add(rax, r9);

            // Load scale
            Zmm d = zmm_scratch(3);
            vpbroadcastw(Xmm(d.getIdx()), ptr[rax]);
            vcvtph2ps(Xmm(d.getIdx()), Xmm(d.getIdx()));
            vbroadcastss(d, Xmm(d.getIdx()));

            // Load data
            Zmm wo_lo = zmm_scratch(4);
            Zmm wo_hi = zmm_scratch(5);
            vpmovsxbd(wo_lo, ptr[rax + 4]);
            vpmovsxbd(wo_hi, ptr[rax + 4 + 16]);
            vcvtdq2ps(wo_lo, wo_lo);
            vcvtdq2ps(wo_hi, wo_hi);
            vmulps(wo_lo, wo_lo, d);
            vmulps(wo_hi, wo_hi, d);

            vfmadd231ps(acc, ctx_lo, wo_lo);
            vfmadd231ps(acc, ctx_hi, wo_hi);

            inc(rcx);
            jmp(".blk", T_NEAR);
            L(".blk_end");

            emit_horizontal_sum_to_scalar(acc);
            vmovss(ptr[reg_out + r8 * 4], Xmm(acc.getIdx()));

            inc(r8);
            jmp(".row", T_NEAR);
            L(".end");
            outLocalLabel();
        }

        /**
         * @brief Emit attention computation for a single query position
         *
         * For each head:
         * 1. Copy Q[q,h] blocks to stack
         * 2. Loop over KV positions (with optional causal masking):
         *    - Compute score = Q·K with Q from stack
         *    - Online softmax update
         *    - Weighted V accumulation
         * 3. Normalize context by 1/sum
         * 4. Store to context buffer
         * 5. Apply Wo projection: output = context * Wo^T
         *
         * Causal masking: when config_.causal is true, each query position
         * q can only attend to KV positions [0, q + position_offset].
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
            int position_offset_spill_offset)
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
                load_constant_f32(zmm_max(), -3.4028235e+38f); // -FLT_MAX
                vxorps(zmm_sum(), zmm_sum(), zmm_sum());

                // Initialize constant for unsigned Q conversion
                // NOTE: Use rsi as scratch (rdi is used for Q_base below)
                emit_broadcast_i32_const(zmm_128(), 0x80808080, rsi);

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

                // Normalize context by 1/sum for this head
                Zmm zmm_inv_sum = zmm_scratch(4);
                load_constant_f32(zmm_scratch(5), 1.0f);
                vdivps(zmm_inv_sum, zmm_scratch(5), zmm_sum());
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
         * @brief Copy Q blocks for one head to stack
         *
         * Copies Q8_1 blocks from [reg_Q_base + h * num_blocks * 36] to
         * [rsp + q_stack_offset] with proper alignment.
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
            // zmm_scratch(0) = zmm20, we'll use the lower 256 bits
            Zmm zmm_tmp = zmm_scratch(0);

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
                // Use zmm10-13 to avoid conflict with StateRegs (16-19) and ConstRegs (26-31)
                Ymm ymm_q(10 + b);
                vmovdqu8(ymm_q, ptr[reg_Q_base + src_offset + 4]);

                // Convert to unsigned (XOR 0x80)
                vpxord(ymm_q, ymm_q, Ymm(zmm_128().getIdx()));
            }
        }

        /**
         * @brief Store head context from accumulators to output
         *
         * Simplified output: directly stores FP32 context to output buffer.
         * Full Wo projection would be applied separately.
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
            vmovups(ptr[reg_out_ptr], zmm_accum(0));      // floats 0-15
            vmovups(ptr[reg_out_ptr + 64], zmm_accum(1)); // floats 16-31

            if (num_blocks >= 2)
            {
                vmovups(ptr[reg_out_ptr + 128], zmm_accum(2)); // floats 32-47
                vmovups(ptr[reg_out_ptr + 192], zmm_accum(3)); // floats 48-63
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

                    vmovups(zmm_scratch(0), ptr[rsp + spill_lo]);
                    vmovups(ptr[reg_out_ptr + out_lo], zmm_scratch(0));

                    vmovups(zmm_scratch(0), ptr[rsp + spill_hi]);
                    vmovups(ptr[reg_out_ptr + out_hi], zmm_scratch(0));
                }
            }
        }

        /**
         * @brief Store head context from accumulators to context buffer on stack
         *
         * Stores FP32 context to the context buffer for later Wo projection.
         * Context buffer layout: [num_heads * head_dim] FP32
         *
         * @param head_idx Head index
         * @param num_blocks Number of Q8_1 blocks per head
         * @param spill_base_offset Stack offset for spilled accumulators
         * @param context_buffer_offset Stack offset for context buffer
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
            vmovups(ptr[rsp + ctx_ptr], zmm_accum(0));      // floats 0-15
            vmovups(ptr[rsp + ctx_ptr + 64], zmm_accum(1)); // floats 16-31

            if (num_blocks >= 2)
            {
                vmovups(ptr[rsp + ctx_ptr + 128], zmm_accum(2)); // floats 32-47
                vmovups(ptr[rsp + ctx_ptr + 192], zmm_accum(3)); // floats 48-63
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
                    vmovups(zmm_scratch(0), ptr[rsp + spill_lo]);
                    vmovups(ptr[rsp + out_lo], zmm_scratch(0));

                    vmovups(zmm_scratch(0), ptr[rsp + spill_hi]);
                    vmovups(ptr[rsp + out_hi], zmm_scratch(0));
                }
            }
        }

        /**
         * @brief Emit Wo projection from context buffer to output
         *
         * Computes: output[i] = sum_j(context[j] * Wo[i, j])
         *
         * Context buffer: [num_heads * head_dim] = [d_model] FP32 on stack
         * Wo weights: [d_model, d_model] (row-major)
         * Output: [d_model] FP32
         *
         * For each output position i:
         *   Dot product of context[0..d_model-1] with Wo row i
         *
         * @param reg_Wo Register holding Wo pointer
         * @param reg_output Register holding output pointer
         * @param reg_q_idx Register holding query index
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

            // Dispatch based on Wo format
            switch (config_.wo_format)
            {
            case WoFormat::FP32:
                emit_wo_projection_fp32(reg_Wo, reg_out_ptr, context_buffer_offset, d_model);
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
            }
        }

        /**
         * @brief Emit FP32 Wo projection
         *
         * For each output row, compute dot product with context vector.
         * Uses runtime loops instead of compile-time unrolling to avoid code bloat.
         * Uses AVX-512 for 16-way SIMD.
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
            Zmm zmm_acc = zmm_scratch(0);
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
            Zmm zmm_ctx = zmm_scratch(1);
            lea(rdi, ptr[rsp + context_buffer_offset]);
            vmovups(zmm_ctx, ptr[rdi + reg_j * 4]);

            // Load Wo row chunk
            Zmm zmm_wo = zmm_scratch(2);
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
         * @brief Emit FP16 Wo projection
         * Uses runtime loops to avoid code bloat.
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

            Zmm zmm_acc = zmm_scratch(0);
            vxorps(zmm_acc, zmm_acc, zmm_acc);

            // Wo row pointer: Wo + out_idx * d_model * 2 (FP16)
            mov(reg_wo_row, reg_out_idx);
            imul(reg_wo_row, reg_wo_row, d_model * 2);
            add(reg_wo_row, reg_Wo);

            xor_(reg_j, reg_j);

            L(".inner_loop");
            cmp(reg_j, d_model);
            jge(".inner_end", T_NEAR);

            Zmm zmm_ctx = zmm_scratch(1);
            Zmm zmm_wo = zmm_scratch(2);
            Ymm ymm_fp16 = Ymm(zmm_wo.getIdx());

            lea(rdi, ptr[rsp + context_buffer_offset]);
            vmovups(zmm_ctx, ptr[rdi + reg_j * 4]);
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
         * @brief Emit BF16 Wo projection
         * Uses runtime loops to avoid code bloat.
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

            Zmm zmm_acc = zmm_scratch(0);
            vxorps(zmm_acc, zmm_acc, zmm_acc);

            // Wo row pointer: Wo + out_idx * d_model * 2 (BF16)
            mov(reg_wo_row, reg_out_idx);
            imul(reg_wo_row, reg_wo_row, d_model * 2);
            add(reg_wo_row, reg_Wo);

            xor_(reg_j, reg_j);

            L(".inner_loop");
            cmp(reg_j, d_model);
            jge(".inner_end", T_NEAR);

            Zmm zmm_ctx = zmm_scratch(1);
            Zmm zmm_wo = zmm_scratch(2);

            lea(rdi, ptr[rsp + context_buffer_offset]);
            vmovups(zmm_ctx, ptr[rdi + reg_j * 4]);
            // BF16 -> FP32: zero-extend to 32-bit, shift left 16
            vpmovzxwd(zmm_wo, ptr[reg_wo_row + reg_j * 2]);
            vpslld(zmm_wo, zmm_wo, 16);
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
         * @brief Emit Q8_1 Wo projection
         *
         * Q8_1 Wo layout: [d_model, num_blocks_per_row] Q8_1 blocks
         * Each block covers 32 elements. Uses runtime loops to avoid code bloat.
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
            Reg64 reg_ctx_ptr = rdi; // Use rdi since it's scratch in inner loop anyway

            xor_(reg_out_idx, reg_out_idx);

            L(".outer_loop");
            cmp(reg_out_idx, d_model);
            jge(".outer_end", T_NEAR);

            Zmm zmm_acc = zmm_scratch(0);
            vxorps(zmm_acc, zmm_acc, zmm_acc);

            // Wo row pointer: Wo + out_idx * num_blocks_per_row * 36
            mov(reg_wo_row, reg_out_idx);
            imul(reg_wo_row, reg_wo_row, num_blocks_per_row * 36);
            add(reg_wo_row, reg_Wo);

            xor_(reg_b, reg_b);

            L(".inner_loop");
            cmp(reg_b, num_blocks_per_row);
            jge(".inner_end", T_NEAR);

            // Context ptr: rsp + context_buffer_offset + b * 128
            mov(reg_ctx_ptr, reg_b);
            shl(reg_ctx_ptr, 7); // * 128 (32 floats * 4 bytes)
            add(reg_ctx_ptr, context_buffer_offset);
            add(reg_ctx_ptr, rsp);

            // Load context block (32 floats in 2 ZMM registers)
            Zmm zmm_ctx_lo = zmm_scratch(1);
            Zmm zmm_ctx_hi = zmm_scratch(2);
            vmovups(zmm_ctx_lo, ptr[reg_ctx_ptr]);
            vmovups(zmm_ctx_hi, ptr[reg_ctx_ptr + 64]);

            // Wo block ptr: reg_wo_row + b * 36
            Reg64 reg_wo_block = rdi;
            mov(reg_wo_block, reg_b);
            imul(reg_wo_block, reg_wo_block, 36);
            add(reg_wo_block, reg_wo_row);

            // Load Q8_1 scale (FP16 at offset 0)
            Zmm zmm_d = zmm_scratch(3);
            vpbroadcastw(Xmm(zmm_d.getIdx()), ptr[reg_wo_block]);
            vcvtph2ps(Xmm(zmm_d.getIdx()), Xmm(zmm_d.getIdx()));
            vbroadcastss(zmm_d, Xmm(zmm_d.getIdx()));

            // Load Q8_1 data (32 int8 at offset 4)
            Zmm zmm_wo_lo = zmm_scratch(4);
            Zmm zmm_wo_hi = zmm_scratch(5);
            vpmovsxbd(zmm_wo_lo, ptr[reg_wo_block + 4]);
            vpmovsxbd(zmm_wo_hi, ptr[reg_wo_block + 4 + 16]);

            // Convert to float and scale
            vcvtdq2ps(zmm_wo_lo, zmm_wo_lo);
            vcvtdq2ps(zmm_wo_hi, zmm_wo_hi);
            vmulps(zmm_wo_lo, zmm_wo_lo, zmm_d);
            vmulps(zmm_wo_hi, zmm_wo_hi, zmm_d);

            // Accumulate dot product
            vfmadd231ps(zmm_acc, zmm_ctx_lo, zmm_wo_lo);
            vfmadd231ps(zmm_acc, zmm_ctx_hi, zmm_wo_hi);

            inc(reg_b);
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
         * @brief Horizontal sum of ZMM register to scalar in xmm[0]
         *
         * Uses the same register for input/output. Uses accum registers as scratch
         * since they're not in use during Wo projection (output was already stored).
         * Result is in element 0 of the input zmm's corresponding xmm.
         */
        void emit_horizontal_sum_to_scalar(const Xbyak::Zmm &zmm)
        {
            using namespace Xbyak;

            // Use zmm25 (scratch) as temporary to avoid clobbering zmm16 (StateRegs::ZMM_MAX)
            // and to avoid aliasing with zmm20 (which is often the accumulator passed as zmm)
            // Also avoid zmm21 which is used as scratch in some contexts
            Zmm zmm_tmp = Zmm(25);
            Ymm ymm_tmp = Ymm(zmm_tmp.getIdx());
            Xmm xmm_tmp = Xmm(zmm_tmp.getIdx());

            // zmm -> ymm: add upper 256 to lower
            vextractf32x8(ymm_tmp, zmm, 1);
            vaddps(Ymm(zmm.getIdx()), Ymm(zmm.getIdx()), ymm_tmp);

            // ymm -> xmm: add upper 128 to lower
            // Use vextractf32x4 which supports EVEX (all registers)
            vextractf32x4(xmm_tmp, Zmm(zmm.getIdx()), 1);
            vaddps(Xmm(zmm.getIdx()), Xmm(zmm.getIdx()), xmm_tmp);

            // xmm -> scalar: horizontal sum
            vshufps(xmm_tmp, Xmm(zmm.getIdx()), Xmm(zmm.getIdx()), 0x4E); // Swap high/low 64 bits
            vaddps(Xmm(zmm.getIdx()), Xmm(zmm.getIdx()), xmm_tmp);
            vshufps(xmm_tmp, Xmm(zmm.getIdx()), Xmm(zmm.getIdx()), 0xB1); // Swap high/low 32 bits
            vaddss(Xmm(zmm.getIdx()), Xmm(zmm.getIdx()), xmm_tmp);
        }

        /**
         * @brief Emit attention for single head at single KV position
         *
         * Computes score = Q[h] · K[kv, kv_h], applies online softmax,
         * and accumulates weighted V into context.
         *
         * Note: This method assumes Q head blocks are already copied to stack
         * at q_stack_offset. The caller must ensure this before calling.
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

            // Calculate K pointer for this KV position and head
            // K layout: [seq_len_kv, num_kv_heads, head_dim] in Q8_1 blocks
            Reg64 reg_K_ptr = rdi; // Temp reuse
            int k_stride = config_.num_kv_heads * num_blocks * 36;

            mov(reg_K_ptr, reg_kv_idx);
            imul(reg_K_ptr, reg_K_ptr, k_stride);
            add(reg_K_ptr, kv_head_idx * num_blocks * 36);
            add(reg_K_ptr, reg_K);

            // Q[q, h] dot K[kv, kv_h]
            // Use Q8 dot product microkernel
            // Q head blocks are pre-loaded in ZMM registers (16+) and XMM scales (24+)
            Xmm xmm_score_result(zmm_scratch(1).getIdx()); // Use xmm21 for score result (avoids accumulators)
            Xmm xmm_scale_local(zmm_scratch(2).getIdx());  // Use xmm22 for scale (avoids accumulators)

            // Load scale from zmm_scale() element 0
            vmovss(xmm_scale_local, Xmm(zmm_scale().getIdx()));

            // Use register-based dot product for Decode mode
            // Q data in ZMM 10+, d_Q in memory (loaded via reg_Q_ptr)
            int q_head_offset = head_idx * num_blocks * 36;
            dot_emitter_.emit_dot_product_register_q_mem_dq(*this, xmm_score_result, reg_K_ptr, reg_Q_ptr,
                                                            q_head_offset, num_blocks, xmm_scale_local, 10);

            // Score is now in xmm_score_result element 0

            // Online softmax update needs a label prefix for internal jumps
            std::string softmax_label = label_prefix + "_h" + std::to_string(head_idx) + "_softmax";
            softmax_emitter_.emit_update(*this, xmm_score_result, softmax_label);

            // After softmax update:
            // - zmm_weight() contains the attention weight for current KV position
            // - zmm_corr() contains correction factor (1.0 if no rescale needed, <1.0 if max changed)
            // Rescale context accumulators by zmm_corr() to maintain online softmax invariant
            v_accum_emitter_.emit_rescale_context(*this, num_blocks, spill_base_offset);

            // Calculate V pointer
            Reg64 reg_V_ptr = rdi;
            int v_stride = config_.num_kv_heads * num_blocks * 36;

            mov(reg_V_ptr, reg_kv_idx);
            imul(reg_V_ptr, reg_V_ptr, v_stride);
            add(reg_V_ptr, kv_head_idx * num_blocks * 36);
            add(reg_V_ptr, reg_V);

            // Weighted V accumulation
            // weight is in zmm_weight() after softmax update

            v_accum_emitter_.emit_weighted_accum(*this, reg_V_ptr, num_blocks, spill_base_offset);
        }
    };

    /**
     * @brief Cache for JIT-generated attention kernels
     *
     * Thread-safe cache that stores generated code to avoid regeneration.
     */
    class JitAttentionKernelCache
    {
    public:
        static JitAttentionKernelCache &instance()
        {
            static JitAttentionKernelCache inst;
            return inst;
        }

        /**
         * @brief Get or generate kernel for config
         *
         * @param config Kernel configuration
         * @return Function pointer to generated kernel
         */
        JitAttentionKernelFn getKernel(const JitAttentionConfig &config)
        {
            std::lock_guard<std::mutex> lock(mutex_);

            auto it = cache_.find(config);
            if (it != cache_.end())
            {
                return it->second->getKernel();
            }

            // Generate new kernel
            auto generator = std::make_unique<JitFusedAttentionWoGenerator>(config);
            auto fn = generator->getKernel();
            cache_[config] = std::move(generator);
            return fn;
        }

        /**
         * @brief Clear all cached kernels
         */
        void clear()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cache_.clear();
        }

        /**
         * @brief Get cache statistics
         */
        size_t size() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return cache_.size();
        }

    private:
        JitAttentionKernelCache() = default;

        mutable std::mutex mutex_;
        std::unordered_map<JitAttentionConfig, std::unique_ptr<JitFusedAttentionWoGenerator>> cache_;
    };

    /**
     * @brief High-level interface for JIT fused attention
     *
     * Usage:
     *   JitFusedAttentionWo attn(config);
     *   attn.compute(Q, K, V, Wo, output, seq_len_q, seq_len_kv, scale, position_offset);
     */
    class JitFusedAttentionWo
    {
    public:
        explicit JitFusedAttentionWo(const JitAttentionConfig &config)
            : config_(config), kernel_(JitAttentionKernelCache::instance().getKernel(config))
        {
        }

        /**
         * @brief Execute fused attention + Wo projection
         *
         * @param Q Q tensor (Q8_1 blocks)
         * @param K K tensor (Q8_1 blocks)
         * @param V V tensor (Q8_1 blocks)
         * @param Wo Wo weights (format depends on config)
         * @param output Output buffer (FP32)
         * @param seq_len_q Number of query positions
         * @param seq_len_kv Number of KV positions
         * @param scale Attention scale (1/sqrt(head_dim))
         * @param position_offset Position offset for causal masking (default 0)
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
