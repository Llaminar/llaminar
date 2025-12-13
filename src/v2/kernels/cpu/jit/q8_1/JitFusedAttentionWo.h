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
     * @brief Configuration for JIT attention kernel
     */
    struct JitAttentionConfig
    {
        int head_dim;       // Dimension per head (e.g., 64)
        int num_heads;      // Number of Q heads
        int num_kv_heads;   // Number of KV heads (GQA support)
        int batch_size;     // Batch size (1 for decode, >1 for prefill)
        WoFormat wo_format; // Output projection weight format
        bool causal;        // Whether to apply causal masking

        bool operator==(const JitAttentionConfig &other) const
        {
            return head_dim == other.head_dim &&
                   num_heads == other.num_heads &&
                   num_kv_heads == other.num_kv_heads &&
                   batch_size == other.batch_size &&
                   wo_format == other.wo_format &&
                   causal == other.causal;
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
            return h;
        }
    };
}

namespace llaminar::v2::kernels::jit
{

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

        // Microkernel emitters
        JitQ8DotProductEmitter dot_emitter_;
        JitOnlineSoftmaxEmitter softmax_emitter_;
        JitVWeightedAccumEmitter v_accum_emitter_;
        JitWoProjectionEmitter wo_emitter_;

        /**
         * @brief Generate the complete kernel
         */
        void generate()
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

            // Calculate Q base pointer for this query position
            // Q layout: [seq_len_q, num_heads, head_dim] in Q8_1 blocks
            Reg64 reg_Q_base = rdi;                             // Use rdi now that we've finished with max_kv_pos in r11
            int q_stride = config_.num_heads * num_blocks * 36; // bytes per query position

            mov(reg_Q_base, reg_q_idx);
            imul(reg_Q_base, reg_Q_base, q_stride);
            add(reg_Q_base, reg_Q);

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
                // NOTE: Use rsi as scratch (rdi is now reg_Q_base)
                emit_broadcast_i32_const(zmm_128(), 0x80808080, rsi);

                // Copy Q[q,h] blocks to stack for this head
                emit_copy_q_head_to_stack(reg_Q_base, h, num_blocks, q_stack_offset);

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

            // Use zmm_accum(4) as temporary - it's zmm4, which is VEX-compatible
            // This avoids the EVEX-only issue with high-numbered registers
            Zmm zmm_tmp = zmm_accum(4);
            Ymm ymm_tmp = Ymm(zmm_tmp.getIdx());
            Xmm xmm_tmp = Xmm(zmm_tmp.getIdx());

            // zmm -> ymm: add upper 256 to lower
            vextractf32x8(ymm_tmp, zmm, 1);
            vaddps(Ymm(zmm.getIdx()), Ymm(zmm.getIdx()), ymm_tmp);

            // ymm -> xmm: add upper 128 to lower
            // vextractf128 needs VEX encoding, so operands must be < 16
            // Move result to ymm_tmp first, then extract
            vmovaps(ymm_tmp, Ymm(zmm.getIdx()));
            vextractf128(xmm_tmp, ymm_tmp, 1);
            vaddps(Xmm(zmm.getIdx()), Xmm(zmm.getIdx()), xmm_tmp);

            // xmm: horizontal add (4 floats -> 1)
            // haddps also requires VEX encoding for xmm operands
            vmovaps(xmm_tmp, Xmm(zmm.getIdx()));
            vhaddps(xmm_tmp, xmm_tmp, xmm_tmp);
            vhaddps(xmm_tmp, xmm_tmp, xmm_tmp);
            vmovaps(Xmm(zmm.getIdx()), xmm_tmp);
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
            // Q head blocks should already be on stack at q_stack_offset
            Xmm xmm_score_result(zmm_scratch(1).getIdx()); // Use xmm21 for score result (avoids accumulators)
            Xmm xmm_scale_local(zmm_scratch(2).getIdx());  // Use xmm22 for scale (avoids accumulators)

            // Load scale from zmm_scale() element 0
            vmovss(xmm_scale_local, Xmm(zmm_scale().getIdx()));

            dot_emitter_.emit_dot_product(*this, xmm_score_result, reg_K_ptr, rsp,
                                          q_stack_offset, num_blocks, xmm_scale_local);

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
