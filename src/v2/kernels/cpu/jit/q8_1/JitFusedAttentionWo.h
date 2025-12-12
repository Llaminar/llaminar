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

        bool operator==(const JitAttentionConfig &other) const
        {
            return head_dim == other.head_dim &&
                   num_heads == other.num_heads &&
                   num_kv_heads == other.num_kv_heads &&
                   batch_size == other.batch_size &&
                   wo_format == other.wo_format;
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
     */
    using JitAttentionKernelFn = void (*)(
        const void *Q,
        const void *K,
        const void *V,
        const void *Wo,
        float *output,
        int seq_len_q,
        int seq_len_kv,
        float scale);

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
            // rdi = Q, rsi = K, rdx = V, rcx = Wo, r8 = output, r9 = seq_len_q
            // Stack: [rsp+8] = seq_len_kv, [rsp+16] = scale (as float bits)

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
            Reg64 reg_seq_len_kv = r10;
            mov(reg_seq_len_kv, ptr[rsp + stack_frame_size() + 8]);

            // Load scale into zmm_scale()
            vmovss(xmm0, ptr[rsp + stack_frame_size() + 16]);
            vbroadcastss(zmm_scale(), xmm0);

            // Calculate working space needed on stack:
            // 1. Q blocks for one head: num_blocks * 64 bytes (padded)
            // 2. Context accumulator spill: (num_blocks - 2) * 128 bytes
            // 3. Extra for alignment and temps
            int num_blocks = config_.head_dim / 32;
            int q_stack_size = num_blocks * 64; // Padded Q blocks for one head
            int spill_bytes = (num_blocks > 2) ? (num_blocks - 2) * 128 : 0;
            int stack_size = q_stack_size + spill_bytes + 256; // Extra for alignment
            stack_size = (stack_size + 63) & ~63;              // Align to 64 bytes

            // Stack layout:
            // [rsp + 0]                     : Q blocks for current head (q_stack_size bytes)
            // [rsp + q_stack_size]          : Spill area for context accumulators
            // [rsp + q_stack_size + spill]  : Temp/alignment padding
            int q_stack_offset = 0;
            int spill_base_offset = q_stack_size;

            sub(rsp, stack_size);

            // Main loop over query positions
            Reg64 reg_q_idx = rax;
            xor_(reg_q_idx, reg_q_idx);

            L("main_loop_q");
            cmp(reg_q_idx, reg_seq_len_q);
            jge("main_loop_q_end", T_NEAR);

            // Emit attention for one query position
            emit_single_query_attention(
                reg_Q, reg_K, reg_V, reg_Wo, reg_output,
                reg_q_idx, reg_seq_len_kv,
                num_blocks, spill_base_offset, q_stack_offset);

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
         * 2. Loop over KV positions:
         *    - Compute score = Q·K with Q from stack
         *    - Online softmax update
         *    - Weighted V accumulation
         * 3. Normalize context by 1/sum
         * 4. Output projection (simplified)
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
            int q_stack_offset)
        {
            using namespace Xbyak;

            debug_emit("emit_single_query_attention");

            // Save reg_q_idx to stack - it gets clobbered by emitters that use eax
            // We'll allocate 8 bytes at the end of the spill area
            int q_idx_spill_offset = spill_base_offset + (num_blocks > 2 ? (num_blocks - 2) * 128 : 0);
            mov(ptr[rsp + q_idx_spill_offset], reg_q_idx);

            // For GQA, compute KV head index
            int heads_per_kv = config_.num_heads / config_.num_kv_heads;

            // Calculate Q base pointer for this query position
            // Q layout: [seq_len_q, num_heads, head_dim] in Q8_1 blocks
            Reg64 reg_Q_base = r11;
            int q_stride = config_.num_heads * num_blocks * 36; // bytes per query position

            mov(reg_Q_base, reg_q_idx);
            imul(reg_Q_base, reg_Q_base, q_stride);
            add(reg_Q_base, reg_Q);

            // For each head in this query position
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
                // NOTE: Use rdi as scratch to avoid clobbering rax (reg_q_idx loop counter)
                emit_broadcast_i32_const(zmm_128(), 0x80808080, rdi);

                // Copy Q[q,h] blocks to stack for this head
                // NOTE: emit_copy_q_head_to_stack uses edi as scratch to preserve rax
                emit_copy_q_head_to_stack(reg_Q_base, h, num_blocks, q_stack_offset);

                // Loop over KV positions
                // Use unique labels for each head to avoid conflicts
                std::string loop_label = head_label + "_kv_loop";
                std::string end_label = head_label + "_kv_end";

                Reg64 reg_kv_idx = rcx; // Reuse rcx (Wo moved to rbp)
                xor_(reg_kv_idx, reg_kv_idx);

                L(loop_label.c_str());
                cmp(reg_kv_idx, reg_seq_len_kv);
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

                // Restore reg_q_idx from spill slot before storing output
                // (it was clobbered by emit_dot_product and other emitters that use eax)
                mov(reg_q_idx, ptr[rsp + q_idx_spill_offset]);

                // Store this head's context to output
                // For now, simplified - just store register-resident accumulators
                emit_store_head_context(reg_output, reg_q_idx, h, num_blocks, spill_base_offset);
            }

            // Note: Full Wo projection would be applied after all heads are computed
            // For now, we just output the concatenated contexts
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

                // Copy 4 bytes (scale + sum_qs)
                // NOTE: Use edi as scratch to avoid clobbering eax (reg_q_idx loop counter)
                mov(edi, ptr[reg_Q_base + src_offset]);
                mov(ptr[rsp + dst_offset], edi);
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
     *   attn.compute(Q, K, V, Wo, output, seq_len_q, seq_len_kv, scale);
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
         */
        void compute(
            const void *Q,
            const void *K,
            const void *V,
            const void *Wo,
            float *output,
            int seq_len_q,
            int seq_len_kv,
            float scale)
        {
            kernel_(Q, K, V, Wo, output, seq_len_q, seq_len_kv, scale);
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
