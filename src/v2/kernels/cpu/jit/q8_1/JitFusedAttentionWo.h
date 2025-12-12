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

namespace llaminar::v2::kernels::jit {

/**
 * @brief Configuration for JIT attention kernel
 */
struct JitAttentionConfig {
    int head_dim;       // Dimension per head (e.g., 64)
    int num_heads;      // Number of Q heads
    int num_kv_heads;   // Number of KV heads (GQA support)
    int batch_size;     // Batch size (1 for decode, >1 for prefill)
    WoFormat wo_format; // Output projection weight format
    
    bool operator==(const JitAttentionConfig& other) const {
        return head_dim == other.head_dim &&
               num_heads == other.num_heads &&
               num_kv_heads == other.num_kv_heads &&
               batch_size == other.batch_size &&
               wo_format == other.wo_format;
    }
};

}  // namespace llaminar::v2::kernels::jit

// Hash for JitAttentionConfig (for cache lookup)
namespace std {
template<>
struct hash<llaminar::v2::kernels::jit::JitAttentionConfig> {
    size_t operator()(const llaminar::v2::kernels::jit::JitAttentionConfig& cfg) const {
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

namespace llaminar::v2::kernels::jit {

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
    const void* Q,
    const void* K,
    const void* V,
    const void* Wo,
    float* output,
    int seq_len_q,
    int seq_len_kv,
    float scale
);

/**
 * @brief JIT code generator for fused attention + Wo
 *
 * Generates optimized x86-64 AVX-512 code at runtime.
 */
class JitFusedAttentionWoGenerator : public JitMicrokernelBase {
public:
    explicit JitFusedAttentionWoGenerator(const JitAttentionConfig& config)
        : JitMicrokernelBase(64 * 1024)  // 64KB code buffer
        , config_(config) {
        generate();
    }
    
    /**
     * @brief Get the generated kernel function pointer
     */
    JitAttentionKernelFn getKernel() {
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
    void generate() {
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
        
        // Calculate working space needed on stack
        int num_blocks = config_.head_dim / 32;
        int spill_bytes = (num_blocks > 2) ? (num_blocks - 2) * 128 : 0;
        int stack_size = spill_bytes + 256;  // Extra for alignment and temps
        stack_size = (stack_size + 63) & ~63;  // Align to 64 bytes
        
        sub(rsp, stack_size);
        
        // Main loop over batch * num_heads * seq_len_q
        Label loop_q, loop_q_end;
        Reg64 reg_q_idx = rax;
        xor_(reg_q_idx, reg_q_idx);
        
        L(loop_q);
        cmp(reg_q_idx, reg_seq_len_q);
        jge(loop_q_end);
        
        // Emit attention for one query position
        emit_single_query_attention(
            reg_Q, reg_K, reg_V, reg_Wo, reg_output,
            reg_q_idx, reg_seq_len_kv,
            num_blocks, 0  // spill_base_offset
        );
        
        inc(reg_q_idx);
        jmp(loop_q);
        
        L(loop_q_end);
        
        // Cleanup
        add(rsp, stack_size);
        pop_callee_saved();
        ret();
        
        ready();
    }
    
    /**
     * @brief Emit attention computation for a single query position
     */
    void emit_single_query_attention(
        const Xbyak::Reg64& reg_Q,
        const Xbyak::Reg64& reg_K,
        const Xbyak::Reg64& reg_V,
        const Xbyak::Reg64& reg_Wo,
        const Xbyak::Reg64& reg_output,
        const Xbyak::Reg64& reg_q_idx,
        const Xbyak::Reg64& reg_seq_len_kv,
        int num_blocks,
        int spill_base_offset
    ) {
        using namespace Xbyak;
        
        debug_emit("emit_single_query_attention");
        
        // For GQA, compute KV head index
        int heads_per_kv = config_.num_heads / config_.num_kv_heads;
        
        // Initialize context accumulators (for this head)
        v_accum_emitter_.emit_init_context(*this, num_blocks, spill_base_offset);
        
        // Initialize softmax state
        // max = -FLT_MAX, sum = 0
        load_constant_f32(zmm_max(), -3.4028235e+38f);  // -FLT_MAX
        vxorps(zmm_sum(), zmm_sum(), zmm_sum());
        
        // Calculate Q block pointer for this query
        // Q layout: [seq_len_q, num_heads, head_dim] in Q8_1 blocks
        Reg64 reg_Q_ptr = r11;
        int q_stride = config_.num_heads * num_blocks * 36;  // bytes per query position
        
        mov(reg_Q_ptr, reg_q_idx);
        imul(reg_Q_ptr, q_stride);
        add(reg_Q_ptr, reg_Q);
        
        // Loop over KV positions
        Label loop_kv, loop_kv_end;
        Reg64 reg_kv_idx = rcx;  // Reuse rcx (Wo moved to rbp)
        xor_(reg_kv_idx, reg_kv_idx);
        
        L(loop_kv);
        cmp(reg_kv_idx, reg_seq_len_kv);
        jge(loop_kv_end);
        
        // For each head in this query position
        for (int h = 0; h < config_.num_heads; ++h) {
            int kv_head = h / heads_per_kv;
            
            emit_single_head_attention(
                reg_Q_ptr, reg_K, reg_V,
                reg_kv_idx, h, kv_head,
                num_blocks, spill_base_offset
            );
        }
        
        inc(reg_kv_idx);
        jmp(loop_kv);
        
        L(loop_kv_end);
        
        // Normalize context by 1/sum
        Zmm zmm_inv_sum = zmm_scratch(4);
        load_constant_f32(zmm_scratch(5), 1.0f);
        vdivps(zmm_inv_sum, zmm_scratch(5), zmm_sum());
        v_accum_emitter_.emit_normalize_context(*this, zmm_inv_sum, num_blocks, spill_base_offset);
        
        // Output projection (context * Wo)
        // For now, emit simplified projection (full version would loop over d_model)
        emit_output_projection(reg_output, reg_Wo, reg_q_idx, num_blocks, spill_base_offset);
    }
    
    /**
     * @brief Emit attention for single head at single KV position
     */
    void emit_single_head_attention(
        const Xbyak::Reg64& reg_Q_ptr,
        const Xbyak::Reg64& reg_K,
        const Xbyak::Reg64& reg_V,
        const Xbyak::Reg64& reg_kv_idx,
        int head_idx,
        int kv_head_idx,
        int num_blocks,
        int spill_base_offset
    ) {
        using namespace Xbyak;
        
        // Calculate K pointer for this KV position and head
        // K layout: [seq_len_kv, num_kv_heads, head_dim] in Q8_1 blocks
        Reg64 reg_K_ptr = rdi;  // Temp reuse
        int k_stride = config_.num_kv_heads * num_blocks * 36;
        
        mov(reg_K_ptr, reg_kv_idx);
        imul(reg_K_ptr, k_stride);
        add(reg_K_ptr, kv_head_idx * num_blocks * 36);
        add(reg_K_ptr, reg_K);
        
        // Q[q, h] dot K[kv, kv_h]
        // Use Q8 dot product microkernel
        int q_head_offset = head_idx * num_blocks * 36;
        Reg64 reg_Q_head = rsi;  // Temp reuse
        lea(reg_Q_head, ptr[reg_Q_ptr + q_head_offset]);
        
        dot_emitter_.emit_multi_block_dot(*this, reg_Q_head, reg_K_ptr, num_blocks);
        
        // Score is now in zmm_score() (element 0)
        // Apply scale
        vmulss(xmm0, Xmm(zmm_score().getIdx()), Xmm(zmm_scale().getIdx()));
        vbroadcastss(zmm_score(), xmm0);
        
        // Online softmax update
        softmax_emitter_.emit_softmax_update(*this, zmm_score());
        
        // Calculate V pointer
        Reg64 reg_V_ptr = rdi;
        int v_stride = config_.num_kv_heads * num_blocks * 36;
        
        mov(reg_V_ptr, reg_kv_idx);
        imul(reg_V_ptr, v_stride);
        add(reg_V_ptr, kv_head_idx * num_blocks * 36);
        add(reg_V_ptr, reg_V);
        
        // Weighted V accumulation
        // weight is in zmm_weight() after softmax update
        v_accum_emitter_.emit_weighted_accum(*this, reg_V_ptr, num_blocks, spill_base_offset);
    }
    
    /**
     * @brief Emit output projection (context * Wo -> output)
     */
    void emit_output_projection(
        const Xbyak::Reg64& reg_output,
        const Xbyak::Reg64& reg_Wo,
        const Xbyak::Reg64& reg_q_idx,
        int num_blocks,
        int spill_base_offset
    ) {
        using namespace Xbyak;
        
        debug_emit("emit_output_projection (simplified)");
        
        // Store context to temp buffer first
        // Then call projection emitter
        // This is a simplified version - full version would handle all heads
        
        // For now, just store the first head's context to output
        // Real implementation would loop over d_model output dimensions
        
        int d_model = config_.num_heads * config_.head_dim;
        int out_offset_per_q = d_model * 4;  // FP32 output stride
        
        Reg64 reg_out_ptr = rdi;
        mov(reg_out_ptr, reg_q_idx);
        imul(reg_out_ptr, out_offset_per_q);
        add(reg_out_ptr, reg_output);
        
        // Store register-resident context
        vmovups(ptr[reg_out_ptr], zmm_accum(0));
        vmovups(ptr[reg_out_ptr + 64], zmm_accum(1));
        
        if (num_blocks >= 2) {
            vmovups(ptr[reg_out_ptr + 128], zmm_accum(2));
            vmovups(ptr[reg_out_ptr + 192], zmm_accum(3));
        }
        
        // Store spilled context
        if (num_blocks > 2) {
            for (int b = 2; b < num_blocks; ++b) {
                int spill_lo = spill_base_offset + (b - 2) * 128;
                int spill_hi = spill_lo + 64;
                int out_lo = b * 64 * 2;
                int out_hi = out_lo + 64;
                
                vmovups(zmm_scratch(0), ptr[rsp + spill_lo]);
                vmovups(ptr[reg_out_ptr + out_lo], zmm_scratch(0));
                
                vmovups(zmm_scratch(0), ptr[rsp + spill_hi]);
                vmovups(ptr[reg_out_ptr + out_hi], zmm_scratch(0));
            }
        }
        
        // Note: Full Wo projection would be:
        // for each output_dim d:
        //   output[q, d] = sum over h: context[q, h, :] dot Wo[d, h*head_dim : (h+1)*head_dim]
        // This is deferred to integration with the full pipeline
    }
};

/**
 * @brief Cache for JIT-generated attention kernels
 *
 * Thread-safe cache that stores generated code to avoid regeneration.
 */
class JitAttentionKernelCache {
public:
    static JitAttentionKernelCache& instance() {
        static JitAttentionKernelCache inst;
        return inst;
    }
    
    /**
     * @brief Get or generate kernel for config
     *
     * @param config Kernel configuration
     * @return Function pointer to generated kernel
     */
    JitAttentionKernelFn getKernel(const JitAttentionConfig& config) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = cache_.find(config);
        if (it != cache_.end()) {
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
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
    }
    
    /**
     * @brief Get cache statistics
     */
    size_t size() const {
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
class JitFusedAttentionWo {
public:
    explicit JitFusedAttentionWo(const JitAttentionConfig& config)
        : config_(config)
        , kernel_(JitAttentionKernelCache::instance().getKernel(config)) {
    }
    
    /**
     * @brief Execute fused attention + Wo projection
     */
    void compute(
        const void* Q,
        const void* K,
        const void* V,
        const void* Wo,
        float* output,
        int seq_len_q,
        int seq_len_kv,
        float scale
    ) {
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

}  // namespace llaminar::v2::kernels::jit
