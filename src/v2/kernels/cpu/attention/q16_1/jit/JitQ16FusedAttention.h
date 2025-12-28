/**
 * @file JitQ16FusedAttention.h
 * @brief JIT Fused Attention kernel for Q16 Integer Attention
 * @author David Sanftenberg
 * @date December 2025
 *
 * Composes all Q16 microkernels into a single fused attention kernel with
 * FA2-style tiling for prefill and streaming decode for single-token inference.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * ARCHITECTURE: ALL-INTEGER PIPELINE (NO FP32 INTERMEDIATES)
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 *   Q, K, V (Q16_1)
 *       │
 *       ▼
 *   STEP 1: Q×K^T (JitQ16DotProduct - vpmaddwd INT16×INT16→INT32)
 *       │ INT32 scores
 *       ▼
 *   STEP 2: Softmax (JitExp2FixedSoftmax - 256-entry LUT)
 *       │ INT16 attention weights [0, 32767]
 *       ▼
 *   STEP 3: P×V Accumulation (JitPVAccumulate - INT64 accumulators)
 *       │ INT64 context accumulators
 *       ▼
 *   STEP 4: Wo Projection (JitWoProjectionVNNI - VPDPWSSD)
 *       │ INT32 → requant → Q16_1 output
 *       ▼
 *   STEP 5: Q16_1 Residual Add (simd::q16_1_add_q16_1)
 *       │ Q16_1 output (all integer!)
 *       ▼
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * EXECUTION MODES
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * DECODE (seq_len_q = 1):
 *   - Flash Decode: Stream through KV cache, Q pinned in registers
 *   - Memory-bound, latency-critical
 *   - No Q tiling, process all KV positions
 *
 * PREFILL (seq_len_q > 1):
 *   - FA2-style tiling: [Br × Bc] tiles
 *   - Compute-bound, GEMM-centric
 *   - Online softmax for numerical stability
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * CACHE HIERARCHY ASSIGNMENT
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 *   L1 (32KB):  Current K/V blocks, Exp2 LUT (256B), scores/weights
 *   L2 (256KB): Q tile, K/V tiles, O accumulators, S/P tiles
 *   L3/DRAM:    Full KV cache, Wo matrix, residual
 *
 * @see Q16RegisterAllocation.h for register zone definitions
 * @see microkernels/ for individual JIT emitters
 */

#pragma once

#include "Q16RegisterAllocation.h"
#include "microkernels/JitQ16DotProduct.h"
#include "microkernels/JitExp2FixedSoftmax.h"
#include "microkernels/JitPVAccumulate.h"
#include "microkernels/JitWoProjectionVNNI.h"
#include "tensors/BlockStructures.h"
#include "../../../jit/JitMicrokernelBase.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <functional>

namespace llaminar2::kernels::q16_1::jit
{

    // ============================================================================
    // Configuration Structures
    // ============================================================================

    /**
     * @brief Configuration for JIT Q16 fused attention kernel
     */
    struct JitQ16AttentionConfig
    {
        // Tensor dimensions
        int seq_len_q = 1;    ///< Number of query positions (1 for decode)
        int kv_len = 0;       ///< Number of KV positions
        int num_heads = 0;    ///< Number of query heads
        int num_kv_heads = 0; ///< Number of KV heads (for GQA)
        int head_dim = 64;    ///< Dimension per head (typically 64 or 128)
        int d_model = 0;      ///< Model dimension (num_heads * head_dim)

        // Attention config
        float attention_scale = 0.0f; ///< 1/sqrt(head_dim), computed if 0
        bool causal = true;           ///< Apply causal masking

        // Tiling config (computed dynamically if 0)
        int Br = 0;            ///< Query tile size (FA2 prefill, default: 16)
        int Bc = 0;            ///< KV tile size (FA2 prefill, default: 64)
        int kv_micro_tile = 4; ///< KV positions per micro-iteration

        // Runtime options
        bool use_jit = true;         ///< Use JIT kernel (false = scalar reference)
        bool enable_prefetch = true; ///< Enable software prefetch

        // ─────────────────────────────────────────────────────────────────────────
        // Derived Properties
        // ─────────────────────────────────────────────────────────────────────────

        bool is_decode() const { return seq_len_q == 1; }
        bool is_prefill() const { return seq_len_q > 1; }
        bool use_gqa() const { return num_kv_heads > 0 && num_kv_heads < num_heads; }

        float get_attention_scale() const
        {
            if (attention_scale > 0.0f)
                return attention_scale;
            return 1.0f / std::sqrt(static_cast<float>(head_dim));
        }

        int get_kv_head(int query_head) const
        {
            if (num_kv_heads == 0 || num_heads == 0)
                return 0;
            return query_head / (num_heads / num_kv_heads);
        }

        int blocks_per_head() const
        {
            return (head_dim + Q16_1Block::BLOCK_SIZE - 1) / Q16_1Block::BLOCK_SIZE;
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Tiling Configuration
        // ─────────────────────────────────────────────────────────────────────────

        /**
         * @brief Get effective tile configuration (compute defaults if needed)
         */
        Q16TileConfig get_tile_config(size_t l2_size = 256 * 1024) const
        {
            if (Br > 0 && Bc > 0)
            {
                Q16TileConfig cfg;
                cfg.Br = Br;
                cfg.Bc = Bc;
                cfg.kv_micro_tile = kv_micro_tile;
                cfg.head_dim = head_dim;
                return cfg;
            }
            return Q16TileConfig::compute(head_dim, l2_size);
        }

        /**
         * @brief Get prefetch configuration
         */
        Q16PrefetchConfig get_prefetch_config() const
        {
            return Q16PrefetchConfig::compute(head_dim);
        }
    };

    /**
     * @brief Runtime parameters for JIT Q16 fused attention execution
     */
    struct JitQ16AttentionParams
    {
        // Input tensors (Q16_1)
        const Q16_1Block *Q = nullptr; ///< Query [seq_len_q × num_heads × blocks_per_head]
        const Q16_1Block *K = nullptr; ///< Key   [kv_len × num_kv_heads × blocks_per_head]
        const Q16_1Block *V = nullptr; ///< Value [kv_len × num_kv_heads × blocks_per_head]

        // Wo projection weights (Q8_0 packed, sign-extended at runtime)
        const int8_t *Wo_weights = nullptr; ///< Wo [d_model × d_model] as Q8_0 blocks
        const float *Wo_scales = nullptr;   ///< Wo block scales

        // Residual tensors (Q16_1, input/output)
        const Q16_1Block *residual_in = nullptr; ///< Residual input [seq_len_q × d_model/32]
        Q16_1Block *residual_out = nullptr;      ///< Residual output [seq_len_q × d_model/32]

        // Tensor strides (in blocks)
        int q_stride_seq = 0;        ///< Q stride between sequence positions
        int q_stride_head = 0;       ///< Q stride between heads
        int kv_stride_seq = 0;       ///< K/V stride between sequence positions
        int kv_stride_head = 0;      ///< K/V stride between heads
        int residual_stride_seq = 0; ///< Residual stride between sequence positions

        // Runtime config
        int position_offset = 0; ///< Position offset for causal masking
    };

    // ============================================================================
    // JIT Kernel Base Class
    // ============================================================================

    /**
     * @brief Base class for JIT Q16 attention kernel
     *
     * Provides infrastructure for generating and caching JIT code, with support
     * for both decode (GEMV) and prefill (GEMM) execution paths.
     */
    class JitQ16AttentionKernelBase
    {
    public:
        explicit JitQ16AttentionKernelBase(const JitQ16AttentionConfig &config)
            : config_(config), tile_config_(config.get_tile_config()), prefetch_config_(config.get_prefetch_config())
        {
            validate_config();
        }

        virtual ~JitQ16AttentionKernelBase() = default;

        // Non-copyable (JIT code is expensive to generate)
        JitQ16AttentionKernelBase(const JitQ16AttentionKernelBase &) = delete;
        JitQ16AttentionKernelBase &operator=(const JitQ16AttentionKernelBase &) = delete;

        /**
         * @brief Execute the fused attention pipeline
         * @param params Runtime parameters
         * @return true on success
         */
        virtual bool compute(const JitQ16AttentionParams &params) = 0;

    protected:
        JitQ16AttentionConfig config_;
        Q16TileConfig tile_config_;
        Q16PrefetchConfig prefetch_config_;

        void validate_config()
        {
            // Validate dimensions
            if (config_.head_dim <= 0 || config_.head_dim > 256)
            {
                throw std::invalid_argument("Invalid head_dim: must be in (0, 256]");
            }
            if (config_.num_heads <= 0)
            {
                throw std::invalid_argument("Invalid num_heads: must be > 0");
            }
            if (config_.kv_len < 0)
            {
                throw std::invalid_argument("Invalid kv_len: must be >= 0");
            }
            // head_dim must be multiple of 32 for Q16_1 block alignment
            if (config_.head_dim % 32 != 0)
            {
                throw std::invalid_argument("head_dim must be multiple of 32 for Q16_1");
            }
        }
    };

    // ============================================================================
    // Flash Decode Kernel (seq_len_q = 1)
    // ============================================================================

    /**
     * @brief JIT kernel for Flash Decode mode (single query)
     *
     * Streaming decode through KV cache with Q pinned in registers.
     * Memory-bound, optimized for latency.
     *
     * REGISTER FLOW:
     *   1. Load Q → Input0-3 (persistent)
     *   2. Clear Accum0-7 (INT64 accumulators)
     *   3. KV Loop:
     *      a. Load K[kv:kv+4] → Input4-7
     *      b. Q×K^T → Scratch0-3 (scores)
     *      c. Exp2FixedSoftmax → Scratch0-3 (weights)
     *      d. Load V[kv:kv+4] → Input4-7
     *      e. P×V → Accum0-7 (accumulate)
     *   4. Normalize Accum0-7 by weight_sum → INT32 context
     *   5. Wo projection → Q16_1 to memory
     *   6. Q16_1 + Q16_1 residual add
     */
    class JitQ16FlashDecodeKernel : public JitQ16AttentionKernelBase
    {
    public:
        using JitQ16AttentionKernelBase::JitQ16AttentionKernelBase;

        bool compute(const JitQ16AttentionParams &params) override;

    private:
        // ─────────────────────────────────────────────────────────────────────────
        // JIT Code Generation (called once, cached)
        // ─────────────────────────────────────────────────────────────────────────

        /**
         * @brief Generate JIT code for decode kernel
         */
        void generate_jit_code();

        /**
         * @brief Emit constant initialization (zmm26-31)
         */
        void emit_init_constants(Xbyak::CodeGenerator &gen);

        /**
         * @brief Emit Q vector load → Input0-3
         */
        void emit_load_q_vector(Xbyak::CodeGenerator &gen);

        /**
         * @brief Emit KV loop body (process kv_micro_tile positions)
         */
        void emit_kv_loop_body(Xbyak::CodeGenerator &gen, int kv_offset);

        /**
         * @brief Emit finalization (normalize, Wo projection, residual add)
         */
        void emit_finalization(Xbyak::CodeGenerator &gen);

        // ─────────────────────────────────────────────────────────────────────────
        // Reference Implementation (fallback)
        // ─────────────────────────────────────────────────────────────────────────

        /**
         * @brief Reference implementation for testing
         */
        bool compute_reference(const JitQ16AttentionParams &params);

        // ─────────────────────────────────────────────────────────────────────────
        // Generated Code Cache
        // ─────────────────────────────────────────────────────────────────────────

        using KernelFunc = void (*)(const JitQ16AttentionParams *);
        KernelFunc jit_kernel_ = nullptr;
        std::unique_ptr<Xbyak::CodeGenerator> code_gen_;
    };

    // ============================================================================
    // FA2 Prefill Kernel (seq_len_q > 1)
    // ============================================================================

    /**
     * @brief JIT kernel for FA2-style prefill mode (multiple queries)
     *
     * Tiled execution with [Br × Bc] tiles for compute efficiency.
     * Compute-bound, GEMM-centric.
     *
     * TILING STRATEGY:
     *   Outer loop: Q tiles (Br queries at a time)
     *   Inner loop: KV tiles (Bc keys at a time)
     *
     * REGISTER FLOW (per Q row):
     *   1. Load Q[q_local] → Input0-3
     *   2. For each KV micro-tile:
     *      a. Load K[kv:kv+4] → Input4-7
     *      b. Q×K^T → Scratch0-3 (scores)
     *      c. Exp2FixedSoftmax → Scratch0-3 (weights) + update StateZone
     *      d. Load V[kv:kv+4] → Input4-7
     *      e. P×V → Accum0-7 (accumulate)
     *   3. Store Accum0-7 → O_acc[q_local] in L2 buffer
     *
     * FINALIZATION (per Q tile):
     *   1. Normalize O_acc by l (from StateZone)
     *   2. Wo projection (batched or row-by-row)
     *   3. Q16_1 residual add
     */
    class JitQ16FA2PrefillKernel : public JitQ16AttentionKernelBase
    {
    public:
        using JitQ16AttentionKernelBase::JitQ16AttentionKernelBase;

        bool compute(const JitQ16AttentionParams &params) override;

    private:
        // ─────────────────────────────────────────────────────────────────────────
        // JIT Code Generation
        // ─────────────────────────────────────────────────────────────────────────

        /**
         * @brief Generate JIT code for prefill kernel
         */
        void generate_jit_code();

        /**
         * @brief Emit outer Q tile loop
         */
        void emit_q_tile_loop(Xbyak::CodeGenerator &gen);

        /**
         * @brief Emit inner KV tile loop
         */
        void emit_kv_tile_loop(Xbyak::CodeGenerator &gen);

        /**
         * @brief Emit micro-tile GEMM (4×4 blocking)
         */
        void emit_micro_tile_gemm(Xbyak::CodeGenerator &gen);

        /**
         * @brief Emit online softmax rescaling
         */
        void emit_online_softmax_rescale(Xbyak::CodeGenerator &gen);

        // ─────────────────────────────────────────────────────────────────────────
        // Reference Implementation
        // ─────────────────────────────────────────────────────────────────────────

        bool compute_reference(const JitQ16AttentionParams &params);

        // ─────────────────────────────────────────────────────────────────────────
        // L2 Buffers (allocated for tile working set)
        // ─────────────────────────────────────────────────────────────────────────

        std::vector<int64_t> o_acc_buffer_;     ///< O accumulators [Br × head_dim]
        std::vector<int32_t> state_max_buffer_; ///< Running max [Br]
        std::vector<int64_t> state_sum_buffer_; ///< Running sum [Br]
    };

    // ============================================================================
    // Factory Function
    // ============================================================================

    /**
     * @brief Create appropriate JIT kernel based on configuration
     */
    inline std::unique_ptr<JitQ16AttentionKernelBase> create_jit_q16_attention_kernel(
        const JitQ16AttentionConfig &config)
    {
        if (config.is_decode())
        {
            return std::make_unique<JitQ16FlashDecodeKernel>(config);
        }
        else
        {
            return std::make_unique<JitQ16FA2PrefillKernel>(config);
        }
    }

    // ============================================================================
    // Inline Implementation Stubs
    // ============================================================================

    inline bool JitQ16FlashDecodeKernel::compute(const JitQ16AttentionParams &params)
    {
        // TODO: Implement JIT path
        // if (jit_kernel_) {
        //     jit_kernel_(&params);
        //     return true;
        // }
        return compute_reference(params);
    }

    inline bool JitQ16FlashDecodeKernel::compute_reference(const JitQ16AttentionParams &params)
    {
        // Reference implementation delegates to individual microkernel references
        // This will be filled in when microkernels are implemented
        (void)params;
        return false; // Not yet implemented
    }

    inline bool JitQ16FA2PrefillKernel::compute(const JitQ16AttentionParams &params)
    {
        // TODO: Implement JIT path
        return compute_reference(params);
    }

    inline bool JitQ16FA2PrefillKernel::compute_reference(const JitQ16AttentionParams &params)
    {
        (void)params;
        return false; // Not yet implemented
    }

} // namespace llaminar2::kernels::q16_1::jit
