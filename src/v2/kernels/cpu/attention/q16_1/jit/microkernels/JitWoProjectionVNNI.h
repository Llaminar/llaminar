// SPDX-License-Identifier: MIT
// Llaminar V2 - Q16 Integer Attention JIT Microkernels
// JitWoProjectionVNNI.h - Wo projection using VPDPWSSD (INT16×INT16→INT32)

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <cmath>
#include "../Q16RegisterAllocation.h"
#include "tensors/BlockStructures.h"

namespace llaminar2::kernels::q16_1::jit
{

    // ============================================================================
    // Wo Projection Microkernel Configuration
    // ============================================================================

    /**
     * @brief Configuration for Wo projection microkernel
     *
     * The Wo projection implements: proj[d_model] = context[inner_dim] × Wo^T
     *
     * Key design choices from project plan:
     * - INT32 context → INT16 requant → VPDPWSSD (preserves 16-bit precision)
     * - INT8 weights sign-extended to INT16 at runtime (vpmovsxbw, essentially free)
     * - Output requantized to Q16_1 for native residual add
     */
    struct JitWoProjectionVNNIConfig
    {
        size_t d_model = 896;              // Output dimension (model hidden size)
        size_t inner_dim = 896;            // Input dimension (num_heads × head_dim)
        size_t tile_n = 32;                // Output tile size (matches Q16_1 block)
        size_t tile_k = 64;                // Inner dimension tile for L1 cache fit
        bool use_streaming_weights = true; // Stream Wo from L3/DRAM
        bool fuse_residual_add = false;    // Fuse with Q16_1 + Q16_1 residual
    };

    /**
     * @brief Parameters for Wo projection microkernel at runtime
     */
    struct JitWoProjectionVNNIParams
    {
        const int32_t *context_int32;  // INT32 P×V output [inner_dim]
        const int8_t *wo_weights_int8; // INT8 packed Wo [d_model × inner_dim]
        int16_t *output_int32;         // INT32 output before requant [d_model]

        // Scale tracking
        float context_scale; // Scale used to requant context to INT16
        float wo_scale;      // Wo weight scale (per-tensor or per-row)
        float *output_scale; // Combined scale for final requant

        // Optional: Q16_1 output and residual for fused operation
        void *output_q16_1;         // Q16_1Block output [d_model / 32]
        const void *residual_q16_1; // Q16_1Block residual [d_model / 32]

        size_t d_model;   // Output dimension
        size_t inner_dim; // Input dimension
    };

    // ============================================================================
    // Q16_1 Block Structure (for output)
    // ============================================================================

    /**
     * @brief Q16_1 block for Wo projection output (72 bytes, 32 INT16 elements)
     */
    struct Q16_1OutputBlock
    {
        float d;        // Scale factor
        int32_t sum_qs; // Sum of quantized values (for faster dequant)
        int16_t qs[32]; // Quantized values
    };
    static_assert(sizeof(Q16_1OutputBlock) == 72, "Q16_1OutputBlock must be 72 bytes");

    // ============================================================================
    // JIT Wo Projection Emitter Class
    // ============================================================================

    /**
     * @brief JIT emitter for Wo projection microkernel
     *
     * REGISTER CONTRACT (from project plan):
     *
     * INPUTS:
     * - Input0-7 (zmm8-15): INT16 context (requantized from INT32)
     *   - For inner_dim > 128, reload in tiles
     *   - Concatenated from all heads: [num_heads × head_dim]
     *
     * STREAMING (from L3/DRAM):
     * - Scratch0-3 (zmm20-23): Wo weight rows (INT8 → sign-extend to INT16)
     *   - One output element computed per row of Wo
     *
     * SCRATCH:
     * - Scratch4-5 (zmm24-25): Temporaries for sign-extension, FMA intermediates
     * - Accum4-7 (zmm4-7): Additional accumulators for 8-way unroll
     *
     * CONSTANTS:
     * - Const128 (zmm26): Constant 128 for bias adjustment (if needed)
     *
     * OUTPUT:
     * - INT32 accumulators → requant to Q16_1 → write to memory
     */
    class JitWoProjectionVNNIEmitter
    {
    public:
        using Config = JitWoProjectionVNNIConfig;

        explicit JitWoProjectionVNNIEmitter(const Config &config) : config_(config) {}

        // ========================================================================
        // JIT Emission Methods (stubs - to be implemented with Xbyak)
        // ========================================================================

        /**
         * @brief Emit code to load and requantize INT32 context to INT16
         *
         * This converts P×V INT32 output to INT16 for VPDPWSSD.
         * We preserve 16 bits of precision (vs only 8 for INT8 requant).
         *
         * Register usage:
         * - Input: rdi = context_int32 ptr, rsi = inner_dim, rdx = context_scale ptr
         * - Output: Input0-3 (zmm8-11) loaded with INT16 context (first 128 elements)
         */
        void emit_load_context_int16()
        {
            // TODO: JIT emit with Xbyak
            // 1. Load INT32 context values (16 per zmm)
            // 2. Find max for dynamic scaling (or use pre-computed scale)
            // 3. Multiply by scale, round to nearest
            // 4. Pack INT32 → INT16 using vpackssdw
            // 5. Store in Input0-3 registers
        }

        /**
         * @brief Emit code to sign-extend INT8 Wo weights to INT16
         *
         * The weight sign-extension is essentially free using vpmovsxbw.
         * For each Wo row, we load INT8 and sign-extend to INT16 on the fly.
         *
         * Register usage:
         * - Input: rdi = wo_weights ptr (INT8), rsi = row index
         * - Output: Scratch0-1 (zmm20-21) with INT16 weights (32 elements each)
         */
        void emit_sign_extend_weights(size_t k_tile_start, size_t k_tile_size)
        {
            // TODO: JIT emit with Xbyak
            // For each 32-byte chunk of INT8 weights:
            // 1. vmovdqu ymm_src, [wo_ptr + k_offset] ; Load 32 INT8 values
            // 2. vpmovsxbw zmm_dst, ymm_src          ; Sign-extend to 32 INT16
            (void)k_tile_start;
            (void)k_tile_size;
        }

        /**
         * @brief Emit VPDPWSSD dot product for one output element
         *
         * Computes: acc += context_int16 · weight_int16 (INT32 accumulator)
         *
         * This is the core operation: proj[n] = sum_k(context[k] × Wo[n,k])
         *
         * Register usage:
         * - Input: Input0-3 (context INT16), Scratch0-1 (weight INT16)
         * - Accumulator: Accum0 (zmm0) - horizontal sum at end
         */
        void emit_vpdpwssd_row(size_t n_idx)
        {
            // TODO: JIT emit with Xbyak
            // For each 32-element chunk of the inner dimension:
            // 1. vpdpwssd zmm_acc, zmm_ctx, zmm_wgt ; INT16×INT16→INT32 accumulate
            // 2. After all chunks: horizontal sum to scalar INT32
            (void)n_idx;
        }

        /**
         * @brief Emit complete Wo projection with GEMV loop
         *
         * For decode (single query): standard GEMV streaming Wo rows
         * For prefill (batched): tiled GEMM for Wo weight reuse
         */
        void emit_wo_projection_gemv()
        {
            // TODO: JIT emit with Xbyak
            // Loop structure:
            // for n in [0, d_model):
            //     acc = 0
            //     emit_sign_extend_weights(row=n)
            //     for k_tile in [0, inner_dim, tile_k):
            //         emit_vpdpwssd_row(k_tile)
            //     horizontal_sum(acc) → output[n]
        }

        /**
         * @brief Emit INT32 output requantization to Q16_1 blocks
         *
         * Converts INT32 GEMM output to Q16_1 format for residual add.
         *
         * Algorithm:
         * 1. Find max_abs across 32-element block
         * 2. Compute scale = max_abs / 32767
         * 3. Requantize: qs[i] = round(val[i] / scale)
         * 4. Store Q16_1 block (scale + sum_qs + qs[32])
         */
        void emit_requantize_to_q16_1()
        {
            // TODO: JIT emit with Xbyak
            // Per 32-element block:
            // 1. vpmaxsd + horizontal max for max_abs
            // 2. Compute scale (may need FP32 divide)
            // 3. vcvtdq2ps + vmulps + vrndscaleps + vcvtps2dq for requant
            // 4. vpackssdw to pack INT32 → INT16
            // 5. Store Q16_1 block
        }

        /**
         * @brief Emit fused Q16_1 residual add (optional)
         *
         * If fuse_residual_add is enabled, performs:
         * output_q16 = projection_q16 + residual_q16
         *
         * Uses simd::q16_1_add_q16_1() pattern from SIMDHelpers.h
         */
        void emit_fused_residual_add()
        {
            // TODO: JIT emit with Xbyak if fused
            // Per block:
            // 1. Load projection Q16_1 block
            // 2. Load residual Q16_1 block
            // 3. Dequant both, add, find new max, requant
            // 4. Store result
        }

        // ========================================================================
        // Reference Implementation (for testing/validation)
        // ========================================================================

        /**
         * @brief Reference implementation of FP32 context to INT16 requantization
         *
         * This is for testing - actual attention uses INT32 context from P×V.
         *
         * @param fp32_context Input FP32 values [count]
         * @param int16_out Output INT16 values [count]
         * @param count Number of elements
         * @param scale_out Output scale factor
         */
        static void requantize_fp32_to_int16(
            const float *fp32_context,
            int16_t *int16_out,
            size_t count,
            float *scale_out)
        {
            // Find max absolute value
            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                float abs_val = std::abs(fp32_context[i]);
                if (abs_val > max_abs)
                    max_abs = abs_val;
            }

            // Compute scale (FP32 range to INT16 range)
            float scale = (max_abs > 0.0f) ? max_abs / 32767.0f : 1.0f;
            *scale_out = scale;

            // Requantize
            for (size_t i = 0; i < count; ++i)
            {
                float normalized = fp32_context[i] / scale;
                int16_out[i] = static_cast<int16_t>(std::round(normalized));
            }
        }

        /**
         * @brief Reference implementation of INT32 context to INT16 requantization
         *
         * @param context_int32 Input INT32 values [inner_dim]
         * @param inner_dim Number of elements
         * @param context_int16 Output INT16 values [inner_dim]
         * @param scale_out Output scale factor
         */
        static void requantize_context_reference(
            const int32_t *context_int32,
            size_t inner_dim,
            int16_t *context_int16,
            float *scale_out)
        {
            // Find max absolute value
            int32_t max_abs = 0;
            for (size_t i = 0; i < inner_dim; ++i)
            {
                int32_t abs_val = std::abs(context_int32[i]);
                if (abs_val > max_abs)
                    max_abs = abs_val;
            }

            // Compute scale (INT32 range to INT16 range)
            // Using 32767 as max INT16 positive value
            float scale = (max_abs > 0) ? static_cast<float>(max_abs) / 32767.0f : 1.0f;
            *scale_out = scale;

            // Requantize
            for (size_t i = 0; i < inner_dim; ++i)
            {
                float normalized = static_cast<float>(context_int32[i]) / scale;
                context_int16[i] = static_cast<int16_t>(std::round(normalized));
            }
        }

        /**
         * @brief Reference implementation of Wo projection GEMV
         *
         * @param context_int16 INT16 context [inner_dim]
         * @param wo_int8 INT8 packed Wo weights [d_model × inner_dim]
         * @param output_int32 INT32 output [d_model]
         * @param d_model Output dimension
         * @param inner_dim Input dimension (K)
         */
        static void wo_gemv_reference(
            const int16_t *context_int16,
            const int8_t *wo_int8,
            int32_t *output_int32,
            size_t d_model,
            size_t inner_dim)
        {
            // Standard GEMV with INT8 sign-extension
            for (size_t n = 0; n < d_model; ++n)
            {
                int32_t acc = 0;
                for (size_t k = 0; k < inner_dim; ++k)
                {
                    // Sign-extend INT8 to INT16 (this is what vpmovsxbw does)
                    int16_t w_int16 = static_cast<int16_t>(wo_int8[n * inner_dim + k]);
                    // INT16 × INT16 → INT32 accumulate (this is what vpdpwssd does)
                    acc += static_cast<int32_t>(context_int16[k]) * static_cast<int32_t>(w_int16);
                }
                output_int32[n] = acc;
            }
        }

        /**
         * @brief Reference implementation of INT32 to Q16_1 requantization
         *
         * @param input_int32 INT32 values [count]
         * @param count Number of elements (must be multiple of 32)
         * @param output_blocks Q16_1 output blocks [count / 32]
         */
        static void requantize_to_q16_1_reference(
            const int32_t *input_int32,
            size_t count,
            Q16_1OutputBlock *output_blocks)
        {
            const size_t num_blocks = (count + 31) / 32;

            for (size_t b = 0; b < num_blocks; ++b)
            {
                const size_t block_start = b * 32;
                const size_t block_end = std::min(block_start + 32, count);

                // Find max abs for this block
                int32_t max_abs = 0;
                for (size_t i = block_start; i < block_end; ++i)
                {
                    int32_t abs_val = std::abs(input_int32[i]);
                    if (abs_val > max_abs)
                        max_abs = abs_val;
                }

                // Compute scale
                float scale = (max_abs > 0) ? static_cast<float>(max_abs) / 32767.0f : 1.0f;
                output_blocks[b].d = scale;

                // Requantize and compute sum
                int32_t sum_qs = 0;
                for (size_t i = block_start; i < block_end; ++i)
                {
                    float normalized = static_cast<float>(input_int32[i]) / scale;
                    int16_t q = static_cast<int16_t>(std::round(normalized));
                    output_blocks[b].qs[i - block_start] = q;
                    sum_qs += q;
                }
                // Zero-pad if block not full
                for (size_t i = block_end - block_start; i < 32; ++i)
                {
                    output_blocks[b].qs[i] = 0;
                }
                output_blocks[b].sum_qs = sum_qs;
            }
        }

        /**
         * @brief Complete reference Wo projection: INT32 context → Q16_1 output
         *
         * @param params Runtime parameters
         */
        static void compute_reference(const JitWoProjectionVNNIParams &params)
        {
            // Step 1: Requantize INT32 context to INT16
            std::vector<int16_t> context_int16(params.inner_dim);
            float context_scale;
            requantize_context_reference(
                params.context_int32, params.inner_dim,
                context_int16.data(), &context_scale);

            // Step 2: GEMV with INT8 weights (sign-extended to INT16)
            std::vector<int32_t> gemv_output(params.d_model);
            wo_gemv_reference(
                context_int16.data(), params.wo_weights_int8,
                gemv_output.data(), params.d_model, params.inner_dim);

            // Step 3: Requantize INT32 → Q16_1
            if (params.output_q16_1)
            {
                requantize_to_q16_1_reference(
                    gemv_output.data(), params.d_model,
                    reinterpret_cast<Q16_1OutputBlock *>(params.output_q16_1));
            }

            // Track combined scale for downstream
            if (params.output_scale)
            {
                *params.output_scale = context_scale * params.wo_scale;
            }
        }

    private:
        Config config_;
    };

    // ============================================================================
    // Helper: INT16 Context Buffer (for holding requantized context)
    // ============================================================================

    /**
     * @brief Aligned buffer for INT16 context after requantization
     */
    class Int16ContextBuffer
    {
    public:
        explicit Int16ContextBuffer(size_t inner_dim)
            : size_(inner_dim), scale_(1.0f)
        {
            // 64-byte aligned for AVX-512
            data_ = static_cast<int16_t *>(
                std::aligned_alloc(64, ((inner_dim * sizeof(int16_t) + 63) / 64) * 64));
        }

        ~Int16ContextBuffer()
        {
            if (data_)
                std::free(data_);
        }

        // Non-copyable
        Int16ContextBuffer(const Int16ContextBuffer &) = delete;
        Int16ContextBuffer &operator=(const Int16ContextBuffer &) = delete;

        // Movable
        Int16ContextBuffer(Int16ContextBuffer &&other) noexcept
            : data_(other.data_), size_(other.size_), scale_(other.scale_)
        {
            other.data_ = nullptr;
        }

        Int16ContextBuffer &operator=(Int16ContextBuffer &&other) noexcept
        {
            if (this != &other)
            {
                if (data_)
                    std::free(data_);
                data_ = other.data_;
                size_ = other.size_;
                scale_ = other.scale_;
                other.data_ = nullptr;
            }
            return *this;
        }

        int16_t *data() { return data_; }
        const int16_t *data() const { return data_; }
        size_t size() const { return size_; }

        float scale() const { return scale_; }
        void set_scale(float s) { scale_ = s; }

        /**
         * @brief Fill from INT32 context with requantization
         */
        void from_int32(const int32_t *context_int32)
        {
            JitWoProjectionVNNIEmitter::requantize_context_reference(
                context_int32, size_, data_, &scale_);
        }

    private:
        int16_t *data_ = nullptr;
        size_t size_;
        float scale_;
    };

    // ============================================================================
    // Helper: Q16_1 Output Buffer
    // ============================================================================

    /**
     * @brief Aligned buffer for Q16_1 projection output
     */
    class Q16_1ProjectionBuffer
    {
    public:
        explicit Q16_1ProjectionBuffer(size_t d_model)
            : num_blocks_((d_model + 31) / 32)
        {
            // 64-byte aligned
            blocks_ = static_cast<Q16_1OutputBlock *>(
                std::aligned_alloc(64, num_blocks_ * sizeof(Q16_1OutputBlock)));
        }

        ~Q16_1ProjectionBuffer()
        {
            if (blocks_)
                std::free(blocks_);
        }

        // Non-copyable
        Q16_1ProjectionBuffer(const Q16_1ProjectionBuffer &) = delete;
        Q16_1ProjectionBuffer &operator=(const Q16_1ProjectionBuffer &) = delete;

        Q16_1OutputBlock *blocks() { return blocks_; }
        const Q16_1OutputBlock *blocks() const { return blocks_; }
        size_t num_blocks() const { return num_blocks_; }

        /**
         * @brief Fill from INT32 GEMV output with requantization
         */
        void from_int32(const int32_t *gemv_output, size_t count)
        {
            JitWoProjectionVNNIEmitter::requantize_to_q16_1_reference(
                gemv_output, count, blocks_);
        }

    private:
        Q16_1OutputBlock *blocks_ = nullptr;
        size_t num_blocks_;
    };

} // namespace llaminar2::kernels::q16_1::jit
