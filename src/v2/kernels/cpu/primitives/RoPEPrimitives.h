/**
 * @file RoPEPrimitives.h
 * @brief Vectorized RoPE primitives for V2 (ported from V1)
 * @author David Sanftenberg
 *
 * High-performance RoPE implementation with:
 * - AVX512/AVX2 vectorization (8-16× speedup)
 * - Persistent thread-local state for decode
 * - Complex recurrence for angle advancement
 * - Inverse frequency caching
 * - Angle recurrence across tokens in prefill
 * - Q8_1 pure-integer RoPE (no FP32 round-trips)
 */
#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include "../../../tensors/BlockStructures.h" // For Q8_1Block

namespace llaminar2::primitives
{

    /**
     * @brief Persistent state for single-token decode optimization
     */
    struct RoPEPersistentState
    {
        int cached_head_dim = -1;
        float cached_freq_base = -1.0f;
        int last_pos = -1;
        std::vector<float> cos_curr;
        std::vector<float> sin_curr;
        std::vector<float> cos_delta;
        std::vector<float> sin_delta;

        // Q15 cache for Q8_1 RoPE
        std::vector<int16_t> cos_curr_q15;
        std::vector<int16_t> sin_curr_q15;

        void reset()
        {
            last_pos = -1;
            cos_curr.clear();
            sin_curr.clear();
            cos_delta.clear();
            sin_delta.clear();
            cos_curr_q15.clear();
            sin_curr_q15.clear();
        }
    };

    /**
     * @brief Apply RoPE to Q and K tensors (vectorized implementation)
     *
     * Layout: [seq_len, num_heads, head_dim] in row-major order
     *
     * @param q Query tensor (modified in-place)
     * @param k Key tensor (modified in-place)
     * @param seq_len Sequence length
     * @param head_dim Dimension per head (must be even)
     * @param q_heads Number of query heads
     * @param k_heads Number of key heads (may differ for GQA)
     * @param n_past Number of tokens already processed
     * @param freq_base Base frequency for RoPE (model-specific)
     * @param persistent_state Optional persistent state for decode optimization
     */
    void apply_rope_vectorized(
        float *q, float *k,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base,
        RoPEPersistentState *persistent_state = nullptr);

    /**
     * @brief Update persistent state cache for a specific position
     *
     * Ensures that the sin/cos cache in the persistent state is valid for the target position.
     * This is useful for kernels that need to access the cache directly (e.g. fused kernels).
     *
     * @param head_dim Dimension per head
     * @param freq_base Base frequency for RoPE
     * @param target_pos Target position index
     * @param state Persistent state to update
     */
    void update_rope_cache(
        int head_dim, float freq_base, int target_pos,
        RoPEPersistentState &state);

    /**
     * @brief Get cached inverse frequencies for (head_dim, freq_base)
     * @return Vector of inverse frequencies, length = head_dim/2
     */
    const std::vector<float> &get_inv_freq_cached(int head_dim, float freq_base);

    /**
     * @brief Apply RoPE to Q and K tensors (native BF16 implementation)
     *
     * Operates directly on BF16 buffers without intermediate FP32 conversion.
     * Uses vectorized BF16 operations where available (AVX512 BF16, etc.).
     *
     * @param q_bf16 Query tensor in BF16 format (modified in-place)
     * @param k_bf16 Key tensor in BF16 format (modified in-place)
     * @param seq_len Sequence length
     * @param head_dim Dimension per head (must be even)
     * @param q_heads Number of query heads
     * @param k_heads Number of key heads (may differ for GQA)
     * @param n_past Number of tokens already processed
     * @param freq_base Base frequency for RoPE (model-specific)
     * @param persistent_state Optional persistent state for decode optimization
     */
    void apply_rope_bf16(
        uint16_t *q_bf16, uint16_t *k_bf16,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base,
        RoPEPersistentState *persistent_state = nullptr);

    /**
     * @brief Apply RoPE to Q and K tensors (native FP16 implementation)
     *
     * Operates directly on FP16 buffers without intermediate FP32 conversion.
     * Uses vectorized FP16 operations where available (F16C, AVX512 FP16, etc.).
     *
     * @param q_fp16 Query tensor in FP16 format (modified in-place)
     * @param k_fp16 Key tensor in FP16 format (modified in-place)
     * @param seq_len Sequence length
     * @param head_dim Dimension per head (must be even)
     * @param q_heads Number of query heads
     * @param k_heads Number of key heads (may differ for GQA)
     * @param n_past Number of tokens already processed
     * @param freq_base Base frequency for RoPE (model-specific)
     * @param persistent_state Optional persistent state for decode optimization
     */
    void apply_rope_fp16(
        uint16_t *q_fp16, uint16_t *k_fp16,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base,
        RoPEPersistentState *persistent_state = nullptr);

    /**
     * @brief Apply RoPE to Q and K tensors (INT32 not supported)
     *
     * RoPE is an activation operation and cannot be applied to quantized INT32 accumulators.
     * This function exists for API completeness but will always fail.
     *
     * @return Always false (operation not supported)
     */
    bool apply_rope_int32(
        int32_t *q_int32, int32_t *k_int32,
        int seq_len, int head_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base);

    // ============================================================================
    // Individual Implementation Functions (for testing)
    // ============================================================================

    /**
     * @brief Apply RoPE rotation using scalar implementation (reference)
     *
     * This is the baseline implementation that all vectorized versions must match.
     * Used for testing and validation.
     *
     * @param head_ptr Pointer to head data [head_dim] (modified in-place)
     * @param position Token position
     * @param inv_freq Inverse frequencies [head_dim/2]
     * @param head_dim Dimension per head (must be even)
     * @param start_idx Starting pair index (for partial processing)
     */
    void apply_rope_to_head_scalar(
        float *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim,
        int start_idx = 0);

#if defined(__AVX2__)
    /**
     * @brief Apply RoPE rotation using AVX2 implementation
     *
     * Processes 8 float pairs at a time. Tail must be handled by scalar version.
     *
     * @return Number of pairs processed (always multiple of 8)
     */
    int apply_rope_to_head_avx2(
        float *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim);
#endif

#if defined(__AVX512F__)
    /**
     * @brief Apply RoPE rotation using AVX512 implementation
     *
     * Processes 16 float pairs at a time. Tail must be handled by scalar version.
     *
     * @return Number of pairs processed (always multiple of 16)
     */
    int apply_rope_to_head_avx512(
        float *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim);
#endif

    // ============================================================================
    // BF16 Native Precision Implementations (for testing)
    // ============================================================================

    /**
     * @brief Apply RoPE rotation to a single head (BF16 scalar)
     */
    void apply_rope_to_head_bf16_scalar(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim,
        int start_idx = 0);

#if defined(__AVX2__)
    /**
     * @brief Apply RoPE rotation to a single head (BF16 AVX2)
     * @return Number of pairs processed (always multiple of 8)
     */
    int apply_rope_to_head_bf16_avx2(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim);
#endif

#if defined(__AVX512F__)
    /**
     * @brief Apply RoPE rotation to a single head (BF16 AVX512)
     * @return Number of pairs processed (always multiple of 16)
     */
    int apply_rope_to_head_bf16_avx512(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim);
#endif

    // ============================================================================
    // FP16 Native Precision Implementations (for testing)
    // ============================================================================

    /**
     * @brief Apply RoPE rotation to a single head (FP16 scalar)
     */
    void apply_rope_to_head_fp16_scalar(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim,
        int start_idx = 0);

#if defined(__AVX2__)
    /**
     * @brief Apply RoPE rotation to a single head (FP16 AVX2)
     * @return Number of pairs processed (always multiple of 8)
     */
    int apply_rope_to_head_fp16_avx2(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim);
#endif

#if defined(__AVX512F__)
    /**
     * @brief Apply RoPE rotation to a single head (FP16 AVX512)
     * @return Number of pairs processed (always multiple of 16)
     */
    int apply_rope_to_head_fp16_avx512(
        uint16_t *head_ptr,
        int position,
        const std::vector<float> &inv_freq,
        int head_dim);
#endif

    // ============================================================================
    // Q8_1 Pure-Integer RoPE Primitives
    // ============================================================================
    // These primitives perform RoPE rotation entirely in the integer domain,
    // avoiding FP32 round-trips for the quantized data. Sin/cos values are
    // pre-computed as FP32 but quantized to Q15 for the rotation operation.
    //
    // Design rationale:
    // - Sin/cos computation is position-dependent, not data-dependent
    // - Only head_dim/2 sin/cos values needed per position (cheap FP32 compute)
    // - Rotation uses integer multiply: x' = (x*cos_q15 - y*sin_q15) >> 15
    // - Avoids expensive dequant→rotate→requant per head
    // ============================================================================

    /**
     * @brief Cached sin/cos values in Q15 fixed-point format
     *
     * Q15 format: value * 32768 stored as int16_t
     * Range: [-1.0, 1.0) maps to [-32768, 32767]
     */
    struct RoPESinCosQ15
    {
        std::vector<int16_t> cos_q15;
        std::vector<int16_t> sin_q15;

        void resize(int half_dim)
        {
            cos_q15.resize(half_dim);
            sin_q15.resize(half_dim);
        }
    };

    /**
     * @brief Compute Q15 sin/cos values for a position
     *
     * @param position Token position index
     * @param inv_freq Inverse frequency array [head_dim/2]
     * @param head_dim Dimension per head (must be even)
     * @param out Output Q15 sin/cos arrays (pre-sized to head_dim/2)
     */
    void compute_rope_sincos_q15(
        int position,
        const std::vector<float> &inv_freq,
        int head_dim,
        RoPESinCosQ15 &out);

    /**
     * @brief Apply pure-integer RoPE rotation to a single Q8_1 head
     *
     * Performs rotation entirely in integer domain:
     * - Q8_1 blocks contain scale (d) and int8 values (qs)
     * - Rotation: x' = (x*cos - y*sin), y' = (x*sin + y*cos)
     * - Uses 32-bit intermediate precision
     * - Output blocks have updated scales and int8 values
     *
     * @param head_blocks Q8_1 blocks for one head [blocks_per_head]
     * @param blocks_per_head Number of blocks (head_dim / 32)
     * @param cos_q15 Pre-computed cosine values in Q15 [head_dim/2]
     * @param sin_q15 Pre-computed sine values in Q15 [head_dim/2]
     */
    void apply_rope_q8_1_integer_head(
        Q8_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15);

    /**
     * @brief Scalar implementation of Q8_1 RoPE (for testing/reference)
     */
    void apply_rope_q8_1_integer_head_scalar(
        Q8_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15);

#if defined(__AVX2__)
    /**
     * @brief AVX2 implementation of Q8_1 RoPE
     */
    void apply_rope_q8_1_integer_head_avx2(
        Q8_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15);
#endif

#if defined(__AVX512F__)
    /**
     * @brief AVX512 implementation of Q8_1 RoPE
     */
    void apply_rope_q8_1_integer_head_avx512(
        Q8_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15);
#endif

    /**
     * @brief Apply pure-integer RoPE to Q8_1 Q and K tensors
     *
     * High-level wrapper that handles:
     * - Position ID processing (skip padding with -1)
     * - Sin/cos computation and quantization
     * - Parallelization across tokens
     * - Both Q and K tensor processing
     *
     * @param Q Q8_1 Q tensor [seq_len * n_heads * blocks_per_head]
     * @param K Q8_1 K tensor [seq_len * n_kv_heads * blocks_per_head] or nullptr
     * @param position_ids Position indices [seq_len], -1 = padding
     * @param seq_len Sequence length
     * @param n_heads Number of query heads
     * @param n_kv_heads Number of key/value heads
     * @param head_dim Head dimension (must be divisible by 32)
     * @param rope_theta RoPE base frequency (e.g., 10000.0f)
     */
    void apply_rope_q8_1_integer(
        Q8_1Block *Q,
        Q8_1Block *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        RoPEPersistentState *persistent_state = nullptr);

    /**
     * @brief Apply RoPE to Q8_1 input, output to FP32 (Hybrid mode)
     *
     * This eliminates the dequant→rotate→requant cycle by outputting
     * directly to FP32. Used in Hybrid activation precision mode.
     *
     * @param Q_in Q8_1 Q input tensor [seq_len * n_heads * blocks_per_head]
     * @param K_in Q8_1 K input tensor [seq_len * n_kv_heads * blocks_per_head] or nullptr
     * @param Q_out FP32 Q output tensor [seq_len, n_heads * head_dim]
     * @param K_out FP32 K output tensor [seq_len, n_kv_heads * head_dim] or nullptr
     * @param position_ids Position indices [seq_len], -1 = padding
     * @param seq_len Sequence length
     * @param n_heads Number of query heads
     * @param n_kv_heads Number of key/value heads
     * @param head_dim Head dimension (must be divisible by 32)
     * @param rope_theta RoPE base frequency (e.g., 10000.0f)
     */
    void apply_rope_q8_1_to_fp32(
        const Q8_1Block *Q_in,
        const Q8_1Block *K_in,
        float *Q_out,
        float *K_out,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta);

    // ============================================================================
    // Q16_1 In-Place RoPE Primitives
    // ============================================================================
    // Q16_1 provides 256× finer quantization than Q8_1 with FP32 scale.
    // This makes in-place RoPE feasible with acceptable precision loss.
    //
    // Key advantages over Q8_1:
    // - int16 values: ±32767 range vs ±127 (256× finer)
    // - FP32 scale: no scale quantization error
    // - Better preservation of rotation precision
    //
    // Algorithm:
    // 1. Dequantize paired blocks (blockA = first half, blockB = second half)
    // 2. Apply rotation: x' = x*cos - y*sin, y' = x*sin + y*cos
    // 3. Requantize back to Q16_1 with new scale
    // ============================================================================

    /**
     * @brief Apply in-place RoPE rotation to a single Q16_1 head
     *
     * @param head_blocks Q16_1 blocks for one head [blocks_per_head]
     * @param blocks_per_head Number of blocks (head_dim / 32)
     * @param cos_q15 Pre-computed cosine values in Q15 [head_dim/2]
     * @param sin_q15 Pre-computed sine values in Q15 [head_dim/2]
     */
    void apply_rope_q16_1_integer_head(
        Q16_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15);

    /**
     * @brief Scalar implementation for Q16_1 RoPE (for testing/reference)
     */
    void apply_rope_q16_1_integer_head_scalar(
        Q16_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15);

#if defined(__AVX2__) || defined(__AVX512F__)
    /**
     * @brief AVX2 implementation for Q16_1 RoPE (for testing)
     */
    void apply_rope_q16_1_integer_head_avx2(
        Q16_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15);
#endif

#if defined(__AVX512F__)
    /**
     * @brief AVX512 implementation for Q16_1 RoPE (for testing)
     */
    void apply_rope_q16_1_integer_head_avx512(
        Q16_1Block *head_blocks,
        int blocks_per_head,
        const int16_t *cos_q15,
        const int16_t *sin_q15);
#endif

    /**
     * @brief Apply in-place RoPE to Q16_1 Q and K tensors
     *
     * High-level wrapper that handles:
     * - Position ID processing (skip padding with -1)
     * - Sin/cos computation and Q15 quantization
     * - Parallelization across tokens
     * - Both Q and K tensor processing
     *
     * @param Q Q16_1 Q tensor [seq_len * n_heads * blocks_per_head]
     * @param K Q16_1 K tensor [seq_len * n_kv_heads * blocks_per_head] or nullptr
     * @param position_ids Position indices [seq_len], -1 = padding
     * @param seq_len Sequence length
     * @param n_heads Number of query heads
     * @param n_kv_heads Number of key/value heads
     * @param head_dim Head dimension (must be divisible by 32)
     * @param rope_theta RoPE base frequency (e.g., 10000.0f)
     * @param persistent_state Optional persistent state for decode optimization
     */
    void apply_rope_q16_1_integer(
        Q16_1Block *Q,
        Q16_1Block *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        RoPEPersistentState *persistent_state = nullptr);

    // ============================================================================
    // Q8_1 → Q16_1 RoPE Primitives (HybridQ16 Mode)
    // ============================================================================
    // These primitives take Q8_1 input, apply RoPE rotation, and output Q16_1.
    // This is the optimal path for HybridQ16 mode:
    // - Q8_1 input from QKV GEMM projection
    // - Q16_1 output for Q16 fused attention kernel
    //
    // Advantages:
    // - No intermediate FP32 dequant/requant round-trip
    // - Direct Q8_1 → rotate → Q16_1 quantization
    // - Q16_1 output has 256× finer precision than Q8_1
    // ============================================================================

    /**
     * @brief Apply RoPE to Q8_1 input, output to Q16_1 (HybridQ16 mode)
     *
     * This primitive:
     * 1. Dequantizes Q8_1 blocks to FP32
     * 2. Applies RoPE rotation
     * 3. Requantizes to Q16_1 (higher precision output)
     *
     * Used in HybridQ16 mode where:
     * - QKV projection outputs Q8_1 (standard quantized GEMM)
     * - Q16 fused attention kernel expects Q16_1 inputs
     *
     * @param Q_in Q8_1 Q input tensor [seq_len * n_heads * blocks_per_head]
     * @param K_in Q8_1 K input tensor [seq_len * n_kv_heads * blocks_per_head] or nullptr
     * @param Q_out Q16_1 Q output tensor [seq_len * n_heads * blocks_per_head]
     * @param K_out Q16_1 K output tensor [seq_len * n_kv_heads * blocks_per_head] or nullptr
     * @param position_ids Position indices [seq_len], -1 = padding
     * @param seq_len Sequence length
     * @param n_heads Number of query heads
     * @param n_kv_heads Number of key/value heads
     * @param head_dim Head dimension (must be divisible by 32)
     * @param rope_theta RoPE base frequency (e.g., 10000.0f)
     */
    void apply_rope_q8_1_to_q16_1(
        const Q8_1Block *Q_in,
        const Q8_1Block *K_in,
        Q16_1Block *Q_out,
        Q16_1Block *K_out,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta);

} // namespace llaminar2::primitives
