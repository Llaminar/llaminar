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
     * @brief Apply partial RoPE to Q and K tensors
     *
     * For models with partial_rotary_factor < 1.0 (e.g. Qwen3.5 FA layers).
     * Rotates only the first `rotary_dim` elements of each head while leaving
     * the remaining (head_dim - rotary_dim) elements untouched.
     *
     * Uses `head_dim` for stride (memory layout) and `rotary_dim` for rotation
     * loop bounds and frequency computation.
     *
     * @param q Query tensor [seq_len, q_heads * head_dim] (modified in-place)
     * @param k Key tensor [seq_len, k_heads * head_dim] (modified in-place, nullable)
     * @param seq_len Sequence length
     * @param head_dim Full dimension per head (for stride)
     * @param rotary_dim Number of dims to rotate (must be even, <= head_dim)
     * @param q_heads Number of query heads
     * @param k_heads Number of key heads
     * @param n_past Number of tokens already processed
     * @param freq_base Base frequency for RoPE
     */
    void apply_rope_partial(
        float *q, float *k,
        int seq_len, int head_dim, int rotary_dim,
        int q_heads, int k_heads,
        int n_past, float freq_base);

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
    // Templated Q16 RoPE Primitives (Variable Block Size Support)
    // ============================================================================
    // These templated functions support all Q16 block sizes (32, 64, 128).
    // The block type determines BLOCK_SIZE at compile time for optimal vectorization.
    //
    // Template parameter BlockType must have:
    //   - static constexpr size_t BLOCK_SIZE
    //   - float d (scale factor)
    //   - int32_t sum_qs (pre-computed sum)
    //   - int16_t qs[BLOCK_SIZE] (quantized values)
    //
    // Supported block types: Q16_1Block, Q16_1Block_64, Q16_1Block_128
    //
    // Note: For MLA architectures (DeepSeek V3, Kimi K2), use separate
    // NOPE (128-dim) and ROPE (64-dim) tensors with their own scales.
    // ============================================================================

    /**
     * @brief Templated scalar RoPE implementation for any Q16 block type
     *
     * Reference implementation that works with any block size.
     * Used for testing and as fallback for non-vectorized paths.
     *
     * @tparam BlockType Q16 block type (Q16_1Block, Q16_1Block_64, etc.)
     * @param head_blocks Q16 blocks for one head (1 block for 1-block-per-head)
     * @param num_blocks Number of blocks in the head
     * @param cos_q15 Pre-computed cosine values in Q15 [head_dim/2]
     * @param sin_q15 Pre-computed sine values in Q15 [head_dim/2]
     */
    template <typename BlockType>
    void apply_rope_q16_integer_head_scalar(
        BlockType *head_blocks,
        int num_blocks,
        const int16_t *cos_q15,
        const int16_t *sin_q15);

#if defined(__AVX2__) || defined(__AVX512F__)
    /**
     * @brief Templated AVX2 RoPE implementation for any Q16 block type
     *
     * Processes 8 elements at a time using AVX2 intrinsics.
     * Uses a loop over chunks for variable block sizes.
     *
     * @tparam BlockType Q16 block type (Q16_1Block, Q16_1Block_64, etc.)
     * @param head_blocks Q16 blocks for one head
     * @param num_blocks Number of blocks in the head
     * @param cos_q15 Pre-computed cosine values in Q15 [head_dim/2]
     * @param sin_q15 Pre-computed sine values in Q15 [head_dim/2]
     */
    template <typename BlockType>
    void apply_rope_q16_integer_head_avx2(
        BlockType *head_blocks,
        int num_blocks,
        const int16_t *cos_q15,
        const int16_t *sin_q15);
#endif

#if defined(__AVX512F__)
    /**
     * @brief Templated AVX512 RoPE implementation for any Q16 block type
     *
     * Processes 16 elements at a time using AVX512 intrinsics.
     * Uses a loop over chunks for variable block sizes.
     *
     * @tparam BlockType Q16 block type (Q16_1Block, Q16_1Block_64, etc.)
     * @param head_blocks Q16 blocks for one head
     * @param num_blocks Number of blocks in the head
     * @param cos_q15 Pre-computed cosine values in Q15 [head_dim/2]
     * @param sin_q15 Pre-computed sine values in Q15 [head_dim/2]
     */
    template <typename BlockType>
    void apply_rope_q16_integer_head_avx512(
        BlockType *head_blocks,
        int num_blocks,
        const int16_t *cos_q15,
        const int16_t *sin_q15);
#endif

    /**
     * @brief Templated auto-dispatching RoPE implementation
     *
     * Automatically selects optimal implementation (AVX512 > AVX2 > scalar).
     *
     * @tparam BlockType Q16 block type (Q16_1Block, Q16_1Block_64, etc.)
     */
    template <typename BlockType>
    void apply_rope_q16_integer_head(
        BlockType *head_blocks,
        int num_blocks,
        const int16_t *cos_q15,
        const int16_t *sin_q15);

    /**
     * @brief Apply in-place RoPE to Q16 Q and K tensors (templated version)
     *
     * High-level wrapper for any Q16 block size.
     *
     * @tparam BlockType Q16 block type (Q16_1Block, Q16_1Block_64, etc.)
     * @param Q Q16 Q tensor [seq_len * n_heads * blocks_per_head]
     * @param K Q16 K tensor [seq_len * n_kv_heads * blocks_per_head] or nullptr
     * @param position_ids Position indices [seq_len], -1 = padding
     * @param seq_len Sequence length
     * @param n_heads Number of query heads
     * @param n_kv_heads Number of key/value heads
     * @param head_dim Head dimension (must be divisible by BlockType::BLOCK_SIZE)
     * @param rope_theta RoPE base frequency (e.g., 10000.0f)
     * @param persistent_state Optional persistent state for decode optimization
     */
    template <typename BlockType>
    void apply_rope_q16_integer(
        BlockType *Q,
        BlockType *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        RoPEPersistentState *persistent_state = nullptr);

    /**
     * @brief Apply RoPE with runtime block size selection
     *
     * Dispatches to correct template instantiation based on Q16BlockSize enum.
     *
     * @param Q Q16 Q tensor (void* for runtime polymorphism)
     * @param K Q16 K tensor or nullptr
     * @param block_size Runtime block size selector
     * @param position_ids Position indices [seq_len]
     * @param seq_len Sequence length
     * @param n_heads Number of query heads
     * @param n_kv_heads Number of key/value heads
     * @param head_dim Head dimension
     * @param rope_theta RoPE base frequency
     * @param persistent_state Optional persistent state
     */
    void apply_rope_q16_integer_dispatch(
        void *Q,
        void *K,
        Q16BlockSize block_size,
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
     * @brief Apply RoPE to Q8_1 input, output to Q16_1 (HybridQ16 mode) - LEGACY
     *
     * This is the legacy 32-block version. Use the templated version below for
     * variable block size support.
     *
     * @deprecated Use apply_rope_q8_1_to_q16<OutBlockType>() for variable block sizes.
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

    // ============================================================================
    // Templated Q8_1 → Q16 RoPE Primitives (Variable Output Block Size)
    // ============================================================================
    // These templated functions convert Q8_1 input to variable-size Q16 output.
    //
    // Key design:
    // - Input: Always Q8_1Block (32-element), head_dim/32 blocks per head
    // - Output: OutBlockType template, head_dim/OutBlockType::BLOCK_SIZE blocks per head
    // - Per-head normalization: All output blocks share ONE scale (max |dequant| across input)
    // - Returns: Head scale for use in attention score computation
    //
    // Supported output types: Q16_1Block, Q16_1Block_64, Q16_1Block_128
    //
    // Note: For MLA architectures (DeepSeek V3, Kimi K2), use separate
    // NOPE (128-dim) and ROPE (64-dim) tensors with their own scales.
    //
    // The optimal path is when OutBlockType::BLOCK_SIZE == head_dim, giving 1 block
    // per head and eliminating multi-block scale mixing in attention.
    // ============================================================================

    /**
     * @brief Per-head Q8_1→Q16 scalar implementation (reference)
     *
     * Processes one head: reads Q8_1 input blocks, applies RoPE rotation,
     * outputs to variable-size Q16 blocks with unified per-head scale.
     *
     * @tparam OutBlockType Output Q16 block type (Q16_1Block, Q16_1Block_64, etc.)
     * @param q8_in Input Q8_1 blocks [head_dim / 32 blocks]
     * @param q16_out Output Q16 blocks [head_dim / OutBlockType::BLOCK_SIZE blocks]
     * @param head_dim Head dimension (must be divisible by both 32 and BLOCK_SIZE)
     * @param cos_q15 Pre-computed cosine values in Q15 [head_dim/2]
     * @param sin_q15 Pre-computed sine values in Q15 [head_dim/2]
     * @return Head scale factor (max |dequant| / 127.0f from input, used for output blocks)
     */
    template <typename OutBlockType>
    float apply_rope_q8_1_to_q16_head_scalar(
        const Q8_1Block *q8_in,
        OutBlockType *q16_out,
        int head_dim,
        const int16_t *cos_q15,
        const int16_t *sin_q15);

#if defined(__AVX2__) || defined(__AVX512F__)
    /**
     * @brief Per-head Q8_1→Q16 AVX2 implementation
     *
     * Vectorized version processing 8 elements at a time.
     *
     * @tparam OutBlockType Output Q16 block type
     * @return Head scale factor
     */
    template <typename OutBlockType>
    float apply_rope_q8_1_to_q16_head_avx2(
        const Q8_1Block *q8_in,
        OutBlockType *q16_out,
        int head_dim,
        const int16_t *cos_q15,
        const int16_t *sin_q15);
#endif

#if defined(__AVX512F__)
    /**
     * @brief Per-head Q8_1→Q16 AVX512 implementation
     *
     * Vectorized version processing 16 elements at a time.
     *
     * @tparam OutBlockType Output Q16 block type
     * @return Head scale factor
     */
    template <typename OutBlockType>
    float apply_rope_q8_1_to_q16_head_avx512(
        const Q8_1Block *q8_in,
        OutBlockType *q16_out,
        int head_dim,
        const int16_t *cos_q15,
        const int16_t *sin_q15);
#endif

    /**
     * @brief Per-head Q8_1→Q16 auto-dispatching implementation
     *
     * Automatically selects optimal implementation (AVX512 > AVX2 > scalar).
     *
     * @tparam OutBlockType Output Q16 block type
     * @return Head scale factor
     */
    template <typename OutBlockType>
    float apply_rope_q8_1_to_q16_head(
        const Q8_1Block *q8_in,
        OutBlockType *q16_out,
        int head_dim,
        const int16_t *cos_q15,
        const int16_t *sin_q15);

    /**
     * @brief Apply RoPE to Q8_1 Q and K tensors, output to variable-size Q16
     *
     * High-level wrapper for any Q16 output block size. Processes all heads
     * in parallel using OMP.
     *
     * @tparam OutBlockType Output Q16 block type (Q16_1Block, Q16_1Block_64, etc.)
     * @param Q_in Q8_1 Q input [seq_len * n_heads * (head_dim/32)]
     * @param K_in Q8_1 K input [seq_len * n_kv_heads * (head_dim/32)] or nullptr
     * @param Q_out Q16 Q output [seq_len * n_heads * (head_dim/BLOCK_SIZE)]
     * @param K_out Q16 K output [seq_len * n_kv_heads * (head_dim/BLOCK_SIZE)] or nullptr
     * @param Q_head_scales Output: per-head Q scales [seq_len * n_heads] or nullptr
     * @param K_head_scales Output: per-head K scales [seq_len * n_kv_heads] or nullptr
     * @param position_ids Position indices [seq_len], nullptr = use index
     * @param seq_len Sequence length
     * @param n_heads Number of query heads
     * @param n_kv_heads Number of key/value heads
     * @param head_dim Head dimension
     * @param rope_theta RoPE base frequency
     */
    template <typename OutBlockType>
    void apply_rope_q8_1_to_q16(
        const Q8_1Block *Q_in,
        const Q8_1Block *K_in,
        OutBlockType *Q_out,
        OutBlockType *K_out,
        float *Q_head_scales,
        float *K_head_scales,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta);

    /**
     * @brief Apply RoPE Q8_1→Q16 with runtime block size dispatch
     *
     * Dispatches to correct template instantiation based on Q16BlockSize enum.
     *
     * @param Q_in Q8_1 Q input
     * @param K_in Q8_1 K input or nullptr
     * @param Q_out Q16 Q output (void* for runtime polymorphism)
     * @param K_out Q16 K output (void*) or nullptr
     * @param Q_head_scales Output: per-head Q scales or nullptr
     * @param K_head_scales Output: per-head K scales or nullptr
     * @param block_size Runtime block size selector
     * @param position_ids Position indices
     * @param seq_len Sequence length
     * @param n_heads Number of query heads
     * @param n_kv_heads Number of key/value heads
     * @param head_dim Head dimension
     * @param rope_theta RoPE base frequency
     */
    void apply_rope_q8_1_to_q16_dispatch(
        const Q8_1Block *Q_in,
        const Q8_1Block *K_in,
        void *Q_out,
        void *K_out,
        float *Q_head_scales,
        float *K_head_scales,
        Q16BlockSize block_size,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta);

    // ============================================================================
    // Fixed-Scale Q8_1 → Q16 RoPE Primitives
    // ============================================================================

    /**
     * @brief Per-head Q8_1→Q16 with FIXED output scale (pure integer per-element)
     *
     * Unlike apply_rope_q8_1_to_q16_head_scalar() which uses data-adaptive scaling,
     * this function outputs Q16 blocks with a FIXED scale = kv_cache_scale / 32767.
     * This ensures Q, K, and V all use the same scale factor for integer attention.
     *
     * Algorithm:
     * 1. Compute per-block scale ratios: ratio_q16 = d_block / d_fixed * 65536 (O(head_dim/32) FP32)
     * 2. Rescale Q8 to fixed scale: scaled = (q8 * ratio_q16 + 32768) >> 16 (pure integer)
     * 3. Apply RoPE rotation: out = (x*cos - y*sin + 16384) >> 15 (pure integer)
     * 4. Pack to Q16 with fixed d = kv_cache_scale / 32767
     *
     * @tparam OutBlockType Output Q16 block type (Q16_1Block, Q16_1Block_64, etc.)
     * @param q8_in Input Q8_1 blocks [head_dim/32]
     * @param q16_out Output Q16 blocks [head_dim/BLOCK_SIZE]
     * @param head_dim Head dimension (must be multiple of 32)
     * @param cos_q15 Pre-computed cosine values in Q15 [head_dim/2]
     * @param sin_q15 Pre-computed sine values in Q15 [head_dim/2]
     * @param kv_cache_scale Fixed scale for output (e.g., 8.0f)
     */
    template <typename OutBlockType>
    void apply_rope_q8_1_to_q16_head_fixed_scale(
        const Q8_1Block *q8_in,
        OutBlockType *q16_out,
        int head_dim,
        const int16_t *cos_q15,
        const int16_t *sin_q15,
        float kv_cache_scale);

    /**
     * @brief Apply fixed-scale RoPE to Q8_1 Q and K tensors, output to Q16
     *
     * High-level wrapper that processes all heads in parallel. All output Q16 blocks
     * have d = kv_cache_scale / 32767, enabling true integer attention.
     *
     * LAYOUT NOTE:
     * - Q output is HEAD-MAJOR [n_heads][seq_len][head_dim] for direct attention kernel use
     * - K output is POSITION-MAJOR [seq_len][n_kv_heads][head_dim] for KV cache storage
     *
     * @tparam OutBlockType Output Q16 block type
     * @param Q_in Q8_1 Q input [seq_len * n_heads * (head_dim/32)]
     * @param K_in Q8_1 K input [seq_len * n_kv_heads * (head_dim/32)] or nullptr
     * @param Q_out Q16 Q output HEAD-MAJOR [n_heads][seq_len][head_dim/BLOCK_SIZE]
     * @param K_out Q16 K output position-major [seq_len][n_kv_heads][head_dim/BLOCK_SIZE] or nullptr
     * @param position_ids Position indices [seq_len], nullptr = use index
     * @param seq_len Sequence length
     * @param n_heads Number of query heads
     * @param n_kv_heads Number of key/value heads
     * @param head_dim Head dimension
     * @param rope_theta RoPE base frequency
     * @param kv_cache_scale Fixed scale for all output blocks (e.g., 8.0f)
     */
    template <typename OutBlockType>
    void apply_rope_q8_1_to_q16_fixed_scale(
        const Q8_1Block *Q_in,
        const Q8_1Block *K_in,
        OutBlockType *Q_out,
        OutBlockType *K_out,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        float kv_cache_scale);

    /**
     * @brief Apply fixed-scale RoPE Q8_1→Q16 with runtime block size dispatch
     *
     * @param block_size Runtime block size selector
     * @param kv_cache_scale Fixed scale for all output blocks
     */
    void apply_rope_q8_1_to_q16_fixed_scale_dispatch(
        const Q8_1Block *Q_in,
        const Q8_1Block *K_in,
        void *Q_out,
        void *K_out,
        Q16BlockSize block_size,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        float kv_cache_scale);

    // ============================================================================
    // Fixed-Scale Q16_1 → Q16_1 RoPE Primitives (HybridQ16 K Precision Fix)
    // ============================================================================
    // These primitives apply RoPE rotation to Q16_1 input and rescale output to
    // a FIXED kv_cache_scale. Used when K projection outputs Q16_1 directly from
    // GEMM (bypassing Q8_1 intermediate) to preserve precision for small values.
    //
    // Why needed:
    // - GEMM outputs Q16_1 with dynamic per-block scales (based on output magnitude)
    // - KV cache requires FIXED scales for efficient integer attention
    // - This function bridges the gap: applies RoPE and normalizes to kv_cache_scale
    //
    // Algorithm:
    // 1. Dequantize Q16_1 blocks to FP32 using per-block dynamic scales
    // 2. Apply RoPE rotation in FP32 (x' = x*cos - y*sin, y' = x*sin + y*cos)
    // 3. Requantize to Q16_1 with FIXED d = kv_cache_scale / 32767
    //
    // Note: Unlike Q8_1→Q16 path, input is already Q16_1 with 256× finer precision.
    // The FP32 intermediate is acceptable here since we're already in high-precision
    // quantized domain and the rescale operation naturally requires dequantization.
    // ============================================================================

    /**
     * @brief Per-head Q16_1→Q16_1 RoPE with FIXED output scale (scalar implementation)
     *
     * Scalar fallback implementation. Always available on all platforms.
     */
    template <typename BlockType>
    void apply_rope_q16_to_q16_head_fixed_scale_scalar(
        const BlockType *q16_in,
        BlockType *q16_out,
        int head_dim,
        const int16_t *cos_q15,
        const int16_t *sin_q15,
        float kv_cache_scale);

#if defined(__AVX2__)
    /**
     * @brief Per-head Q16_1→Q16_1 RoPE with FIXED output scale (AVX2 implementation)
     *
     * Optimized AVX2 version processing 4 chunks of 8 elements each.
     * Only available for Q16_1Block (32-element blocks).
     */
    void apply_rope_q16_to_q16_head_fixed_scale_avx2(
        const Q16_1Block *q16_in,
        Q16_1Block *q16_out,
        int head_dim,
        const int16_t *cos_q15,
        const int16_t *sin_q15,
        float kv_cache_scale);
#endif

#if defined(__AVX512F__)
    /**
     * @brief Per-head Q16_1→Q16_1 RoPE with FIXED output scale (AVX512 implementation)
     *
     * Optimized AVX512 version processing 2 chunks of 16 elements each.
     * Only available for Q16_1Block (32-element blocks).
     */
    void apply_rope_q16_to_q16_head_fixed_scale_avx512(
        const Q16_1Block *q16_in,
        Q16_1Block *q16_out,
        int head_dim,
        const int16_t *cos_q15,
        const int16_t *sin_q15,
        float kv_cache_scale);
#endif

    /**
     * @brief Per-head Q16_1→Q16_1 RoPE with FIXED output scale (auto-dispatch)
     *
     * Takes Q16_1 input (with dynamic per-block scales from GEMM), applies RoPE,
     * outputs Q16_1 with uniform d = kv_cache_scale / 32767 for integer attention.
     *
     * Automatically dispatches to AVX512 > AVX2 > scalar based on platform.
     *
     * @tparam BlockType Q16 block type (Q16_1Block, Q16_1Block_64, Q16_1Block_128)
     * @param q16_in Input Q16 blocks with dynamic scales [head_dim / BLOCK_SIZE]
     * @param q16_out Output Q16 blocks with fixed scale [head_dim / BLOCK_SIZE]
     * @param head_dim Head dimension (must be multiple of BLOCK_SIZE)
     * @param cos_q15 Pre-computed cosine values in Q15 [head_dim/2]
     * @param sin_q15 Pre-computed sine values in Q15 [head_dim/2]
     * @param kv_cache_scale Fixed scale for output (e.g., 8.0f)
     */
    template <typename BlockType>
    void apply_rope_q16_to_q16_head_fixed_scale(
        const BlockType *q16_in,
        BlockType *q16_out,
        int head_dim,
        const int16_t *cos_q15,
        const int16_t *sin_q15,
        float kv_cache_scale);

    /**
     * @brief Apply fixed-scale RoPE to Q16 K tensor (batch wrapper)
     *
     * High-level wrapper for processing K tensor with fixed output scale.
     * Used in HybridQ16 pipeline where K comes from GEMM as Q16_1.
     *
     * LAYOUT NOTE:
     * - K output is POSITION-MAJOR [seq_len][n_kv_heads][head_dim] for KV cache storage
     *
     * @tparam BlockType Q16 block type
     * @param K_in Q16 K input [seq_len * n_kv_heads * (head_dim/BLOCK_SIZE)]
     * @param K_out Q16 K output with fixed scale [seq_len * n_kv_heads * (head_dim/BLOCK_SIZE)]
     * @param position_ids Position indices [seq_len], nullptr = use index
     * @param seq_len Sequence length
     * @param n_kv_heads Number of key/value heads
     * @param head_dim Head dimension
     * @param rope_theta RoPE base frequency
     * @param kv_cache_scale Fixed scale for all output blocks
     */
    template <typename BlockType>
    void apply_rope_q16_to_q16_fixed_scale(
        const BlockType *K_in,
        BlockType *K_out,
        const int *position_ids,
        int seq_len,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        float kv_cache_scale);

    /**
     * @brief Apply fixed-scale RoPE Q16→Q16 with runtime block size dispatch
     *
     * @param block_size Runtime block size selector
     * @param kv_cache_scale Fixed scale for all output blocks
     */
    void apply_rope_q16_to_q16_fixed_scale_dispatch(
        const void *K_in,
        void *K_out,
        Q16BlockSize block_size,
        const int *position_ids,
        int seq_len,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        float kv_cache_scale);

    // ============================================================================
    // Q16→Q16 Dynamic-Scale RoPE (Phase 12)
    // ============================================================================
    //
    // These functions preserve the full dynamic range of spiky K projections by
    // unifying to the max input scale rather than a fixed scale. The output scale
    // is returned for subsequent VNNI-safe normalization.
    //
    // Key difference from fixed-scale:
    // - Fixed-scale: All output blocks get d = kv_cache_scale / 32767 (may clip peaks)
    // - Dynamic-scale: All output blocks get d = max(d_input) (preserves peaks)
    //
    // Typical flow:
    // 1. K proj GEMM → Q16_1 K (dynamic per-block scales, peaks ~130)
    // 2. Dynamic-scale RoPE → Q16_1 K (unified scale = max_d, peaks preserved)
    // 3. VNNI-safe normalization → Q16_1 K (VNNI-safe qs, norm_factor returned)
    // 4. Store K in KV cache with norm_factor metadata
    // 5. Attention applies norm_factor correction to scores
    // ============================================================================

    /**
     * @brief Per-head Q16_1→Q16_1 RoPE with DYNAMIC output scale (scalar implementation)
     *
     * Finds the maximum scale across all input blocks within the head, then applies
     * RoPE with that unified scale as output. This preserves spiky values that would
     * be clipped by fixed-scale quantization.
     *
     * INTEGER-ONLY inner loop: Only O(blocks) FP32 for scale math, O(elements) integer.
     *
     * @tparam BlockType Q16 block type (Q16_1Block, Q16_1Block_64, Q16_1Block_128)
     * @param q16_in Input Q16 blocks with dynamic scales [head_dim / BLOCK_SIZE]
     * @param q16_out Output Q16 blocks with unified scale [head_dim / BLOCK_SIZE]
     * @param head_dim Head dimension (must be multiple of BLOCK_SIZE)
     * @param cos_q15 Pre-computed cosine values in Q15 [head_dim/2]
     * @param sin_q15 Pre-computed sine values in Q15 [head_dim/2]
     * @param[out] out_unified_scale Returns the unified output scale (max input d)
     */
    template <typename BlockType>
    void apply_rope_q16_to_q16_head_dynamic_scale_scalar(
        const BlockType *q16_in,
        BlockType *q16_out,
        int head_dim,
        const int16_t *cos_q15,
        const int16_t *sin_q15,
        float *out_unified_scale);

#if defined(__AVX2__)
    /**
     * @brief Per-head Q16→Q16 RoPE with DYNAMIC output scale (AVX2 templated implementation)
     *
     * Optimized AVX2 version that works with any Q16 block size (32, 64, 128).
     * Uses 256-bit vectors to process 8 elements at a time.
     *
     * @tparam BlockType Q16 block type (Q16_1Block, Q16_1Block_64, Q16_1Block_128)
     */
    template <typename BlockType>
    void apply_rope_q16_to_q16_head_dynamic_scale_avx2_impl(
        const BlockType *q16_in,
        BlockType *q16_out,
        int head_dim,
        const int16_t *cos_q15,
        const int16_t *sin_q15,
        float *out_unified_scale);

    /**
     * @brief Per-head Q16_1→Q16_1 RoPE with DYNAMIC output scale (AVX2 legacy wrapper)
     *
     * Legacy wrapper for Q16_1Block (32-element blocks) for API compatibility.
     */
    void apply_rope_q16_to_q16_head_dynamic_scale_avx2(
        const Q16_1Block *q16_in,
        Q16_1Block *q16_out,
        int head_dim,
        const int16_t *cos_q15,
        const int16_t *sin_q15,
        float *out_unified_scale);
#endif

#if defined(__AVX512F__)
    /**
     * @brief Per-head Q16→Q16 RoPE with DYNAMIC output scale (AVX512 templated implementation)
     *
     * Optimized AVX512 version that works with any Q16 block size (32, 64, 128).
     * Uses 512-bit vectors to process 16 elements at a time.
     *
     * @tparam BlockType Q16 block type (Q16_1Block, Q16_1Block_64, Q16_1Block_128)
     */
    template <typename BlockType>
    void apply_rope_q16_to_q16_head_dynamic_scale_avx512_impl(
        const BlockType *q16_in,
        BlockType *q16_out,
        int head_dim,
        const int16_t *cos_q15,
        const int16_t *sin_q15,
        float *out_unified_scale);

    /**
     * @brief Per-head Q16_1→Q16_1 RoPE with DYNAMIC output scale (AVX512 legacy wrapper)
     *
     * Legacy wrapper for Q16_1Block (32-element blocks) for API compatibility.
     */
    void apply_rope_q16_to_q16_head_dynamic_scale_avx512(
        const Q16_1Block *q16_in,
        Q16_1Block *q16_out,
        int head_dim,
        const int16_t *cos_q15,
        const int16_t *sin_q15,
        float *out_unified_scale);
#endif

    /**
     * @brief Per-head Q16_1→Q16_1 RoPE with DYNAMIC output scale (auto-dispatch)
     *
     * Takes Q16_1 input (with dynamic per-block scales from GEMM), applies RoPE,
     * outputs Q16_1 with unified d = max(d_input) for full dynamic range preservation.
     *
     * Automatically dispatches to AVX512 > AVX2 > scalar based on platform.
     * All block sizes (32, 64, 128) are now supported with SIMD acceleration.
     *
     * @tparam BlockType Q16 block type (Q16_1Block, Q16_1Block_64, Q16_1Block_128)
     * @param q16_in Input Q16 blocks with dynamic scales [head_dim / BLOCK_SIZE]
     * @param q16_out Output Q16 blocks with unified scale [head_dim / BLOCK_SIZE]
     * @param head_dim Head dimension (must be multiple of BLOCK_SIZE)
     * @param cos_q15 Pre-computed cosine values in Q15 [head_dim/2]
     * @param sin_q15 Pre-computed sine values in Q15 [head_dim/2]
     * @param[out] out_unified_scale Returns the unified output scale (max input d)
     */
    template <typename BlockType>
    void apply_rope_q16_to_q16_head_dynamic_scale(
        const BlockType *q16_in,
        BlockType *q16_out,
        int head_dim,
        const int16_t *cos_q15,
        const int16_t *sin_q15,
        float *out_unified_scale);

    /**
     * @brief Apply dynamic-scale RoPE to Q16 K tensor (batch wrapper)
     *
     * High-level wrapper for processing K tensor with dynamic output scale.
     * Returns per-head unified scales for subsequent VNNI-safe normalization.
     *
     * @tparam BlockType Q16 block type
     * @param K_in Q16 K input [seq_len * n_kv_heads * (head_dim/BLOCK_SIZE)]
     * @param K_out Q16 K output with unified scale [seq_len * n_kv_heads * (head_dim/BLOCK_SIZE)]
     * @param position_ids Position indices [seq_len], nullptr = use index
     * @param seq_len Sequence length
     * @param n_kv_heads Number of key/value heads
     * @param head_dim Head dimension
     * @param rope_theta RoPE base frequency
     * @param[out] out_head_scales Per-head unified scales [seq_len * n_kv_heads]
     */
    template <typename BlockType>
    void apply_rope_q16_to_q16_dynamic_scale(
        const BlockType *K_in,
        BlockType *K_out,
        const int *position_ids,
        int seq_len,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        float *out_head_scales);

    /**
     * @brief Apply dynamic-scale RoPE Q16→Q16 with runtime block size dispatch
     *
     * @param block_size Runtime block size selector
     * @param[out] out_head_scales Per-head unified scales [seq_len * n_kv_heads]
     */
    void apply_rope_q16_to_q16_dynamic_scale_dispatch(
        const void *K_in,
        void *K_out,
        Q16BlockSize block_size,
        const int *position_ids,
        int seq_len,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        float *out_head_scales);

} // namespace llaminar2::primitives
