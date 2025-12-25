/**
 * @file FusedAttentionWoRef.h
 * @brief Reference implementation of fused attention + Wo projection kernel.
 *
 * This kernel fuses the attention computation with Wo projection to eliminate
 * the intermediate quantization step (Context FP32 → Q8_1 → FP32 for Wo).
 *
 * By keeping the attention context in FP32 and immediately projecting through Wo,
 * we preserve numerical precision and reduce quantization noise accumulation.
 *
 * @author David Sanftenberg
 * @date December 2025
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "microkernels/Q8DotProduct.h" // Brings in llaminar2::Q8_1Block via using declaration
#include "microkernels/WoProjection.h" // For WoWeightType

namespace llaminar::v2::kernels
{

    // Use types from microkernels namespace (Q8_1Block is now llaminar2::Q8_1Block)
    using microkernels::Q8_1Block;
    using microkernels::WoWeightType;

    /**
     * @brief Parameters for the fused attention + Wo projection kernel.
     *
     * Supports three execution modes:
     * 1. **Single sequence** (batch_size=1): Traditional prefill or single decode
     * 2. **Batched uniform** (batch_size>1, kv_seq_lens=nullptr): All sequences same KV length
     * 3. **Batched variable** (batch_size>1, kv_seq_lens!=nullptr): Per-sequence KV lengths
     *
     * Tensor layouts when batched:
     * - Q: [batch_size, seq_len, num_heads, head_dim/32 blocks]
     * - K: [batch_size, max_kv_seq_len, num_kv_heads, head_dim/32 blocks]
     * - V: [batch_size, max_kv_seq_len, num_kv_heads, head_dim/32 blocks]
     * - output: [batch_size, seq_len, d_model]
     */
    struct FusedAttentionWoParams
    {
        // ============== Input tensors ==============

        /**
         * Q tensor blocks
         * - Non-batched: [seq_len, num_heads, head_dim/32 blocks]
         * - Batched: [batch_size, seq_len, num_heads, head_dim/32 blocks]
         */
        const Q8_1Block *Q;

        /**
         * K tensor blocks
         * - Non-batched: [kv_seq_len, num_kv_heads, head_dim/32 blocks]
         * - Batched: [batch_size, max_kv_seq_len, num_kv_heads, head_dim/32 blocks]
         */
        const Q8_1Block *K;

        /**
         * V tensor blocks
         * - Non-batched: [kv_seq_len, num_kv_heads, head_dim/32 blocks]
         * - Batched: [batch_size, max_kv_seq_len, num_kv_heads, head_dim/32 blocks]
         */
        const Q8_1Block *V;

        /** Wo weight matrix (opaque pointer - cast based on wo_type) */
        const void *Wo;

        /** Type of Wo weights (FP32, Q8_1, etc.) */
        WoWeightType wo_type;

        // ============== Output tensor ==============

        /**
         * Output buffer in FP32
         * - Non-batched: [seq_len, d_model]
         * - Batched: [batch_size, seq_len, d_model]
         */
        float *output;

        /**
         * Optional context snapshot buffer in FP32 (for debugging/parity testing)
         * When non-null, the kernel writes the pre-Wo attention context here.
         * - Non-batched: [seq_len, num_heads * head_dim]
         * - Batched: [batch_size, seq_len, num_heads * head_dim]
         */
        float *context_snapshot = nullptr;

        // ============== Batch dimensions ==============

        /**
         * Batch size (number of independent sequences)
         * Default: 1 (single sequence mode)
         */
        int batch_size;

        /**
         * Per-sequence KV lengths for variable-length batched decode.
         * - nullptr: All sequences use kv_seq_len
         * - Array of batch_size ints: Each sequence has its own KV length
         *
         * Use case: Batched decode where different sequences have different
         * amounts of context in their KV caches.
         */
        const int *kv_seq_lens;

        /**
         * Per-sequence position offsets for batched decode with causal masking.
         * - nullptr: All sequences use position_offset
         * - Array of batch_size ints: Each sequence has its own position offset
         */
        const int *position_offsets;

        // ============== Sequence dimensions ==============

        /** Sequence length (number of query positions per batch item) */
        int seq_len;

        /**
         * KV sequence length
         * - Non-batched: Actual KV length
         * - Batched uniform: KV length for all sequences
         * - Batched variable: Maximum KV length (for stride calculation)
         */
        int kv_seq_len;

        // ============== Model dimensions ==============

        /** Number of attention heads */
        int num_heads;

        /** Number of KV heads (for GQA/MQA) */
        int num_kv_heads;

        /** Dimension per head (typically 64 or 128) */
        int head_dim;

        /** Model dimension (num_heads * head_dim) */
        int d_model;

        // ============== Attention config ==============

        /** Attention scaling factor: 1/sqrt(head_dim) */
        float scale;

        /** Whether to apply causal masking (decode) */
        bool causal;

        /**
         * Position offset for causal masking (decode mode)
         * Used when kv_seq_lens is nullptr (uniform batching)
         */
        int position_offset;

        // ============== Default initialization ==============
        FusedAttentionWoParams()
            : Q(nullptr), K(nullptr), V(nullptr), Wo(nullptr), wo_type(WoWeightType::Q8_1), output(nullptr), batch_size(1), kv_seq_lens(nullptr), position_offsets(nullptr), seq_len(0), kv_seq_len(0), num_heads(0), num_kv_heads(0), head_dim(64), d_model(0), scale(0.125f) // 1/sqrt(64)
              ,
              causal(true), position_offset(0)
        {
        }

        // ============== Helper methods ==============

        /** Check if this is a batched execution */
        bool is_batched() const { return batch_size > 1; }

        /** Check if using variable KV lengths */
        bool has_variable_kv_lens() const { return kv_seq_lens != nullptr; }

        /** Get KV length for a specific batch item */
        int get_kv_len(int batch_idx) const
        {
            if (kv_seq_lens != nullptr)
            {
                return kv_seq_lens[batch_idx];
            }
            return kv_seq_len;
        }

        /** Get position offset for a specific batch item */
        int get_position_offset(int batch_idx) const
        {
            if (position_offsets != nullptr)
            {
                return position_offsets[batch_idx];
            }
            return position_offset;
        }
    };

    /**
     * @brief Reference implementation of fused attention + Wo projection.
     *
     * This implementation composes the individual microkernels:
     * - μK1: Q8DotProduct (Q·K computation)
     * - μK2: OnlineSoftmax (streaming softmax)
     * - μK3: VWeightedAccum (weighted V accumulation)
     * - μK4: WoProjection (output projection)
     * - μK5: FastExp (exponential for softmax)
     *
     * Supports three execution modes:
     * 1. Single sequence prefill (seq_len = prompt_length)
     * 2. Single sequence decode (seq_len = 1, with position_offset)
     * 3. Batched decode (batch_size > 1, variable KV lengths per sequence)
     *
     * The reference implementation prioritizes correctness and clarity.
     * A JIT implementation will be created for production performance.
     */
    class FusedAttentionWoRef
    {
    public:
        /**
         * @brief Execute the fused attention + Wo projection.
         *
         * @param params Kernel parameters (inputs, outputs, dimensions)
         * @return true on success, false on error
         *
         * Flow:
         * 1. For each batch item b:
         *    a. For each query position m:
         *       i. For each attention head h:
         *          - Compute Q·K scores for all K positions (μK1)
         *          - Online softmax to get weights (μK2 + μK5)
         *          - Accumulate weighted V into FP32 context (μK3)
         *          - Project context through Wo into output (μK4)
         */
        static bool execute(const FusedAttentionWoParams &params);

        /**
         * @brief Execute for a single head (useful for testing).
         *
         * @param params Kernel parameters
         * @param query_pos Query position index [0, seq_len)
         * @param head_idx Head index [0, num_heads)
         * @param context_buffer Pre-allocated buffer [head_dim] for intermediate context
         * @param output_buffer Output slice [d_model] to accumulate into
         * @param batch_idx Batch index (default 0 for non-batched)
         */
        static void execute_single_head(
            const FusedAttentionWoParams &params,
            int query_pos,
            int head_idx,
            float *context_buffer,
            float *output_buffer,
            int batch_idx = 0);

        /**
         * @brief Get the required context buffer size.
         *
         * @param head_dim Dimension per attention head
         * @return Size in floats for the context buffer
         */
        static size_t context_buffer_size(int head_dim)
        {
            return static_cast<size_t>(head_dim);
        }

        /**
         * @brief Validate parameters before execution.
         *
         * @param params Parameters to validate
         * @return true if parameters are valid, false otherwise
         */
        static bool validate_params(const FusedAttentionWoParams &params);

    private:
        /**
         * @brief Process one batch item (internal implementation).
         *
         * @param params Kernel parameters
         * @param batch_idx Batch index
         * @param context_buffer Scratch buffer for FP32 context [head_dim]
         */
        static void process_batch_item(
            const FusedAttentionWoParams &params,
            int batch_idx,
            float *context_buffer);

        /**
         * @brief Process one query head (internal implementation).
         *
         * @param params Kernel parameters
         * @param batch_idx Batch index
         * @param m Query position
         * @param h Head index
         * @param context Scratch buffer for FP32 context [head_dim]
         */
        static void process_head(
            const FusedAttentionWoParams &params,
            int batch_idx,
            int m,
            int h,
            float *context);

        /**
         * @brief Get the KV head index for a given query head (GQA/MQA support).
         *
         * @param query_head Query head index
         * @param num_heads Number of query heads
         * @param num_kv_heads Number of KV heads
         * @return Corresponding KV head index
         */
        static int get_kv_head(int query_head, int num_heads, int num_kv_heads)
        {
            // GQA: multiple query heads share one KV head
            // MHA: 1:1 mapping (num_heads == num_kv_heads)
            // MQA: all query heads share one KV head (num_kv_heads == 1)
            return query_head / (num_heads / num_kv_heads);
        }

        /**
         * @brief Compute tensor offsets for batch indexing.
         */
        struct BatchOffsets
        {
            size_t q_batch_stride;      // Elements between batch items in Q
            size_t kv_batch_stride;     // Elements between batch items in K/V
            size_t output_batch_stride; // Elements between batch items in output
        };

        static BatchOffsets compute_batch_offsets(const FusedAttentionWoParams &params);
    };

} // namespace llaminar::v2::kernels
