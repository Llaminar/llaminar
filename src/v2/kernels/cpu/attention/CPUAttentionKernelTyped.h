/**
 * @file CPUAttentionKernelTyped.h
 * @brief Typed CPU attention kernel supporting all activation precisions
 *
 * Replaces CpuAttentionKernelT with a precision-agnostic template implementation
 * based on ActivationPrecision enum.
 *
 * Benefits:
 * - Single implementation supports FP32, BF16, FP16, Q8_1 (zero code duplication)
 * - Compile-time dispatch via ActivationTraits (zero runtime overhead)
 * - Type-safe (compile errors for unsupported types)
 * - Eliminates dummy tensor creation (uses traits directly)
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/SIMDHelpers.h"
#include "../CPUKernelBase.h"
#include "../primitives/ActivationTraits.h"
#include "../primitives/SoftmaxPrimitivesImpl.h"
#include "../gemm_v4/QuantisedGemmJit_Q8_1_OnlineSoftmax.h"
#include "../gemm_v4/QuantisedGemmJit_FP32_x_Q8_1.h"
#include "../gemm_v4/QuantisedAttentionJit_Q8_1_Fused.h"
#include "../gemm_v4/AttentionInputDumper.h"
#include "../../../pipelines/AttentionUtils.h"
#include "../../../pipelines/PipelineConfig.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"
#include <memory>
#include <vector>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include <mutex>
#include <omp.h>

namespace llaminar2
{
    namespace detail
    {
        template <ActivationPrecision P>
        struct PrecisionToTensor;
        template <>
        struct PrecisionToTensor<ActivationPrecision::FP32>
        {
            using Type = FP32Tensor;
        };
        template <>
        struct PrecisionToTensor<ActivationPrecision::BF16>
        {
            using Type = BF16Tensor;
        };
        template <>
        struct PrecisionToTensor<ActivationPrecision::FP16>
        {
            using Type = FP16Tensor;
        };
        template <>
        struct PrecisionToTensor<ActivationPrecision::Q8_1>
        {
            using Type = Q8_1Tensor;
        };
        // INT32 is not an ActivationPrecision enum value usually used for this, but check Enums.h if needed.
        // Assuming Q8_1 covers the quantized case.
    }

    /**
     * @brief Template-based CPU attention kernel
     *
     * @tparam Precision Activation precision (FP32, BF16, FP16, Q8_1)
     *
     * Uses ActivationTraits<TensorT> for precision-specific operations:
     * - apply_softmax(): Native precision softmax (FP32/BF16/FP16) or conversion (INT32)
     * - create_activation_gemm(): Precision-specific GEMM kernel
     * - allocate_workspace(): Precision-specific workspace allocation
     */
    template <ActivationPrecision Precision>
    class CPUAttentionKernelTyped : public ITensorAttention, public CPUKernelBase
    {
    public:
        using TensorT = typename detail::PrecisionToTensor<Precision>::Type;
        // Infer ElementType from ActivationTraits (which knows the mapping)
        using ElementType = typename primitives::ActivationTraits<TensorT>::ElementType;
        using Traits = primitives::ActivationTraits<TensorT>;

        CPUAttentionKernelTyped() = default;
        ~CPUAttentionKernelTyped() override = default;

        /**
         * @brief Check if kernel supports specific device
         *
         * @param device_idx Device index (-1 = CPU, ≥0 = GPU)
         * @return true if device_idx == -1 (CPU only)
         */
        bool supports_device(int device_idx) const override
        {
            return device_idx == -1; // CPU only
        }

        /**
         * @brief Compute attention (ITensorAttention interface implementation)
         *
         * Takes float* parameters (interface requirement) and casts to ElementType* internally.
         * For FP32Tensor: ElementType=float, no cast needed
         * For BF16/FP16/INT32Tensor: ElementType=uint16_t/int32_t, requires cast
         *
         * Algorithm:
         * 1. Broadcast K/V if GQA/MQA (n_heads != n_kv_heads)
         * 2. Per-head Q@K^T with fused scaling (1/sqrt(d_k)) via GEMM alpha
         * 3. Softmax with optional causal masking (native precision via traits)
         * 4. Context = scores@V using strided GEMM (zero-copy multi-head)
         */
        bool compute(
            const float *Q, const float *K, const float *V,
            float *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_buffer = nullptr,
            TensorBase *workspace_context = nullptr,
            TensorBase *workspace_mask = nullptr,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            // Cast inputs to ElementType
            const ElementType *Q_typed = reinterpret_cast<const ElementType *>(Q);
            const ElementType *K_typed = reinterpret_cast<const ElementType *>(K);
            const ElementType *V_typed = reinterpret_cast<const ElementType *>(V);

            // Allocate workspaces if not provided
            std::shared_ptr<TensorBase> scores_alloc, buffer_alloc, context_alloc;

            float *scores_ptr = nullptr;
            float *buffer_ptr = nullptr;
            float *context_ptr = nullptr;
            const float *mask_ptr = nullptr;

            if (!workspace_scores)
            {
                scores_alloc = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(n_heads * seq_len * seq_len)});
                scores_ptr = scores_alloc->mutable_data();
            }
            else
            {
                scores_ptr = workspace_scores->mutable_data();
            }

            // buffer_ptr is no longer used (Virtual GQA)

            if (!workspace_context)
            {
                // Not strictly needed if writing directly to output, but kept for API compatibility
                // compute_typed writes directly to output
            }
            else
            {
                context_ptr = workspace_context->mutable_data();
            }

            if (workspace_mask)
            {
                mask_ptr = workspace_mask->data();
            }

            return compute_typed(
                Q_typed, K_typed, V_typed, output,
                seq_len, n_heads, n_kv_heads, head_dim,
                causal, window_size,
                scores_ptr, context_ptr,
                mask_ptr,
                device_idx,
                seq_len); // kv_len = seq_len for standard compute
        }

        /**
         * @brief Compute attention with separate sequence and KV lengths (Decoding)
         */
        bool compute_decode(
            const float *Q, const float *K, const float *V,
            float *output,
            int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_buffer = nullptr,
            TensorBase *workspace_context = nullptr,
            TensorBase *workspace_mask = nullptr,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1)
        {
            const ElementType *Q_typed = reinterpret_cast<const ElementType *>(Q);
            const ElementType *K_typed = reinterpret_cast<const ElementType *>(K);
            const ElementType *V_typed = reinterpret_cast<const ElementType *>(V);

            // Allocate workspaces if not provided
            std::shared_ptr<TensorBase> scores_alloc;
            float *scores_ptr = nullptr;
            float *context_ptr = nullptr;
            const float *mask_ptr = nullptr;

            if (!workspace_scores)
            {
                scores_alloc = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(n_heads * seq_len * kv_len)});
                scores_ptr = scores_alloc->mutable_data();
            }
            else
            {
                scores_ptr = workspace_scores->mutable_data();
            }

            if (workspace_context)
            {
                context_ptr = workspace_context->mutable_data();
            }

            if (workspace_mask)
            {
                mask_ptr = workspace_mask->data();
            }

            return compute_typed(
                Q_typed, K_typed, V_typed, output,
                seq_len, n_heads, n_kv_heads, head_dim,
                causal, window_size,
                scores_ptr, context_ptr,
                mask_ptr,
                device_idx,
                kv_len);
        }

        /**
         * @brief Native Q8_1 single-sequence attention (override for Q8_1 specialization)
         *
         * Only implemented for Q8_1 precision - returns false for all other precisions.
         * This avoids ugly reinterpret_cast at the call site.
         */
        bool compute_q8_1(
            const void *Q, const void *K, const void *V, void *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_mask = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            (void)mpi_ctx;

            // Only Q8_1 precision supports this path
            if constexpr (!std::is_same_v<TensorT, Q8_1Tensor>)
            {
                LOG_ERROR("[CPUAttentionKernelTyped] compute_q8_1 called on non-Q8_1 kernel");
                return false;
            }
            else
            {
                const Q8_1Block *Q_blocks = static_cast<const Q8_1Block *>(Q);
                const Q8_1Block *K_blocks = static_cast<const Q8_1Block *>(K);
                const Q8_1Block *V_blocks = static_cast<const Q8_1Block *>(V);
                Q8_1Block *output_blocks = static_cast<Q8_1Block *>(output);

                const float *mask_ptr = workspace_mask ? workspace_mask->data() : nullptr;

                return compute_q8_1_native(
                    Q_blocks, K_blocks, V_blocks,
                    reinterpret_cast<float *>(output_blocks), // Internal API still uses float*
                    seq_len, n_heads, n_kv_heads, head_dim,
                    causal, window_size,
                    nullptr, // scores (not used by fused kernel)
                    nullptr, // context (not used)
                    mask_ptr,
                    seq_len,  // kv_len = seq_len
                    nullptr); // gemm_fallback (not used)
            }
        }

        /**
         * @brief Native Q8_1 batched attention (override for Q8_1 specialization)
         *
         * Only implemented for Q8_1 precision - returns false for all other precisions.
         */
        bool compute_batch_q8_1(
            const void *Q, const void *K, const void *V, void *output,
            int batch_size, int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_mask = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            (void)mpi_ctx;
            (void)device_idx;

            // Only Q8_1 precision supports this path
            if constexpr (!std::is_same_v<TensorT, Q8_1Tensor>)
            {
                LOG_ERROR("[CPUAttentionKernelTyped] compute_batch_q8_1 called on non-Q8_1 kernel");
                return false;
            }
            else
            {
                const Q8_1Block *Q_blocks = static_cast<const Q8_1Block *>(Q);
                const Q8_1Block *K_blocks = static_cast<const Q8_1Block *>(K);
                const Q8_1Block *V_blocks = static_cast<const Q8_1Block *>(V);
                Q8_1Block *output_blocks = static_cast<Q8_1Block *>(output);

                const float *mask_ptr = workspace_mask ? workspace_mask->data() : nullptr;

                // Process each batch item
                const int head_dim_blocks = head_dim / Q8_1Block::BLOCK_SIZE;
                const size_t q_batch_stride = seq_len * n_heads * head_dim_blocks;
                const size_t kv_batch_stride = seq_len * n_kv_heads * head_dim_blocks;
                const size_t out_batch_stride = seq_len * n_heads * head_dim_blocks;
                const int total_tokens = batch_size * seq_len;

                // Mask tile buffer for extracting per-batch mask
                std::vector<float> mask_tile(mask_ptr ? seq_len * seq_len : 0);

                for (int b = 0; b < batch_size; ++b)
                {
                    const Q8_1Block *Q_slice = Q_blocks + b * q_batch_stride;
                    const Q8_1Block *K_slice = K_blocks + b * kv_batch_stride;
                    const Q8_1Block *V_slice = V_blocks + b * kv_batch_stride;
                    Q8_1Block *out_slice = output_blocks + b * out_batch_stride;

                    // Extract per-batch mask tile if needed
                    const float *mask_slice = nullptr;
                    if (mask_ptr)
                    {
                        const int row_offset = b * seq_len;
                        const int col_offset = b * seq_len;
                        for (int i = 0; i < seq_len; ++i)
                        {
                            for (int j = 0; j < seq_len; ++j)
                            {
                                mask_tile[i * seq_len + j] = mask_ptr[(row_offset + i) * total_tokens + (col_offset + j)];
                            }
                        }
                        mask_slice = mask_tile.data();
                    }

                    bool success = compute_q8_1_native(
                        Q_slice, K_slice, V_slice,
                        reinterpret_cast<float *>(out_slice),
                        seq_len, n_heads, n_kv_heads, head_dim,
                        causal, window_size,
                        nullptr, nullptr, mask_slice,
                        seq_len, nullptr);

                    if (!success)
                        return false;
                }

                return true;
            }
        }

    private:
        /**
         * @brief Q8_1 attention with FP32 Q·K scores for improved precision
         *
         * Hybrid approach that dequantizes Q and K to FP32 for score computation,
         * then uses optimized FP32×Q8_1 path for V accumulation.
         *
         * Benefits:
         * - Higher precision Q·K dot products (no quantization noise in scores)
         * - Softmax operates on exact FP32 scores (reduces amplification of errors)
         * - V accumulation still benefits from Q8_1 memory bandwidth savings
         *
         * Performance Impact: ~10-20% slower than pure integer path
         * Precision Impact: ~0.999 cosine similarity at ATTENTION_CONTEXT vs ~0.89
         */
        bool compute_q8_1_fp32_scores(
            const Q8_1Block *Q, const Q8_1Block *K, const Q8_1Block *V,
            Q8_1Block *output_q8,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal,
            const float *mask,
            int kv_len) const
        {
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            const int heads_per_kv = n_heads / n_kv_heads;

            // Q8_1 block layout
            const int head_dim_blocks = (head_dim + 31) / 32;
            const int q_blocks_per_row = n_heads * head_dim_blocks;
            const int k_blocks_per_row = n_kv_heads * head_dim_blocks;
            const int out_blocks_per_row = n_heads * head_dim_blocks;

            // Allocate FP32 workspace for dequantized Q and K
            // Per-thread to avoid false sharing
            const int max_threads = omp_get_max_threads();
            std::vector<std::vector<float>> Q_fp32_threads(max_threads);
            std::vector<std::vector<float>> K_fp32_threads(max_threads);
            std::vector<std::vector<float>> scores_threads(max_threads);
            std::vector<std::vector<float>> context_threads(max_threads);

            for (int t = 0; t < max_threads; ++t)
            {
                Q_fp32_threads[t].resize(seq_len * head_dim);
                K_fp32_threads[t].resize(kv_len * head_dim);
                scores_threads[t].resize(seq_len * kv_len);
                context_threads[t].resize(seq_len * head_dim);
            }

            static bool logged = false;
            if (!logged)
            {
                LOG_INFO("[CPUAttentionKernelTyped] Using FP32 scores mode for Q8_1 attention (LLAMINAR_Q8_ATTENTION_FP32_SCORES=1)");
                LOG_INFO("[CPUAttentionKernelTyped] FP32 scores params: seq_len=" << seq_len << " kv_len=" << kv_len
                                                                                  << " n_heads=" << n_heads << " head_dim=" << head_dim << " causal=" << causal);
                logged = true;
            }

#pragma omp parallel for
            for (int h = 0; h < n_heads; ++h)
            {
                int tid = omp_get_thread_num();
                int kv_h = h / heads_per_kv;

                float *Q_fp32 = Q_fp32_threads[tid].data();
                float *K_fp32 = K_fp32_threads[tid].data();
                float *scores_fp32 = scores_threads[tid].data();
                float *context_fp32 = context_threads[tid].data();

                // 1. Dequantize Q for this head to FP32
                for (int m = 0; m < seq_len; ++m)
                {
                    const Q8_1Block *Q_row = Q + m * q_blocks_per_row + h * head_dim_blocks;
                    float *Q_fp32_row = Q_fp32 + m * head_dim;
                    for (int b = 0; b < head_dim_blocks; ++b)
                    {
                        Q8_1Tensor::decodeBlock(Q_row[b], Q_fp32_row + b * 32);
                    }
                }

                // 2. Dequantize K for this KV head to FP32
                for (int n = 0; n < kv_len; ++n)
                {
                    const Q8_1Block *K_row = K + n * k_blocks_per_row + kv_h * head_dim_blocks;
                    float *K_fp32_row = K_fp32 + n * head_dim;
                    for (int b = 0; b < head_dim_blocks; ++b)
                    {
                        Q8_1Tensor::decodeBlock(K_row[b], K_fp32_row + b * 32);
                    }
                }

                // 3. Compute Q @ K^T in FP32 with scaling
                for (int m = 0; m < seq_len; ++m)
                {
                    const float *Q_row = Q_fp32 + m * head_dim;
                    for (int n = 0; n < kv_len; ++n)
                    {
                        const float *K_row = K_fp32 + n * head_dim;
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; ++d)
                        {
                            dot += Q_row[d] * K_row[d];
                        }
                        float score = dot * scale;

                        // Apply mask if present
                        if (mask)
                        {
                            score += mask[m * kv_len + n];
                        }

                        // Apply causal mask (positions n > m are masked)
                        if (causal && n > m)
                        {
                            score = -std::numeric_limits<float>::infinity();
                        }

                        scores_fp32[m * kv_len + n] = score;
                    }
                }

                // 4. Apply softmax row-wise (in FP32 for precision)
                for (int m = 0; m < seq_len; ++m)
                {
                    float *row = scores_fp32 + m * kv_len;

                    // Find max for numerical stability
                    float max_val = row[0];
                    for (int n = 1; n < kv_len; ++n)
                    {
                        if (row[n] > max_val)
                            max_val = row[n];
                    }

                    // Compute exp and sum
                    float sum = 0.0f;
                    for (int n = 0; n < kv_len; ++n)
                    {
                        row[n] = std::exp(row[n] - max_val);
                        sum += row[n];
                    }

                    // Normalize
                    float inv_sum = 1.0f / sum;
                    for (int n = 0; n < kv_len; ++n)
                    {
                        row[n] *= inv_sum;
                    }
                }

                // 5. Compute context = scores @ V
                // Dequantize V on-the-fly and accumulate in FP32
                std::memset(context_fp32, 0, seq_len * head_dim * sizeof(float));
                for (int m = 0; m < seq_len; ++m)
                {
                    const float *weights = scores_fp32 + m * kv_len;
                    float *context_row = context_fp32 + m * head_dim;

                    for (int n = 0; n < kv_len; ++n)
                    {
                        float w = weights[n];
                        if (w < 1e-10f)
                            continue; // Skip negligible weights

                        const Q8_1Block *V_row = V + n * k_blocks_per_row + kv_h * head_dim_blocks;
                        for (int b = 0; b < head_dim_blocks; ++b)
                        {
                            float d_v = simd::fp16_to_fp32(V_row[b].d);
                            for (int i = 0; i < 32 && b * 32 + i < head_dim; ++i)
                            {
                                context_row[b * 32 + i] += w * (V_row[b].qs[i] * d_v);
                            }
                        }
                    }
                }

                // 6. Requantize context to Q8_1 output
                for (int m = 0; m < seq_len; ++m)
                {
                    const float *ctx_row = context_fp32 + m * head_dim;
                    Q8_1Block *out_row = output_q8 + m * out_blocks_per_row + h * head_dim_blocks;

                    for (int b = 0; b < head_dim_blocks; ++b)
                    {
                        const float *src = ctx_row + b * 32;
                        Q8_1Block &blk = out_row[b];

                        // Find max abs for scale
                        float max_abs = 0.0f;
                        for (int i = 0; i < 32; ++i)
                        {
                            max_abs = std::max(max_abs, std::abs(src[i]));
                        }

                        float d = max_abs / 127.0f;
                        float inv_d = (d > 1e-10f) ? (1.0f / d) : 0.0f;

                        blk.d = simd::fp32_to_fp16(d);

                        int16_t sum_qs = 0;
                        for (int i = 0; i < 32; ++i)
                        {
                            int8_t q = static_cast<int8_t>(std::round(std::clamp(src[i] * inv_d, -127.0f, 127.0f)));
                            blk.qs[i] = q;
                            sum_qs += q;
                        }
                        blk.sum_qs = sum_qs;
                    }
                }
            }

            return true;
        }

        /**
         * @brief Q8_1-native FULLY FUSED attention computation
         *
         * Uses the QuantisedAttentionJit_Q8_1_Fused kernel which fuses:
         * 1. Q8_1 × Q8_1 dot product (Q @ K^T)
         * 2. Online softmax (NO intermediate score storage!)
         * 3. Online weighted V accumulation (on-the-fly dequant)
         * 4. Online requantization to Q8_1 output
         *
         * Benefits:
         * - NO intermediate FP32 scores matrix (saves O(seq×kv) memory)
         * - V values read once while cache-hot
         * - Output directly in Q8_1 format for downstream ops
         *
         * @note output parameter is reinterpreted as Q8_1Block* for this path!
         */
        bool compute_q8_1_native(
            const Q8_1Block *Q, const Q8_1Block *K, const Q8_1Block *V,
            float *output, // Actually Q8_1Block* in disguise!
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal,
            int window_size,
            float *scores,
            float *context,
            const float *mask,
            int kv_len,
            ITensorGemm *gemm_fallback) const
        {
            (void)window_size;
            (void)context;
            (void)scores; // NOT USED - fused kernel has no intermediate scores!
            (void)gemm_fallback;

            // Output is actually Q8_1Block*, reinterpret
            Q8_1Block *output_q8 = reinterpret_cast<Q8_1Block *>(output);

            // Check if FP32 scores mode is enabled
            const auto &env = debugEnv();
            if (env.attention.fp32_scores)
            {
                return compute_q8_1_fp32_scores(
                    Q, K, V, output_q8,
                    seq_len, n_heads, n_kv_heads, head_dim,
                    causal, mask, kv_len);
            }

            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            const int heads_per_kv = n_heads / n_kv_heads;

            // Q8_1 block layout: head_dim elements = head_dim/32 blocks per head
            const int head_dim_blocks = (head_dim + 31) / 32;
            const int q_blocks_per_row = n_heads * head_dim_blocks;    // Q row stride in blocks
            const int k_blocks_per_row = n_kv_heads * head_dim_blocks; // K row stride in blocks
            const int out_blocks_per_row = n_heads * head_dim_blocks;  // Output row stride

            // Get or create the fused JIT kernel for this head_dim
            // Note: head_dim is typically 64 or 128, so we use a simple cache
            static std::unique_ptr<gemm_v4::QuantisedAttentionJit_Q8_1_Fused> jit_fused_64;
            static std::unique_ptr<gemm_v4::QuantisedAttentionJit_Q8_1_Fused> jit_fused_128;
            static std::mutex jit_mutex;

            gemm_v4::QuantisedAttentionJit_Q8_1_Fused *jit_kernel = nullptr;

            {
                std::lock_guard<std::mutex> lock(jit_mutex);
                if (head_dim == 64)
                {
                    if (!jit_fused_64)
                    {
                        jit_fused_64 = std::make_unique<gemm_v4::QuantisedAttentionJit_Q8_1_Fused>(64);
                    }
                    jit_kernel = jit_fused_64.get();
                }
                else if (head_dim == 128)
                {
                    if (!jit_fused_128)
                    {
                        jit_fused_128 = std::make_unique<gemm_v4::QuantisedAttentionJit_Q8_1_Fused>(128);
                    }
                    jit_kernel = jit_fused_128.get();
                }
            }

            if (jit_kernel)
            {
                static bool logged = false;
                if (!logged)
                {
                    LOG_INFO("[CPUAttentionKernelTyped] Using JIT kernel for head_dim=" << head_dim);
                    logged = true;
                }
            }

            // Fallback to reference implementation for unsupported head_dim
            if (!jit_kernel)
            {
                LOG_WARN("[CPUAttentionKernelTyped] head_dim=" << head_dim
                                                               << " not supported by JIT, falling back to reference implementation");

                // Use reference implementation
#pragma omp parallel for
                for (int h = 0; h < n_heads; ++h)
                {
                    int kv_h = h / heads_per_kv;
                    const Q8_1Block *Q_h = Q + h * head_dim_blocks;
                    const Q8_1Block *K_h = K + kv_h * head_dim_blocks;
                    const Q8_1Block *V_h = V + kv_h * head_dim_blocks;
                    Q8_1Block *out_h = output_q8 + h * head_dim_blocks;

                    gemm_v4::fused_q8_1_attention_reference(
                        Q_h, K_h, V_h, out_h,
                        seq_len, kv_len, head_dim,
                        q_blocks_per_row, k_blocks_per_row, k_blocks_per_row, out_blocks_per_row,
                        scale, mask, kv_len);
                }
                return true;
            }

            // Parallelism: parallelize over heads for small seq_len
            const int max_threads = omp_get_max_threads();
            bool parallelize_heads = (seq_len < 128) || (n_heads * 2 >= max_threads);

            // Debug: Print params once per call (outside parallel region)
            static bool debug_printed = false;
            if (!debug_printed && mask != nullptr)
            {
                LOG_INFO("[compute_q8_1_native] seq_len=" << seq_len << " kv_len=" << kv_len
                                                          << " n_heads=" << n_heads << " n_kv_heads=" << n_kv_heads
                                                          << " head_dim=" << head_dim << " heads_per_kv=" << heads_per_kv
                                                          << " q_blocks_per_row=" << q_blocks_per_row << " k_blocks_per_row=" << k_blocks_per_row
                                                          << " mask=" << (mask ? "yes" : "no") << " mask_stride=" << kv_len);
                debug_printed = true;
            }

#pragma omp parallel for if (parallelize_heads)
            for (int h = 0; h < n_heads; ++h)
            {
                // Virtual GQA: Map head h to kv_head
                int kv_h = h / heads_per_kv;

                // Pointers to head-specific Q8_1 blocks
                const Q8_1Block *Q_h = Q + h * head_dim_blocks;
                const Q8_1Block *K_h = K + kv_h * head_dim_blocks;
                const Q8_1Block *V_h = V + kv_h * head_dim_blocks;
                Q8_1Block *out_h = output_q8 + h * head_dim_blocks;

                // Strides in bytes
                const int q_stride_bytes = q_blocks_per_row * static_cast<int>(sizeof(Q8_1Block));
                const int k_stride_bytes = k_blocks_per_row * static_cast<int>(sizeof(Q8_1Block));
                const int out_stride_bytes = out_blocks_per_row * static_cast<int>(sizeof(Q8_1Block));

                // Build params for fused kernel
                gemm_v4::FusedQ8_1AttentionParams params;
                params.Q = Q_h;
                params.K = K_h;
                params.V = V_h;
                params.output = out_h;
                params.M = seq_len;
                params.N = kv_len;
                params.head_dim = head_dim;
                params.Q_stride_bytes = q_stride_bytes;
                params.K_stride_bytes = k_stride_bytes;
                params.V_stride_bytes = k_stride_bytes; // V has same stride as K
                params.output_stride_bytes = out_stride_bytes;
                params.scale = scale;
                params.mask = mask;
                params.mask_stride = kv_len;

                // Dump inputs for debugging (compile-time optional)
                gemm_v4::dump_attention_inputs(params, h, -1);

                // Execute fused kernel
                auto func = jit_kernel->get_kernel();
                func(&params);
            }

            return true;
        }

        /**
         * @brief Typed implementation of attention
         *
         * @param Q Input Q [seq_len, n_heads, head_dim]
         * @param K Input K [kv_len, n_kv_heads, head_dim]
         * @param V Input V [kv_len, n_kv_heads, head_dim]
         * @param output Output [seq_len, n_heads, head_dim] (always float*)
         * @param scores Workspace for attention scores [n_heads, seq_len, kv_len] (FP32)
         * @param buffer Workspace for KV broadcast [2, seq_len, n_heads, head_dim] (FP32/BF16/etc)
         * @param context Workspace for context (unused, writes to output)
         */
        bool compute_typed(
            const ElementType *Q, const ElementType *K, const ElementType *V,
            float *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal,
            int window_size,
            float *scores,
            float *context,
            const float *mask,
            int device_idx,
            int kv_len) const
        {
            // Validate device
            if (device_idx != -1)
            {
                LOG_ERROR("[CPUAttentionKernelTyped] compute: device_idx must be -1 (CPU), got " << device_idx);
                return false;
            }

            // Validate inputs
            if (!Q || !K || !V || !output)
            {
                LOG_ERROR("[CPUAttentionKernelTyped] compute: null tensor data");
                return false;
            }

            if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || seq_len <= 0 || kv_len <= 0)
            {
                LOG_ERROR("[CPUAttentionKernelTyped] compute: invalid dimensions");
                return false;
            }

            if (n_heads % n_kv_heads != 0)
            {
                LOG_ERROR("[CPUAttentionKernelTyped] compute: n_heads (" << n_heads
                                                                         << ") must be divisible by n_kv_heads (" << n_kv_heads << ")");
                return false;
            }

            // Create GEMM kernel once (reused across heads) using ActivationTraits!
            auto gemm = Traits::create_activation_gemm();

            // ================================================================
            // Q8_1-Native Path: Fully quantized attention computation
            // ================================================================
            // Uses JIT-optimized Q8_1 × Q8_1 GEMM with online softmax for Q @ K^T
            // Uses optimized FP32 scores × Q8_1 V → FP32 for context computation
            if constexpr (std::is_same_v<TensorT, Q8_1Tensor>)
            {
                return compute_q8_1_native(
                    Q, K, V, output,
                    seq_len, n_heads, n_kv_heads, head_dim,
                    causal, window_size,
                    scores, context, mask,
                    kv_len, gemm.get());
            }

            // Optimized path for FP32, BF16, FP16 (Native & Mixed Precision)
            if constexpr (std::is_same_v<TensorT, FP32Tensor> ||
                          std::is_same_v<TensorT, BF16Tensor> ||
                          std::is_same_v<TensorT, FP16Tensor>)
            {
                ActivationFormat format = ActivationFormat::FP32;
                if constexpr (std::is_same_v<TensorT, BF16Tensor>)
                    format = ActivationFormat::BF16;
                else if constexpr (std::is_same_v<TensorT, FP16Tensor>)
                    format = ActivationFormat::FP16;

                // 1. Compute Q @ K^T -> Scores (FP32) + Mask + Softmax
                // No GQA broadcast needed! We handle strides virtually.
                const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
                const int heads_per_kv = n_heads / n_kv_heads;

                // Parallelism strategy:
                // 1. Decoding (seq_len < 128): Always parallelize over heads to hide latency/overhead.
                //    BLAS threading is inefficient for small matrices (M=1).
                // 2. Prefill (seq_len >= 128):
                //    - If we have enough heads to saturate ~50% of cores, parallelize over heads.
                //      (Single-threaded GEMM is efficient, avoids sync overhead).
                //    - If few heads (e.g. 1-4) on many cores, run sequentially to let BLAS use all cores.
                const int max_threads = omp_get_max_threads();
                bool parallelize_heads = true;
                if (seq_len >= 128 && n_heads * 2 < max_threads)
                {
                    parallelize_heads = false;
                }

                // 2. Scores @ V -> Output (FP32)
                std::memset(output, 0, seq_len * n_heads * head_dim * sizeof(float));

#pragma omp parallel for if (parallelize_heads)
                for (int h = 0; h < n_heads; ++h)
                {
                    // --- Step 1: Q @ K^T -> Scores ---
                    float *scores_h = scores + h * seq_len * kv_len;
                    const ElementType *Q_h = Q + h * head_dim;

                    // Virtual GQA: Map head h to kv_head
                    int kv_h = h / heads_per_kv;
                    const ElementType *K_h = K + kv_h * head_dim;

                    const int lda_k = n_heads * head_dim;
                    const int ldb_k = n_kv_heads * head_dim; // Stride of K is based on n_kv_heads
                    const int ldc_k = kv_len;

                    // Optimized path for decoding (seq_len == 1) on FP32
                    if (seq_len == 1 && std::is_same_v<TensorT, FP32Tensor>)
                    {
                        if constexpr (std::is_same_v<TensorT, FP32Tensor>)
                        {
                            // 1. Q @ K^T
                            for (int t = 0; t < kv_len; ++t)
                            {
                                const float *k_ptr = K_h + t * ldb_k;
                                float dot = 0.0f;
#pragma omp simd reduction(+ : dot)
                                for (int d = 0; d < head_dim; ++d)
                                {
                                    dot += Q_h[d] * k_ptr[d];
                                }
                                scores_h[t] = dot * scale;
                                if (mask)
                                    scores_h[t] += mask[t];
                            }

                            // 2. Softmax (Vectorized)
                            // Note: scale is already applied in step 1, so we pass 1.0f here.
                            // Causal masking is also handled in step 1 via the mask add.
                            llaminar2::primitives::softmax_row_fp32(
                                scores_h,
                                kv_len,
                                false, // causal
                                1.0f,  // scale
                                -1     // row_idx
                            );

                            // 3. Scores @ V
                            const float *weights_h = scores_h;
                            const ElementType *V_h = V + kv_h * head_dim;
                            float *output_h = output + h * head_dim;

                            const int ldb_v = n_kv_heads * head_dim;

                            // Initialize output to 0
                            std::memset(output_h, 0, head_dim * sizeof(float));

                            for (int t = 0; t < kv_len; ++t)
                            {
                                float s = weights_h[t];
                                const float *v_ptr = V_h + t * ldb_v;
#pragma omp simd
                                for (int d = 0; d < head_dim; ++d)
                                {
                                    output_h[d] += s * v_ptr[d];
                                }
                            }
                            continue; // Skip generic path
                        }
                    }

                    gemm->template multiply_with_softmax_strided_typed<ElementType, ElementType>(
                        Q_h, K_h, scores_h,
                        seq_len, kv_len, head_dim,
                        lda_k, ldb_k, ldc_k,
                        scale,
                        true, // transpose_B
                        1,    // softmax_axis
                        mask,
                        causal,
                        nullptr, -1, format);

                    // --- Step 2: Scores @ V -> Output ---
                    const float *weights_h = scores_h; // Reuse scores_h
                    const ElementType *V_h = V + kv_h * head_dim;
                    float *output_h = output + h * head_dim;

                    const int lda_v = kv_len;
                    const int ldb_v = n_kv_heads * head_dim; // Stride of V is based on n_kv_heads
                    const int ldc_v = n_heads * head_dim;

                    gemm->template multiply_activations_strided_typed<float, ElementType>(
                        weights_h, V_h, output_h,
                        seq_len, head_dim, kv_len,
                        lda_v, ldb_v, ldc_v,
                        false, 1.0f, 0.0f, nullptr, -1, format);
                }

                return true;
            }

            // 1. Convert Q, K, V to FP32 (required for multiply_activations_strided)
            // CRITICAL: ITensorGemm::multiply_activations_strided expects float* inputs!
            std::vector<float> Q_fp32(seq_len * n_heads * head_dim);
            std::vector<float> K_fp32(kv_len * n_kv_heads * head_dim);
            std::vector<float> V_fp32(kv_len * n_kv_heads * head_dim);

            // Convert ElementType -> FP32 using type-specific conversion
            if constexpr (std::is_same_v<ElementType, float>)
            {
                // FP32: direct copy
                std::memcpy(Q_fp32.data(), Q, Q_fp32.size() * sizeof(float));
                std::memcpy(K_fp32.data(), K, K_fp32.size() * sizeof(float));
                std::memcpy(V_fp32.data(), V, V_fp32.size() * sizeof(float));
            }
            else if constexpr (std::is_same_v<TensorT, BF16Tensor>)
            {
                // BF16: convert using simd::bf16_to_fp32
                for (size_t i = 0; i < Q_fp32.size(); ++i)
                {
                    Q_fp32[i] = simd::bf16_to_fp32(Q[i]);
                }
                for (size_t i = 0; i < K_fp32.size(); ++i)
                {
                    K_fp32[i] = simd::bf16_to_fp32(K[i]);
                }
                for (size_t i = 0; i < V_fp32.size(); ++i)
                {
                    V_fp32[i] = simd::bf16_to_fp32(V[i]);
                }
            }
            else if constexpr (std::is_same_v<TensorT, FP16Tensor>)
            {
                // FP16: convert using simd::fp16_to_fp32
                for (size_t i = 0; i < Q_fp32.size(); ++i)
                {
                    Q_fp32[i] = simd::fp16_to_fp32(Q[i]);
                }
                for (size_t i = 0; i < K_fp32.size(); ++i)
                {
                    K_fp32[i] = simd::fp16_to_fp32(K[i]);
                }
                for (size_t i = 0; i < V_fp32.size(); ++i)
                {
                    V_fp32[i] = simd::fp16_to_fp32(V[i]);
                }
            }
            else if constexpr (std::is_same_v<TensorT, Q8_1Tensor>)
            {
                // Q8_1: decode blocks
                // Q is Q8_1Block*
                // Q_fp32 is float*
                // Each block decodes to 32 floats
                size_t n_blocks = Q_fp32.size() / 32;
                for (size_t i = 0; i < n_blocks; ++i)
                {
                    Q8_1Tensor::decodeBlock(Q[i], Q_fp32.data() + i * 32);
                }

                n_blocks = K_fp32.size() / 32;
                for (size_t i = 0; i < n_blocks; ++i)
                {
                    Q8_1Tensor::decodeBlock(K[i], K_fp32.data() + i * 32);
                }

                n_blocks = V_fp32.size() / 32;
                for (size_t i = 0; i < n_blocks; ++i)
                {
                    Q8_1Tensor::decodeBlock(V[i], V_fp32.data() + i * 32);
                }
            }
            // NOTE: INT32 and Q8_0 are not valid activation precisions.
            // INT32 is an accumulator type, Q8_0 is a weight format.
            // The only supported activation precisions are: FP32, BF16, FP16, Q8_1

            // 2. Compute attention scores per head: Q @ K^T with fused scaling
            // No GQA broadcast needed! We handle strides virtually.
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            const int heads_per_kv = n_heads / n_kv_heads;

            // GEMM kernel already created above

            // Parallelism strategy:
            // 1. Decoding (seq_len < 128): Always parallelize over heads to hide latency/overhead.
            //    BLAS threading is inefficient for small matrices (M=1).
            // 2. Prefill (seq_len >= 128):
            //    - If we have enough heads to saturate ~50% of cores, parallelize over heads.
            //      (Single-threaded GEMM is efficient, avoids sync overhead).
            //    - If few heads (e.g. 1-4) on many cores, run sequentially to let BLAS use all cores.
            const int max_threads = omp_get_max_threads();
            bool parallelize_heads = true;
            if (seq_len >= 128 && n_heads * 2 < max_threads)
            {
                parallelize_heads = false;
            }

#pragma omp parallel for if (parallelize_heads)
            for (int h = 0; h < n_heads; ++h)
            {
                float *scores_h = scores + h * seq_len * seq_len; // FP32 workspace

                // Strided Q@K^T with fused scaling
                const float *Q_h = Q_fp32.data() + h * head_dim; // Q_fp32: already FP32

                // Virtual GQA: Map head h to kv_head
                int kv_h = h / heads_per_kv;
                const float *K_h = K_fp32.data() + kv_h * head_dim; // K_fp32: already FP32

                const int lda = n_heads * head_dim;    // Q: stride between rows (skip other heads)
                const int ldb = n_kv_heads * head_dim; // K: stride between rows (skip other heads)
                const int ldc = seq_len;               // scores: contiguous [seq_len, seq_len]

                // Strided GEMM: scores = (1/sqrt(d_k)) * Q @ K^T
                gemm->multiply_activations_strided(
                    Q_h,                        // FP32 Q head
                    K_h,                        // FP32 K head (after broadcast if GQA)
                    scores_h,                   // FP32 scores output
                    seq_len, seq_len, head_dim, // m=seq_len, n=seq_len, k=head_dim
                    lda, ldb, ldc,
                    true,    // transpose_B=true (K^T)
                    scale,   // alpha=1/sqrt(d_k) - FUSED SCALING!
                    0.0f,    // beta
                    nullptr, // mpi_ctx
                    -1);     // device_idx (CPU)
            }

            if (mask)
            {
#pragma omp parallel for if (n_heads > 1)
                for (int h = 0; h < n_heads; ++h)
                {
                    float *scores_h = scores + h * seq_len * seq_len;
                    attention_utils::apply_attention_mask(scores_h, mask, seq_len, seq_len);
                }
            }

            // 3. Apply softmax with optional causal masking
            // Softmax operates on FP32 workspaces (even for BF16/FP16 inputs)
            // Softmax is lightweight, parallelize even with few heads

#pragma omp parallel for if (n_heads > 1)
            for (int h = 0; h < n_heads; ++h)
            {
                float *scores_h = scores + h * seq_len * seq_len; // FP32 workspace

                // Apply FP32 softmax directly (workspaces are always FP32)
                primitives::softmax_row_major_fp32(scores_h, seq_len, seq_len, causal, 1.0f, true);
            }

            // 4. Compute context: weights @ V using strided GEMM (zero-copy multi-head)
            // Zero output once before parallel region
            std::memset(output, 0, seq_len * n_heads * head_dim * sizeof(float));

            // Reuse GEMM kernel from Q@K^T computation
            // Reuse parallelism strategy

#pragma omp parallel for if (parallelize_heads)
            for (int h = 0; h < n_heads; ++h)
            {
                // NOTE: weights_h points to FP32 workspace (scores)
                const float *weights_h = scores + h * seq_len * seq_len;

                // Virtual GQA: Map head h to kv_head
                int kv_h = h / heads_per_kv;
                const float *V_h = V_fp32.data() + kv_h * head_dim; // V_fp32 is float*

                // CRITICAL: output is float* (not ElementType*), pointer arithmetic in floats
                float *output_h = output + h * head_dim; // First element of head h

                // Strided GEMM: context = weights @ V
                const int lda = seq_len;               // weights: contiguous [seq_len, seq_len]
                const int ldb = n_kv_heads * head_dim; // V: stride between rows (skip other heads)
                const int ldc = n_heads * head_dim;    // output: stride between rows

                // Strided GEMM: context = weights @ V
                gemm->multiply_activations_strided(
                    weights_h,                  // Already float*, no cast needed
                    V_h,                        // Already float*, no cast needed
                    output_h,                   // Already float*, no cast needed
                    seq_len, head_dim, seq_len, // m=seq_len, n=head_dim, k=seq_len
                    lda, ldb, ldc,
                    false,   // transpose_B=false (weights @ V, not V^T)
                    1.0f,    // alpha
                    0.0f,    // beta (output already zeroed)
                    nullptr, // mpi_ctx
                    -1);     // device_idx (CPU)
            }

            return true;
        }

        /**
         * @brief Batch attention computation
         */
    public:
        bool compute_batch(
            const float *Q, const float *K, const float *V,
            float *output,
            int batch_size, int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_buffer = nullptr,
            TensorBase *workspace_context = nullptr,
            TensorBase *workspace_mask = nullptr,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            // Allocate workspaces if not provided
            std::shared_ptr<TensorBase> scores_alloc, buffer_alloc, context_alloc;
            float *scores_ptr = nullptr;
            float *buffer_ptr = nullptr;
            float *context_ptr = nullptr;

            // Per-sequence scores: [batch_size, n_heads, seq_len, seq_len]
            // Flattened: batch_size * n_heads * seq_len * seq_len
            if (!workspace_scores)
            {
                scores_alloc = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(batch_size * n_heads * seq_len * seq_len)});
                scores_ptr = scores_alloc->mutable_data();
            }
            else
            {
                scores_ptr = workspace_scores->mutable_data();
            }

            // buffer_ptr is no longer used (Virtual GQA)

            const float *mask_ptr = nullptr;
            const int total_tokens = batch_size * seq_len;
            if (workspace_mask)
            {
                mask_ptr = workspace_mask->data();
            }

            // Allocate mask tile buffer once for reuse across all batch items
            // The combined mask is [total_tokens, total_tokens], but compute_typed expects [seq_len, seq_len]
            // We extract the relevant [seq_len, seq_len] block per batch item
            std::vector<float> mask_tile;
            if (mask_ptr)
            {
                mask_tile.resize(seq_len * seq_len);
            }

            // Cast inputs to ElementType FIRST (before any pointer arithmetic)
            // This is CRITICAL: Q/K/V are void* in disguise, pointing to native ElementType data
            // For FP32: ElementType=float, no-op cast
            // For BF16: ElementType=uint16_t, must cast before arithmetic to avoid wrong stride
            const ElementType *Q_typed = reinterpret_cast<const ElementType *>(Q);
            const ElementType *K_typed = reinterpret_cast<const ElementType *>(K);
            const ElementType *V_typed = reinterpret_cast<const ElementType *>(V);

            // Loop over batch items
            // Parallelize over batch if batch_size is large enough
            // But compute_typed already parallelizes over heads.
            // For small batch size (e.g. 1-8), sequential loop is fine.
            for (int b = 0; b < batch_size; ++b)
            {
                // Offsets in ELEMENT units (ElementType, not float!)
                // Q, K, V: [batch_size, seq_len, n_heads/n_kv_heads, head_dim] (flattened)
                // Stride per batch: seq_len * n_heads * head_dim
                size_t q_offset = b * seq_len * n_heads * head_dim;
                size_t k_offset = b * seq_len * n_kv_heads * head_dim;
                size_t v_offset = b * seq_len * n_kv_heads * head_dim;
                size_t out_offset = b * seq_len * n_heads * head_dim;

                // Scores: [batch_size, n_heads, seq_len, seq_len] (always float)
                size_t scores_offset = b * n_heads * seq_len * seq_len;

                // Extract per-sequence mask tile from combined batch mask
                // Combined mask: [total_tokens, total_tokens] where total_tokens = batch_size * seq_len
                // For batch item b, tokens span [b*seq_len : (b+1)*seq_len]
                // Extract the [seq_len, seq_len] block at rows [b*seq_len : (b+1)*seq_len], cols [b*seq_len : (b+1)*seq_len]
                const float *mask_slice = nullptr;
                if (mask_ptr)
                {
                    const int row_offset = b * seq_len;
                    const int col_offset = b * seq_len;

                    // Copy the relevant block from combined mask to tile buffer
                    for (int i = 0; i < seq_len; ++i)
                    {
                        for (int j = 0; j < seq_len; ++j)
                        {
                            const int global_row = row_offset + i;
                            const int global_col = col_offset + j;
                            mask_tile[i * seq_len + j] = mask_ptr[global_row * total_tokens + global_col];
                        }
                    }

                    mask_slice = mask_tile.data();
                }

                // Pointer arithmetic in ElementType units (already cast above)
                const ElementType *Q_slice = Q_typed + q_offset;
                const ElementType *K_slice = K_typed + k_offset;
                const ElementType *V_slice = V_typed + v_offset;

                bool success = compute_typed(
                    Q_slice,
                    K_slice,
                    V_slice,
                    output + out_offset,
                    seq_len, n_heads, n_kv_heads, head_dim,
                    causal, window_size,
                    scores_ptr + scores_offset,
                    nullptr, // context not used
                    mask_slice,
                    device_idx,
                    seq_len); // kv_len = seq_len for batch compute

                if (!success)
                    return false;
            }

            return true;
        }
    };

    // Explicit instantiations
    extern template class CPUAttentionKernelTyped<ActivationPrecision::FP32>;
    extern template class CPUAttentionKernelTyped<ActivationPrecision::BF16>;
    extern template class CPUAttentionKernelTyped<ActivationPrecision::FP16>;
    extern template class CPUAttentionKernelTyped<ActivationPrecision::Q8_1>;
} // namespace llaminar2
