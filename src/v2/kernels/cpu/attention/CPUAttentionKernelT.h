/**
 * @file CPUAttentionKernelT.h
 * @brief CPU attention kernel supporting all activation precisions
 *
 * Precision-agnostic template implementation based on ActivationPrecision enum.
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
#include "AttentionUtils.h"
#include "../../../execution/config/RuntimeConfig.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/OpenMPUtils.h"
#include "../../../utils/KernelProfiler.h"
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
    class CPUAttentionKernelT : public ITensorAttention, public CPUKernelBase
    {
    public:
        using TensorT = typename detail::PrecisionToTensor<Precision>::Type;
        // Infer ElementType from ActivationTraits (which knows the mapping)
        using ElementType = typename primitives::ActivationTraits<TensorT>::ElementType;
        using Traits = primitives::ActivationTraits<TensorT>;

        CPUAttentionKernelT() = default;
        ~CPUAttentionKernelT() override = default;

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
            int device_idx = -1)
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
            KERNEL_PROFILE_SCOPE(KernelType::ATTENTION);

            const ElementType *Q_typed = reinterpret_cast<const ElementType *>(Q);
            const ElementType *K_typed = reinterpret_cast<const ElementType *>(K);
            const ElementType *V_typed = reinterpret_cast<const ElementType *>(V);

            // Allocate workspaces if not provided
            float *scores_ptr = nullptr;
            float *context_ptr = nullptr;
            const float *mask_ptr = nullptr;

            if (!workspace_scores)
            {
                thread_local std::vector<float> scores_scratch;
                const size_t needed = static_cast<size_t>(n_heads) * static_cast<size_t>(seq_len) * static_cast<size_t>(kv_len);
                if (scores_scratch.size() < needed)
                {
                    scores_scratch.resize(needed);
                }
                scores_ptr = scores_scratch.data();
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
            int device_idx = -1)
        {
            (void)mpi_ctx;

            // Only Q8_1 precision supports this path
            if constexpr (!std::is_same_v<TensorT, Q8_1Tensor>)
            {
                LOG_ERROR("[CPUAttentionKernelT] compute_q8_1 called on non-Q8_1 kernel");
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
            int device_idx = -1)
        {
            (void)mpi_ctx;
            (void)device_idx;

            // Only Q8_1 precision supports this path
            if constexpr (!std::is_same_v<TensorT, Q8_1Tensor>)
            {
                LOG_ERROR("[CPUAttentionKernelT] compute_batch_q8_1 called on non-Q8_1 kernel");
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
         * @brief Q8_1 attention paths (REMOVED)
         *
         * Q8_1 and Q16_1 attention modes on CPU have been removed.
         * FP32 activation precision is the production path.
         */
        bool compute_q8_1_fp32_scores(
            const Q8_1Block *, const Q8_1Block *, const Q8_1Block *,
            Q8_1Block *,
            int, int, int, int,
            bool,
            const float *,
            int) const
        {
            LOG_ERROR("[CPUAttentionKernelT] Q8_1 FP32 scores path has been removed. Use FP32 activations.");
            return false;
        }

        bool compute_q8_1_native(
            const Q8_1Block *, const Q8_1Block *, const Q8_1Block *,
            float *,
            int, int, int, int,
            bool, int,
            float *, float *,
            const float *,
            int,
            ITensorGemm *) const
        {
            LOG_ERROR("[CPUAttentionKernelT] Q8_1 native attention path has been removed. Use FP32 activations.");
            return false;
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
            // Note: device_idx is informational only for CPU kernels.
            // With multi-socket systems, device_idx can be 0, 1, etc. for different NUMA nodes.
            // We accept any device_idx since CPU kernels don't need GPU context.
            (void)device_idx;

            // Validate inputs
            if (!Q || !K || !V || !output)
            {
                LOG_ERROR("[CPUAttentionKernelT] compute: null tensor data");
                return false;
            }

#ifndef NDEBUG
            // Debug-only: Check for uninitialized tensors (likely allocation without population)
            // This would have caught the V_dequant=0 bug in Hybrid mode immediately
            // Only check FP32 path - quantized types would need different checks
            if constexpr (std::is_same_v<ElementType, float>)
            {
                size_t q_elements = static_cast<size_t>(seq_len) * n_heads * head_dim;
                size_t kv_elements = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;

                // Sample first and last elements for quick zero check
                bool q_zero = (Q[0] == 0.0f && Q[q_elements - 1] == 0.0f);
                bool k_zero = (K[0] == 0.0f && K[kv_elements - 1] == 0.0f);
                bool v_zero = (V[0] == 0.0f && V[kv_elements - 1] == 0.0f);

                if (q_zero)
                    LOG_WARN("[CPUAttentionKernelT] Q tensor appears to be all zeros (likely uninitialized)!");
                if (k_zero)
                    LOG_WARN("[CPUAttentionKernelT] K tensor appears to be all zeros (likely uninitialized)!");
                if (v_zero)
                    LOG_WARN("[CPUAttentionKernelT] V tensor appears to be all zeros (likely uninitialized)!");
            }
#endif

            if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || seq_len <= 0 || kv_len <= 0)
            {
                LOG_ERROR("[CPUAttentionKernelT] compute: invalid dimensions");
                return false;
            }

            if (n_heads % n_kv_heads != 0)
            {
                LOG_ERROR("[CPUAttentionKernelT] compute: n_heads (" << n_heads
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
                // No pre-zeroing needed: both the decode fast path and GEMM path write with beta=0.

                // Capture variables for nested-safe parallel execution
                auto typed_attention_work = [&]()
                {
                    // Debug: Log dimensions once at start of loop (thread 0 only)
                    if (omp_get_thread_num() == 0)
                    {
                        LOG_DEBUG("[compute_typed] FP32 path: seq_len=" << seq_len << " kv_len=" << kv_len
                                                                        << " n_heads=" << n_heads << " n_kv_heads=" << n_kv_heads
                                                                        << " head_dim=" << head_dim << " Q_ptr=" << (void *)Q
                                                                        << " K_ptr=" << (void *)K << " V_ptr=" << (void *)V
                                                                        << " output_ptr=" << (void *)output);
                    }
#pragma omp for schedule(static)
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
                                KERNEL_PROFILE_SCOPE(KernelType::GEMV_FP32);
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
                };
                OMP_WORKSHARE_REGION_IF(typed_attention_work, parallelize_heads);

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

            // Q@K^T GEMM for each head (nested-safe)
            auto qkt_gemm_work = [&]()
            {
#pragma omp for schedule(static)
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
            };
            OMP_WORKSHARE_REGION_IF(qkt_gemm_work, parallelize_heads);

            if (mask)
            {
                auto mask_apply_work = [&]()
                {
#pragma omp for schedule(static)
                    for (int h = 0; h < n_heads; ++h)
                    {
                        float *scores_h = scores + h * seq_len * seq_len;
                        attention_utils::apply_attention_mask(scores_h, mask, seq_len, seq_len);
                    }
                };
                OMP_WORKSHARE_REGION_IF(mask_apply_work, n_heads > 1);
            }

            // 3. Apply softmax with optional causal masking (nested-safe)
            // Softmax operates on FP32 workspaces (even for BF16/FP16 inputs)
            // Softmax is lightweight, parallelize even with few heads
            auto softmax_work = [&]()
            {
#pragma omp for schedule(static)
                for (int h = 0; h < n_heads; ++h)
                {
                    float *scores_h = scores + h * seq_len * seq_len; // FP32 workspace

                    // Apply FP32 softmax directly (workspaces are always FP32)
                    primitives::softmax_row_major_fp32(scores_h, seq_len, seq_len, causal, 1.0f, true);
                }
            };
            OMP_WORKSHARE_REGION_IF(softmax_work, n_heads > 1);

            // 4. Compute context: weights @ V using strided GEMM (zero-copy multi-head)
            // Zero output once before parallel region
            std::memset(output, 0, seq_len * n_heads * head_dim * sizeof(float));

            // Reuse GEMM kernel from Q@K^T computation
            // Reuse parallelism strategy (nested-safe)
            auto weights_v_work = [&]()
            {
#pragma omp for schedule(static)
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
            };
            OMP_WORKSHARE_REGION_IF(weights_v_work, parallelize_heads);

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
            int device_idx = -1)
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

                // DEBUG: Print Q_typed values for this batch (only for FP32)
                if constexpr (std::is_same_v<ElementType, float>)
                {
                    LOG_DEBUG("[CPUAttentionKernelT::compute_batch] batch=" << b << " q_offset=" << q_offset
                                                                            << " k_offset=" << k_offset << " v_offset=" << v_offset
                                                                            << " Q_typed[0..3]=[" << Q_typed[q_offset] << ","
                                                                            << Q_typed[q_offset + 1] << ","
                                                                            << Q_typed[q_offset + 2] << ","
                                                                            << Q_typed[q_offset + 3] << "]"
                                                                            << " K_typed[0..3]=[" << K_typed[k_offset] << ","
                                                                            << K_typed[k_offset + 1] << ","
                                                                            << K_typed[k_offset + 2] << ","
                                                                            << K_typed[k_offset + 3] << "]");
                }

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

                // DEBUG: Print pointer values before compute_typed
                LOG_DEBUG("[CPUAttentionKernelT::compute_batch] batch=" << b
                                                                        << " Q_slice=" << (void *)Q_slice
                                                                        << " K_slice=" << (void *)K_slice
                                                                        << " V_slice=" << (void *)V_slice
                                                                        << " output_ptr=" << (void *)(output + out_offset));

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

                // DEBUG: Print output values after compute_typed
                LOG_DEBUG("[CPUAttentionKernelT::compute_batch] batch=" << b << " out_offset=" << out_offset
                                                                        << " output[0..3]=[" << output[out_offset] << "," << output[out_offset + 1]
                                                                        << "," << output[out_offset + 2] << "," << output[out_offset + 3] << "]");

                if (!success)
                    return false;
            }

            return true;
        }

        /**
         * @brief Compute attention using tensor objects with automatic type dispatch
         *
         * Inspects Q tensor native_type() and dispatches to compute() or compute_batch()
         * based on batch_size. Handles FP32, BF16, FP16, and Q8_1 tensors.
         *
         * @param head_start First query head to compute (0-indexed, for tensor parallelism)
         * @param local_n_heads Number of query heads to compute (-1 = all heads)
         * @param local_n_kv_heads Number of KV heads for this slice (-1 = all KV heads)
         *
         * For tensor parallelism across N ranks:
         *   Rank i: head_start = i * (n_heads/N), local_n_heads = n_heads/N
         * The caller is responsible for AllGather to combine output slices.
         *
         * Note: Accepts ITensor* for interface compatibility but requires TensorBase*
         * underneath (CPU tensors only). GPU tensors will fail with an error.
         */
        bool compute_tensor(
            const ITensor *Q,
            const ITensor *K,
            const ITensor *V,
            ITensor *output,
            int batch_size,
            int seq_len,
            int kv_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            bool causal = false,
            int window_size = -1,
            ITensor *workspace_scores = nullptr,
            ITensor *workspace_mask = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            int head_start = 0,
            int local_n_heads = -1,
            int local_n_kv_heads = -1) override
        {
            // Cast ITensor* to TensorBase* - CPU kernel requires CPU tensors
            const TensorBase *Q_base = dynamic_cast<const TensorBase *>(Q);
            const TensorBase *K_base = dynamic_cast<const TensorBase *>(K);
            const TensorBase *V_base = dynamic_cast<const TensorBase *>(V);
            TensorBase *output_base = dynamic_cast<TensorBase *>(output);
            TensorBase *workspace_scores_base = dynamic_cast<TensorBase *>(workspace_scores);
            TensorBase *workspace_mask_base = dynamic_cast<TensorBase *>(workspace_mask);

            if ((Q && !Q_base) || (K && !K_base) || (V && !V_base) || (output && !output_base))
            {
                LOG_ERROR("[CPUAttentionKernelT::compute_tensor] GPU tensors not supported - "
                          << "requires CPU TensorBase");
                return false;
            }

            // Resolve defaults: -1 means use all heads
            const int actual_local_n_heads = (local_n_heads < 0) ? n_heads : local_n_heads;
            const int actual_local_n_kv_heads = (local_n_kv_heads < 0) ? n_kv_heads : local_n_kv_heads;

            // Validate head ranges
            if (head_start < 0 || head_start >= n_heads)
            {
                LOG_ERROR("[CPUAttentionKernelT::compute_tensor] Invalid head_start: "
                          << head_start << " (n_heads=" << n_heads << ")");
                return false;
            }
            if (head_start + actual_local_n_heads > n_heads)
            {
                LOG_ERROR("[CPUAttentionKernelT::compute_tensor] head_start + local_n_heads exceeds n_heads: "
                          << head_start << " + " << actual_local_n_heads << " > " << n_heads);
                return false;
            }

            // TODO: For now, we only support full head range (no TP slicing in attention)
            // When head-sliced attention is implemented, the compute/compute_decode/compute_batch
            // functions will need to accept head_start/local_n_heads parameters and adjust their
            // head loops accordingly. This requires updating the inner loop bounds from:
            //   for (int h = 0; h < n_heads; ++h)
            // to:
            //   for (int h = head_start; h < head_start + local_n_heads; ++h)
            //
            // For now, log a warning if TP slicing is requested but proceed with full computation
            if (head_start != 0 || actual_local_n_heads != n_heads)
            {
                LOG_WARN("[CPUAttentionKernelT::compute_tensor] Head slicing requested (head_start="
                         << head_start << ", local_n_heads=" << actual_local_n_heads
                         << ") but not yet implemented. Computing all " << n_heads << " heads.");
            }

            // Suppress unused variable warnings for TP parameters until implemented
            (void)actual_local_n_kv_heads;
            if (!Q_base || !K_base || !V_base || !output_base)
            {
                LOG_ERROR("[CPUAttentionKernelT::compute_tensor] Null tensor provided");
                return false;
            }

            // Map ActivationPrecision to TensorType
            constexpr TensorType expected_type = []()
            {
                if constexpr (Precision == ActivationPrecision::FP32)
                    return TensorType::FP32;
                else if constexpr (Precision == ActivationPrecision::BF16)
                    return TensorType::BF16;
                else if constexpr (Precision == ActivationPrecision::FP16)
                    return TensorType::FP16;
                else if constexpr (Precision == ActivationPrecision::Q8_1)
                    return TensorType::Q8_1;
                else
                    return TensorType::FP32; // Default
            }();

            if (Q_base->native_type() != expected_type)
            {
                LOG_ERROR("[CPUAttentionKernelT::compute_tensor] Q tensor type mismatch: expected "
                          << static_cast<int>(expected_type) << ", got " << static_cast<int>(Q_base->native_type()));
                return false;
            }

            // Get raw data pointers based on precision
            const float *Q_ptr = nullptr;
            const float *K_ptr = nullptr;
            const float *V_ptr = nullptr;
            float *output_ptr = nullptr;

            if constexpr (Precision == ActivationPrecision::Q8_1)
            {
                // Q8_1 path: use block pointers cast to float* (kernel interprets internally)
                auto Q_q8_1 = static_cast<const Q8_1Tensor *>(Q_base);
                auto K_q8_1 = static_cast<const Q8_1Tensor *>(K_base);
                auto V_q8_1 = static_cast<const Q8_1Tensor *>(V_base);
                auto out_q8_1 = static_cast<Q8_1Tensor *>(output_base);

                Q_ptr = reinterpret_cast<const float *>(Q_q8_1->typed_data());
                K_ptr = reinterpret_cast<const float *>(K_q8_1->typed_data());
                V_ptr = reinterpret_cast<const float *>(V_q8_1->typed_data());
                output_ptr = reinterpret_cast<float *>(out_q8_1->mutable_typed_data());
            }
            else
            {
                // FP32/BF16/FP16 path: use explicit FP32 views for inputs.
                // This avoids accidental calls into deprecated quantized data() paths
                // (e.g. Q8_1Tensor::data()) when mixed-precision tensors are routed here.
                Q_ptr = Q_base->fp32_data();
                K_ptr = K_base->fp32_data();
                V_ptr = V_base->fp32_data();
                output_ptr = output_base->mutable_data();
            }

            // Dispatch to batch or single-sequence compute
            if (batch_size > 1)
            {
                return compute_batch(
                    Q_ptr, K_ptr, V_ptr, output_ptr,
                    batch_size, seq_len, n_heads, n_kv_heads, head_dim,
                    causal, window_size,
                    workspace_scores_base, nullptr, nullptr, workspace_mask_base,
                    false, mpi_ctx, device_idx);
            }
            else
            {
                // Single sequence: use compute() with kv_len support via compute_decode if different
                if (kv_len != seq_len)
                {
                    return compute_decode(
                        Q_ptr, K_ptr, V_ptr, output_ptr,
                        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                        causal, window_size,
                        workspace_scores_base, nullptr, nullptr, workspace_mask_base,
                        false, mpi_ctx, device_idx);
                }
                else
                {
                    return compute(
                        Q_ptr, K_ptr, V_ptr, output_ptr,
                        seq_len, n_heads, n_kv_heads, head_dim,
                        causal, window_size,
                        workspace_scores_base, nullptr, nullptr, workspace_mask_base,
                        false, mpi_ctx, device_idx);
                }
            }
        }

        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::attention()
                .withInput("Q", "query tensor [seq_len, n_heads * head_dim]", KernelBufferDtype::FP32)
                .withInput("K", "key tensor [kv_len, n_kv_heads * head_dim]", KernelBufferDtype::FP32)
                .withInput("V", "value tensor [kv_len, n_kv_heads * head_dim]", KernelBufferDtype::FP32)
                .withOutput("scores", "attention scores [n_heads, seq_len, kv_len]", KernelBufferDtype::FP32, true)
                .withOutput("context", "attention context [seq_len, n_heads * head_dim]", KernelBufferDtype::FP32, true)
                .withOutput("output", "attention output [seq_len, n_heads * head_dim]", KernelBufferDtype::FP32)
                .withScalar("seq_len", "query sequence length", KernelBufferDtype::INT32)
                .withScalar("kv_len", "key/value sequence length", KernelBufferDtype::INT32)
                .withScalar("n_heads", "number of query heads", KernelBufferDtype::INT32)
                .withScalar("n_kv_heads", "number of key/value heads", KernelBufferDtype::INT32)
                .withScalar("head_dim", "dimension per head", KernelBufferDtype::INT32)
                .withScalar("causal", "apply causal masking", KernelBufferDtype::INT32);
        }
    };

    // Explicit instantiations
    extern template class CPUAttentionKernelT<ActivationPrecision::FP32>;
    extern template class CPUAttentionKernelT<ActivationPrecision::BF16>;
    extern template class CPUAttentionKernelT<ActivationPrecision::FP16>;
    extern template class CPUAttentionKernelT<ActivationPrecision::Q8_1>;
} // namespace llaminar2
