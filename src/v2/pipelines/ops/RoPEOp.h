/**
 * @file RoPEOp.h
 * @brief Typed Rotary Position Embedding operation
 * @author David Sanftenberg
 *
 * RoPE applies rotary position embeddings to Q and K tensors.
 * This is applied after Q/K projections, before attention computation.
 *
 * Provides precision-aware implementations for FP32, BF16, FP16, and Q8_1
 * activation precisions.
 */

#pragma once

#include "Op.h"
#include "../RuntimeConfig.h"
#include "../../tensors/Tensors.h"
#include "../../utils/DebugEnv.h"
#include "../../utils/Logger.h"
#include "../../utils/MPIContext.h"

#include <memory>
#include <cstring>

namespace llaminar2
{

    // ============================================================================
    // IRoPEOp Interface
    // ============================================================================

    /**
     * @brief Abstract interface for RoPE operations
     *
     * RoPE applies rotary embeddings based on position IDs:
     * - Q: [seq_len, n_heads * head_dim]
     * - K: [seq_len, n_kv_heads * head_dim]
     *
     * Position IDs of -1 signal padding tokens (RoPE is skipped for those positions).
     */
    class IRoPEOp
    {
    public:
        virtual ~IRoPEOp() = default;

        /**
         * @brief Apply RoPE to Q and K tensors
         *
         * @param Q Query tensor [seq_len, n_heads * head_dim] - modified in-place
         * @param K Key tensor [seq_len, n_kv_heads * head_dim] - modified in-place
         * @param position_ids Position indices [seq_len] (-1 = padding, skip RoPE)
         * @param seq_len Sequence length (number of tokens)
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads (for GQA)
         * @param head_dim Dimension per head
         * @param rope_theta RoPE frequency base (10000.0 for LLaMA, 1000000.0 for Qwen2.5)
         * @param mpi_ctx MPI context (nullptr for single-node)
         * @param device_idx Device index (-1 for CPU)
         *
         * @return true on success, false on validation or execution failure
         */
        virtual bool apply(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const MPIContext *mpi_ctx,
            int device_idx) = 0;

        /**
         * @brief Get the activation precision this op handles
         */
        virtual ActivationPrecision precision() const = 0;
    };

    // ============================================================================
    // RoPEOpTyped<P> Template Implementation
    // ============================================================================

    namespace detail
    {

        /**
         * @brief Map ActivationPrecision to tensor type for RoPE ops
         */
        template <ActivationPrecision P>
        struct RoPETensor;

        template <>
        struct RoPETensor<ActivationPrecision::FP32>
        {
            using type = FP32Tensor;
        };

        template <>
        struct RoPETensor<ActivationPrecision::BF16>
        {
            using type = BF16Tensor;
        };

        template <>
        struct RoPETensor<ActivationPrecision::FP16>
        {
            using type = FP16Tensor;
        };

        template <>
        struct RoPETensor<ActivationPrecision::Q8_1>
        {
            using type = Q8_1Tensor;
        };

    } // namespace detail

    /**
     * @brief Typed RoPE operation implementation
     *
     * Template specializations delegate to the appropriate kernel
     * via apply_tensor() which handles type dispatch internally.
     *
     * @note Q8_1 RoPE uses native integer math (no FP32 round-trip).
     */
    template <ActivationPrecision P>
    class RoPEOpTyped : public IRoPEOp, public OpBase
    {
    public:
        using TensorT = typename detail::RoPETensor<P>::type;

        RoPEOpTyped() = default;

        const char *name() const override { return "RoPEOpTyped"; }
        ActivationPrecision precision() const override { return P; }

        /**
         * @brief Get the expected TensorType for this precision
         */
        static constexpr TensorType expected_tensor_type()
        {
            if constexpr (P == ActivationPrecision::FP32)
                return TensorType::FP32;
            else if constexpr (P == ActivationPrecision::BF16)
                return TensorType::BF16;
            else if constexpr (P == ActivationPrecision::FP16)
                return TensorType::FP16;
            else if constexpr (P == ActivationPrecision::Q8_1)
                return TensorType::Q8_1;
            else
                return TensorType::FP32;
        }

        bool apply(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const MPIContext *mpi_ctx,
            int device_idx) override
        {
            // Compile-time validation
            static_assert(sizeof(TensorT) > 0,
                          "RoPETensor trait must be defined for this ActivationPrecision");

            // Validate inputs
            if (!Q || !K)
            {
                LOG_ERROR("RoPEOpTyped<" << static_cast<int>(P) << ">: null Q or K tensor");
                return false;
            }
            if (!position_ids)
            {
                LOG_ERROR("RoPEOpTyped<" << static_cast<int>(P) << ">: null position_ids");
                return false;
            }
            if (seq_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0)
            {
                LOG_ERROR("RoPEOpTyped<" << static_cast<int>(P) << ">: invalid dimensions (seq_len="
                                         << seq_len << ", n_heads=" << n_heads << ", n_kv_heads=" << n_kv_heads
                                         << ", head_dim=" << head_dim << ")");
                return false;
            }

            // Type validation: ensure Q and K match expected precision
            constexpr TensorType expected = expected_tensor_type();
            if (Q->native_type() != expected)
            {
                LOG_ERROR("RoPEOpTyped<" << static_cast<int>(P) << ">: Q tensor type mismatch (expected "
                                         << static_cast<int>(expected) << ", got " << static_cast<int>(Q->native_type()) << ")");
                return false;
            }
            if (K->native_type() != expected)
            {
                LOG_ERROR("RoPEOpTyped<" << static_cast<int>(P) << ">: K tensor type mismatch (expected "
                                         << static_cast<int>(expected) << ", got " << static_cast<int>(K->native_type()) << ")");
                return false;
            }

            // Get kernel from Q tensor (IActivationTensor interface)
            auto *activation_q = dynamic_cast<IActivationTensor *>(Q);
            if (!activation_q)
            {
                LOG_ERROR("RoPEOpTyped<" << static_cast<int>(P) << ">: Q tensor must implement IActivationTensor");
                return false;
            }

            auto kernel = activation_q->createRoPE();
            if (!kernel)
            {
                LOG_ERROR("RoPEOpTyped<" << static_cast<int>(P) << ">: failed to create RoPE kernel");
                return false;
            }

            // Delegate to kernel's apply_tensor - handles all type dispatch internally
            if (!kernel->apply_tensor(Q, K, position_ids, seq_len, n_heads, n_kv_heads, head_dim, rope_theta, mpi_ctx, device_idx))
            {
                LOG_ERROR("RoPEOpTyped<" << static_cast<int>(P) << ">: kernel execution failed");
                return false;
            }

            return true;
        }
    };

    // ============================================================================
    // Factory Function
    // ============================================================================

    /**
     * @brief Create a typed RoPE op for the given activation precision
     * @param precision The activation precision to use
     * @return Unique pointer to IRoPEOp implementation
     */
    inline std::unique_ptr<IRoPEOp> createRoPEOp(ActivationPrecision precision)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<RoPEOpTyped<ActivationPrecision::FP32>>();
        case ActivationPrecision::BF16:
            return std::make_unique<RoPEOpTyped<ActivationPrecision::BF16>>();
        case ActivationPrecision::FP16:
            return std::make_unique<RoPEOpTyped<ActivationPrecision::FP16>>();
        case ActivationPrecision::Q8_1:
            return std::make_unique<RoPEOpTyped<ActivationPrecision::Q8_1>>();
        default:
            LOG_ERROR("createRoPEOp: unsupported precision " << static_cast<int>(precision));
            return nullptr;
        }
    }

} // namespace llaminar2
