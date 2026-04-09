/**
 * @file CPURoPEKernelT.h
 * @brief Typed RoPE kernel with ActivationPrecision support
 * @author David Sanftenberg
 *
 * This kernel implements ITensorRoPE directly, eliminating the need for adapters.
 * Each precision specialization overrides the relevant apply_* method AND apply_tensor().
 *
 * Architecture:
 * - CPURoPEKernelT<FP32> implements apply() + apply_tensor()
 * - CPURoPEKernelT<BF16> implements apply_bf16() + apply_tensor()
 * - CPURoPEKernelT<FP16> implements apply_fp16() + apply_tensor()
 * - CPURoPEKernelT<Q8_1> implements apply_q8_1() + apply_tensor()
 *
 * KernelFactory can simply return the typed kernel as ITensorRoPE* - no adapters needed.
 *
 * Usage:
 *   auto kernel = std::make_unique<CPURoPEKernelT<ActivationPrecision::FP32>>();
 *   kernel->apply_tensor(Q, K, ...);  // Uses ITensorRoPE interface
 */

#pragma once

#include "../CPUKernelBase.h"
#include "../../../execution/config/RuntimeConfig.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../tensors/TensorKernels.h" // For ITensorRoPE
#include "../primitives/RoPEPrimitives.h"   // For RoPEPersistentState

#include <cstdint>

namespace llaminar2
{

    // Forward declarations
    class IMPIContext;

    // =========================================================================
    // Precision Metadata Traits (same pattern as CPURMSNormKernelT)
    // =========================================================================

    namespace detail
    {
        template <ActivationPrecision P>
        struct RoPEPrecisionMetadata;

        template <>
        struct RoPEPrecisionMetadata<ActivationPrecision::FP32>
        {
            using StorageType = float;
            static constexpr const char *name = "FP32";
            static constexpr float compression_ratio = 1.0f;
        };

        template <>
        struct RoPEPrecisionMetadata<ActivationPrecision::BF16>
        {
            using StorageType = uint16_t;
            static constexpr const char *name = "BF16";
            static constexpr float compression_ratio = 2.0f;
        };

        template <>
        struct RoPEPrecisionMetadata<ActivationPrecision::FP16>
        {
            using StorageType = uint16_t;
            static constexpr const char *name = "FP16";
            static constexpr float compression_ratio = 2.0f;
        };

        template <>
        struct RoPEPrecisionMetadata<ActivationPrecision::Q8_1>
        {
            using StorageType = Q8_1Block;
            static constexpr const char *name = "Q8_1";
            static constexpr float compression_ratio = 4.0f;
        };

        template <>
        struct RoPEPrecisionMetadata<ActivationPrecision::Q16_1>
        {
            using StorageType = Q16_1Block;
            static constexpr const char *name = "Q16_1";
            static constexpr float compression_ratio = 2.0f; // 72 bytes per 32 elements = 2.25 bytes/elem
        };
    } // namespace detail

    // =========================================================================
    // Primary Template (Base Declaration)
    // =========================================================================

    /**
     * @brief Typed RoPE kernel template
     *
     * Primary template provides type traits and common metadata.
     * Specializations implement precision-specific apply_typed().
     */
    template <ActivationPrecision Precision>
    class CPURoPEKernelT : public CPUKernelBase
    {
    public:
        using Metadata = detail::RoPEPrecisionMetadata<Precision>;
        using StorageType = typename Metadata::StorageType;

        virtual ~CPURoPEKernelT() = default;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1; // CPU only
        }

        static constexpr ActivationPrecision precision() { return Precision; }
        static const char *precision_name() { return Metadata::name; }
        static constexpr float compression_ratio() { return Metadata::compression_ratio; }

        /**
         * @brief Apply RoPE with typed storage
         *
         * @param Q Q tensor data (modified in-place)
         * @param K K tensor data (modified in-place), or nullptr
         * @param position_ids Position indices [seq_len], nullptr = sequential, -1 = padding
         * @param seq_len Sequence length
         * @param n_heads Number of query heads
         * @param n_kv_heads Number of key/value heads
         * @param head_dim Head dimension
         * @param rope_theta RoPE base frequency (e.g., 10000.0f)
         * @param device_idx Device index (-1 for CPU)
         * @return true on success
         */
        bool apply_typed(
            StorageType *Q,
            StorageType *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta = 10000.0f,
            int device_idx = -1,
            int rotary_dim = 0);
    };

    // =========================================================================
    // FP32 Specialization - implements ITensorRoPE directly
    // =========================================================================

    template <>
    class CPURoPEKernelT<ActivationPrecision::FP32> : public ITensorRoPE
    {
    public:
        using StorageType = float;

        CPURoPEKernelT() = default;
        ~CPURoPEKernelT() override;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1;
        }

        static constexpr ActivationPrecision precision() { return ActivationPrecision::FP32; }
        static const char *precision_name() { return "FP32"; }
        static constexpr float compression_ratio() { return 1.0f; }

        // IKernelSnapshotCapable interface
        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::rope()
                .withInput("Q", "query tensor [seq_len, n_heads, head_dim]", KernelBufferDtype::FP32)
                .withInput("K", "key tensor [seq_len, n_kv_heads, head_dim]", KernelBufferDtype::FP32)
                .withInput("position_ids", "position indices [seq_len]", KernelBufferDtype::INT32)
                .withOutput("Q_out", "rotated query [seq_len, n_heads, head_dim]", KernelBufferDtype::FP32)
                .withOutput("K_out", "rotated key [seq_len, n_kv_heads, head_dim]", KernelBufferDtype::FP32)
                .withScalar("rope_theta", "RoPE frequency base");
        }

        // Internal typed implementation
        bool apply_typed(
            float *Q,
            float *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta = 10000.0f,
            int device_idx = -1,
            int rotary_dim = 0);

        // Legacy FP32 apply - no longer part of ITensorRoPE interface
        bool apply(
            float *data, float *output,
            const int *pos_ids,
            int batch_size, int seq_len, int head_dim, int num_heads,
            float theta_base, bool interleaved,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1);

        // ITensorRoPE interface - apply_tensor() with automatic dispatch
        bool apply_tensor(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            int pos_offset = 0,
            int rotary_dim = 0) override;
    };

    // =========================================================================
    // BF16 Specialization - implements ITensorRoPE directly
    // =========================================================================

    template <>
    class CPURoPEKernelT<ActivationPrecision::BF16> : public ITensorRoPE
    {
    public:
        using StorageType = uint16_t;

        CPURoPEKernelT() = default;
        ~CPURoPEKernelT() override;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1;
        }

        static constexpr ActivationPrecision precision() { return ActivationPrecision::BF16; }
        static const char *precision_name() { return "BF16"; }
        static constexpr float compression_ratio() { return 2.0f; }

        // IKernelSnapshotCapable interface
        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::rope()
                .withInput("Q", "query tensor [seq_len, n_heads, head_dim]", KernelBufferDtype::BF16)
                .withInput("K", "key tensor [seq_len, n_kv_heads, head_dim]", KernelBufferDtype::BF16)
                .withInput("position_ids", "position indices [seq_len]", KernelBufferDtype::INT32)
                .withOutput("Q_out", "rotated query [seq_len, n_heads, head_dim]", KernelBufferDtype::BF16)
                .withOutput("K_out", "rotated key [seq_len, n_kv_heads, head_dim]", KernelBufferDtype::BF16)
                .withScalar("rope_theta", "RoPE frequency base");
        }

        // Internal typed implementation
        bool apply_typed(
            uint16_t *Q,
            uint16_t *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta = 10000.0f,
            int device_idx = -1,
            int rotary_dim = 0);

        // Legacy BF16 apply_bf16 - no longer part of ITensorRoPE interface
        bool apply_bf16(
            uint16_t *data, uint16_t *output,
            const int *pos_ids,
            int batch_size, int seq_len, int head_dim, int num_heads,
            float theta_base, int device_idx);

        // ITensorRoPE interface - apply_tensor() with automatic dispatch
        bool apply_tensor(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            int pos_offset = 0,
            int rotary_dim = 0) override;
    };

    // =========================================================================
    // FP16 Specialization - implements ITensorRoPE directly
    // =========================================================================

    template <>
    class CPURoPEKernelT<ActivationPrecision::FP16> : public ITensorRoPE
    {
    public:
        using StorageType = uint16_t;

        CPURoPEKernelT() = default;
        ~CPURoPEKernelT() override;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1;
        }

        static constexpr ActivationPrecision precision() { return ActivationPrecision::FP16; }
        static const char *precision_name() { return "FP16"; }
        static constexpr float compression_ratio() { return 2.0f; }

        // IKernelSnapshotCapable interface
        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::rope()
                .withInput("Q", "query tensor [seq_len, n_heads, head_dim]", KernelBufferDtype::FP16)
                .withInput("K", "key tensor [seq_len, n_kv_heads, head_dim]", KernelBufferDtype::FP16)
                .withInput("position_ids", "position indices [seq_len]", KernelBufferDtype::INT32)
                .withOutput("Q_out", "rotated query [seq_len, n_heads, head_dim]", KernelBufferDtype::FP16)
                .withOutput("K_out", "rotated key [seq_len, n_kv_heads, head_dim]", KernelBufferDtype::FP16)
                .withScalar("rope_theta", "RoPE frequency base");
        }

        // Internal typed implementation
        bool apply_typed(
            uint16_t *Q,
            uint16_t *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta = 10000.0f,
            int device_idx = -1,
            int rotary_dim = 0);

        // Legacy FP16 apply_fp16 - no longer part of ITensorRoPE interface
        bool apply_fp16(
            uint16_t *data, uint16_t *output,
            const int *pos_ids,
            int batch_size, int seq_len, int head_dim, int num_heads,
            float theta_base, int device_idx);

        // ITensorRoPE interface - apply_tensor() with automatic dispatch
        bool apply_tensor(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            int pos_offset = 0,
            int rotary_dim = 0) override;
    };

    // =========================================================================
    // Q8_1 Specialization (Pure-Integer) - implements ITensorRoPE directly
    // =========================================================================

    /**
     * @brief Q8_1 specialization - pure-integer RoPE without FP32 round-trips
     *
     * This specialization uses:
     * - Sin/cos quantized to Q15 fixed-point format
     * - Integer rotation arithmetic on Q8_1 blocks
     * - No intermediate FP32 dequantization
     *
     * Constraint: head_dim must be divisible by 32 (Q8_1 block size)
     */
    template <>
    class CPURoPEKernelT<ActivationPrecision::Q8_1> : public ITensorRoPE
    {
    public:
        using StorageType = Q8_1Block;

        CPURoPEKernelT() = default;
        ~CPURoPEKernelT() override;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1;
        }

        static constexpr ActivationPrecision precision() { return ActivationPrecision::Q8_1; }
        static const char *precision_name() { return "Q8_1"; }
        static constexpr float compression_ratio() { return 4.0f; }

        // IKernelSnapshotCapable interface
        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::rope()
                .withInput("Q", "query tensor [seq_len, n_heads, n_blocks_per_head]", KernelBufferDtype::Q8_1)
                .withInput("K", "key tensor [seq_len, n_kv_heads, n_blocks_per_head]", KernelBufferDtype::Q8_1)
                .withInput("position_ids", "position indices [seq_len]", KernelBufferDtype::INT32)
                .withOutput("Q_out", "rotated query [seq_len, n_heads, n_blocks_per_head]", KernelBufferDtype::Q8_1)
                .withOutput("K_out", "rotated key [seq_len, n_kv_heads, n_blocks_per_head]", KernelBufferDtype::Q8_1)
                .withScalar("rope_theta", "RoPE frequency base");
        }

        // Internal typed implementation
        bool apply_typed(
            Q8_1Block *Q,
            Q8_1Block *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta = 10000.0f,
            int device_idx = -1,
            int rotary_dim = 0);

        // Legacy Q8_1 apply_q8_1 - no longer part of ITensorRoPE interface
        bool apply_q8_1(
            void *data, void *output,
            const int *pos_ids,
            int batch_size, int seq_len, int head_dim, int num_heads,
            float theta_base, int device_idx);

        // ITensorRoPE interface - apply_tensor() with automatic dispatch
        bool apply_tensor(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            int pos_offset = 0,
            int rotary_dim = 0) override;

        // Hybrid mode: Q8_1 input → FP32 output (no requantization)
        bool apply_q8_1_to_fp32(
            TensorBase *Q_in,
            TensorBase *K_in,
            TensorBase *Q_out,
            TensorBase *K_out,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        // HybridQ16 mode: Q8_1 input → Q16_1 output (high-precision integer)
        bool apply_q8_1_to_q16_1(
            TensorBase *Q_in,
            TensorBase *K_in,
            TensorBase *Q_out,
            TensorBase *K_out,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        // HybridQ16 mode with head scales: Q8_1 → Q16 with per-head scale output
        // Uses pure integer arithmetic (Q8 fixed-point ratios, Q15 sin/cos)
        bool apply_q8_1_to_q16(
            TensorBase *Q_in,
            TensorBase *K_in,
            TensorBase *Q_out,
            TensorBase *K_out,
            float *Q_head_scales,
            float *K_head_scales,
            Q16BlockSize block_size,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        // HybridQ16 mode with FIXED scale: Q8_1 → Q16 with kv_cache_scale output
        // All output blocks have d = kv_cache_scale / 32767, enabling integer attention
        bool apply_q8_1_to_q16_fixed_scale(
            TensorBase *Q_in,
            TensorBase *K_in,
            TensorBase *Q_out,
            TensorBase *K_out,
            Q16BlockSize block_size,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            float kv_cache_scale,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        // HybridQ16 K precision fix: Q=Q8_1→Q16_1 (fixed), K=Q16_1→Q16_1 (dynamic)
        // Used when GEMM outputs K as Q16_1 to preserve small values.
        // Q is processed normally (Q8_1→Q16_1 fixed scale).
        // K uses dynamic per-head scale output for attention kernel.
        bool apply_mixed_q8_k16_to_q16(
            TensorBase *Q_in,     // Q8_1 input
            TensorBase *K_in,     // Q16_1 input (from GEMM K precision fix!)
            TensorBase *Q_out,    // Q16_1 output (fixed scale)
            TensorBase *K_out,    // Q16_1 output (dynamic scale)
            float *K_head_scales, // Output: per-head K scales [seq_len * n_kv_heads]
            Q16BlockSize block_size,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            float kv_cache_scale,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;
    };

    // =========================================================================
    // Q16_1 Specialization (High-Precision Integer) - implements ITensorRoPE
    // =========================================================================

    /**
     * @brief Q16_1 specialization - high-precision in-place RoPE
     *
     * Q16_1 has 256× finer precision than Q8_1 (±32767 vs ±127) with FP32 scale.
     * This enables in-place rotation with acceptable precision loss:
     * - Sin/cos quantized to Q15 fixed-point format (same as Q8_1)
     * - Dequant → Rotate → Requant using FP32 intermediate
     * - FP32 scale eliminates FP16 conversion errors
     *
     * Constraint: head_dim must be divisible by 32 (Q16_1 block size)
     */
    template <>
    class CPURoPEKernelT<ActivationPrecision::Q16_1> : public ITensorRoPE
    {
    public:
        using StorageType = Q16_1Block;

        CPURoPEKernelT() = default;
        ~CPURoPEKernelT() override = default;

        bool supports_device(int device_idx) const
        {
            return device_idx == -1;
        }

        static constexpr ActivationPrecision precision() { return ActivationPrecision::Q16_1; }
        static const char *precision_name() { return "Q16_1"; }
        static constexpr float compression_ratio() { return 2.0f; }

        // IKernelSnapshotCapable interface
        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::rope()
                .withInput("Q", "query tensor [seq_len, n_heads, n_blocks_per_head]", KernelBufferDtype::Q16_1)
                .withInput("K", "key tensor [seq_len, n_kv_heads, n_blocks_per_head]", KernelBufferDtype::Q16_1)
                .withInput("position_ids", "position indices [seq_len]", KernelBufferDtype::INT32)
                .withOutput("Q_out", "rotated query [seq_len, n_heads, n_blocks_per_head]", KernelBufferDtype::Q16_1)
                .withOutput("K_out", "rotated key [seq_len, n_kv_heads, n_blocks_per_head]", KernelBufferDtype::Q16_1)
                .withScalar("rope_theta", "RoPE frequency base");
        }

        // Internal typed implementation (32-element blocks for backward compatibility)
        bool apply_typed(
            Q16_1Block *Q,
            Q16_1Block *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta = 10000.0f,
            int device_idx = -1,
            int rotary_dim = 0);

        /**
         * @brief Templated apply for variable Q16 block sizes
         *
         * Calls the appropriate templated RoPE primitive for the given block type.
         * Supports: Q16_1Block (32), Q16_1Block_64, Q16_1Block_128
         *
         * @tparam BlockType Q16 block type
         */
        template <typename BlockType>
        bool apply_typed_block(
            BlockType *Q,
            BlockType *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta = 10000.0f,
            int device_idx = -1,
            int rotary_dim = 0);

        // Legacy Q16_1 apply_q16_1 - no longer part of ITensorRoPE interface
        bool apply_q16_1(
            void *Q_data, void *K_data,
            const int *pos_ids,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            float theta_base, int device_idx);

        // ITensorRoPE interface - apply_tensor() with automatic dispatch
        // Dispatches to correct block size based on Q16_1Tensor::q16_block_size()
        bool apply_tensor(
            TensorBase *Q,
            TensorBase *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            int pos_offset = 0,
            int rotary_dim = 0) override;

    private:
        primitives::RoPEPersistentState tls_state_;
    };

} // namespace llaminar2
