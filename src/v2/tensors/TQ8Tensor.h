/**
 * @file TQ8Tensor.h
 * @brief TurboQuant 8-bit tensor for KV cache K-projection storage
 *
 * Stores K-projection cache vectors quantized using 256-level Lloyd-Max
 * TurboQuant codes. Provides higher fidelity than TQ4 (SQNR 38.79 dB vs 20.2 dB)
 * at the cost of 2× storage per element.
 *
 * Used for the K cache in the asymmetric TQ8-K / TQ4-V KV cache strategy,
 * where K accuracy is critical for attention score computation.
 *
 * Each logical "row" of kv_dim elements is stored as n_kv_heads TQ8Blocks,
 * one block per attention head (each block = head_dim elements).
 *
 * Memory layout (POSITION_MAJOR):
 *   [position][n_kv_heads × block_bytes]
 *   where block_bytes = sizeof(TQ8Block<head_dim>)
 */

#pragma once

#include "TensorClasses.h"

#include <vector>
#include <memory>
#include <cstring>

namespace llaminar2
{

    // Forward declarations
    class TurboQuantContext;

    /**
     * @brief TurboQuant 8-bit tensor for KV cache K-projection storage.
     *
     * Uses raw byte storage with runtime head_dim to interpret block layout.
     * typed_data() returns uint8_t* for generic block-level memcpy in ring buffer.
     *
     * @see TQ8Block for the per-head block structure.
     * @see TurboQuantQuantizeTQ8.h for quantization.
     * @see TurboQuantDequantizeTQ8.h for dequantization.
     */
    class TQ8Tensor : public TypedTensorBase<TQ8Tensor, uint8_t>,
                      public TensorBase
    {
    public:
        using value_type = uint8_t;
        static constexpr int static_type_id() { return TensorTypeId::TQ8; }

        // ===== CRTP Implementation for TypedTensorBase =====
        const uint8_t *data_impl() const { return raw_blocks_.data(); }
        uint8_t *mutable_data_impl() { return raw_blocks_.data(); }

        // =====================================================================
        // Constructors
        // =====================================================================

        /**
         * @brief Construct empty TQ8 tensor for KV cache pre-allocation.
         *
         * @param shape   Tensor shape: [num_positions, kv_dim] where kv_dim = n_kv_heads * head_dim
         * @param head_dim Head dimension (64 or 128). Determines block size.
         * @param device  Device placement (default: CPU).
         */
        TQ8Tensor(const std::vector<size_t> &shape, int head_dim, DeviceId device = DeviceId::cpu());

        ~TQ8Tensor() override = default;

        // =====================================================================
        // TensorBase interface
        // =====================================================================

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::TQ8; }
        DeviceId home_device() const override { return device_; }

        /// Returns FP32 dequantized data (lazily cached).
        /// Requires a TurboQuant context to be set via set_turboquant_context().
        const float *data() const override;
        const float *fp32_data() const override { return data(); }
        float *mutable_data() override;

        // ===== Diamond Inheritance Resolution =====
        int native_type_id() const final { return TensorBase::native_type_id(); }
        size_t size_bytes() const final { return TensorBase::size_bytes(); }
        const void *raw_data() const final { return raw_blocks_.data(); }
        void *raw_mutable_data() final { return raw_blocks_.data(); }

    protected:
        size_t byte_size() const override { return raw_blocks_.size(); }
        void *raw_host_data_ptr() override { return raw_blocks_.data(); }
        const void *raw_host_data_ptr() const override { return raw_blocks_.data(); }

    public:
        bool copyFrom(const TensorBase *src) override;

        void release_raw_data() override;
        bool is_raw_data_released() const override { return raw_data_released_; }

        void to_fp32(float *dst) const override;
        void to_bf16(uint16_t *) const override { throw std::runtime_error("TQ8Tensor: to_bf16 not supported"); }
        void to_fp16(uint16_t *) const override { throw std::runtime_error("TQ8Tensor: to_fp16 not supported"); }
        void to_int8_blocked(int8_t *, float *, size_t) const override { throw std::runtime_error("TQ8Tensor: to_int8_blocked not supported"); }
        bool to_int8_perchannel(int8_t *, float *, float *) const override { return false; }
        void to_fp32_row(size_t row_idx, float *buffer) const override;
        void to_fp32_span(size_t offset, size_t count, float *buffer) const override;

        std::unique_ptr<ITensorGemm> createGemm() override { return nullptr; }

        std::shared_ptr<TensorBase> create_view(const std::vector<size_t> &, size_t) override
        {
            throw std::runtime_error("TQ8Tensor: create_view not supported");
        }

        // =====================================================================
        // TQ8-specific interface
        // =====================================================================

        /// Head dimension (determines block size)
        int head_dim() const { return head_dim_; }

        /// Bytes per TQ8 block for this head_dim
        size_t block_bytes() const { return block_bytes_; }

        /// Number of blocks per row (= n_kv_heads = kv_dim / head_dim)
        size_t blocks_per_row() const { return blocks_per_row_; }

        /// Total number of blocks in the tensor
        size_t total_blocks() const { return rows() * blocks_per_row_; }

        void set_turboquant_context(const TurboQuantContext *turboquant_ctx) { turboquant_ctx_ = turboquant_ctx; }

        const TurboQuantContext *turboquant_context() const { return turboquant_ctx_; }

        /**
         * @brief Quantize FP32 data into a new TQ8Tensor (scalar-full mode).
         *
         * @param src       Source FP32 data [num_positions, kv_dim]
         * @param shape     Shape: [num_positions, kv_dim]
         * @param head_dim  Head dimension (64 or 128)
         * @param turboquant_ctx  Pre-computed TurboQuant context
         * @return New TQ8Tensor with quantized data
         */
        static std::shared_ptr<TQ8Tensor> quantize_from_fp32(
            const float *src,
            const std::vector<size_t> &shape,
            int head_dim,
            const TurboQuantContext &turboquant_ctx);

        /**
         * @brief Quantize first num_rows rows of FP32 data into existing storage.
         *
         * @param src_data  Source FP32 data [num_rows, kv_dim]
         * @param num_rows  Number of rows to quantize
         * @param turboquant_ctx  Pre-computed TurboQuant context
         * @return true on success
         */
        bool copyFrom_fp32_rows(const float *src_data, size_t num_rows, const TurboQuantContext &turboquant_ctx);

        /**
         * @brief Dequantize to FP32.
         *
         * @param dst       Output buffer [rows, kv_dim]
         * @param turboquant_ctx  TurboQuant context for dequantization
         */
        void dequantize_to_fp32(float *dst, const TurboQuantContext &turboquant_ctx) const;

    private:
        std::vector<size_t> shape_; ///< [num_positions, kv_dim]
        int head_dim_;              ///< Head dimension (64 or 128)
        size_t block_bytes_;
        size_t blocks_per_row_; ///< kv_dim / head_dim
        DeviceId device_;
        std::vector<uint8_t> raw_blocks_; ///< Raw block data
        bool raw_data_released_ = false;

        mutable std::vector<float> dequant_cache_; ///< Lazy FP32 dequantization cache
        mutable bool dequant_cache_valid_ = false;

        const TurboQuantContext *turboquant_ctx_ = nullptr;

        void invalidate_dequant_cache() { dequant_cache_valid_ = false; }
    };

} // namespace llaminar2
