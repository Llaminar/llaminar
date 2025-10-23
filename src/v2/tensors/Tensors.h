/**
 * @file Tensors.h
 * @brief Minimal tensor interface with device affinity
 *
 * Key design principles:
 * - Per-tensor device placement (not per-rank)
 * - Lazy host↔device synchronization
 * - Direct kernel creation (no operator layer)
 *
 * @author David Sanftenberg
 */

#pragma once

#include "TensorKernels.h"
#include <vector>
#include <memory>
#include <cstddef>

namespace llaminar2
{
    /**
     * @brief IQ4_NL block structure (exactly 18 bytes) representing 32 quantized elements.
     *
     * Layout mirrors GGML's block_iq4_nl. Two 4-bit indices per byte in @p qs select entries
     * in kvalues_iq4nl, scaled by FP16 value @p d.
     */
    struct IQ4_NLBlock
    {
        uint16_t d;     ///< FP16 scale factor
        uint8_t qs[16]; ///< Packed 4-bit indices (2 per byte)

        static constexpr size_t BLOCK_SIZE = 32; ///< Elements per block
    };

    static_assert(sizeof(IQ4_NLBlock) == 18, "IQ4_NLBlock must be 18 bytes");

    /**
     * @brief Tensor data type
     */
    enum class TensorType
    {
        FP32,   // 32-bit float
        BF16,   // 16-bit bfloat
        FP16,   // 16-bit float
        IQ4_NL, // 4-bit quantized (non-linear)
        Q6_K,   // 6-bit quantized
        Q8_0    // 8-bit quantized
    };

    /**
     * @brief Abstract tensor interface
     */
    class TensorBase
    {
    public:
        virtual ~TensorBase() = default;

        // Shape and type
        virtual const std::vector<size_t> &shape() const = 0;
        virtual TensorType native_type() const = 0;

        // Device affinity
        virtual int device_index() const = 0;        // -1 = host, ≥0 = device index from DeviceManager
        virtual bool set_device(int device_idx) = 0; // Upload to device
        virtual bool is_on_device(int device_idx) const = 0;

        // Data access (fallback for non-fused operations)
        virtual const float *data() const = 0; // Returns host pointer (syncs from device if needed)
        virtual float *mutable_data() = 0;     // Returns host pointer, marks dirty

        // Kernel creation (fused operations)
        virtual std::unique_ptr<ITensorGemm> createGemm() = 0;
        virtual std::unique_ptr<ITensorRoPE> createRoPE() = 0;
        virtual std::unique_ptr<ITensorSwiGLU> createSwiGLU() = 0;
        virtual std::unique_ptr<ITensorSoftmax> createSoftmax() = 0;
        virtual std::unique_ptr<ITensorRMSNorm> createRMSNorm() = 0;
    };

    /**
     * @brief FP32 tensor with optional device storage
     */
    class FP32Tensor : public TensorBase
    {
    public:
        explicit FP32Tensor(const std::vector<size_t> &shape);
        ~FP32Tensor() override;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::FP32; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

    private:
        std::vector<size_t> shape_;
        int device_idx_; // -1 = host, ≥0 = device index

        std::vector<float> host_data_; // Always allocated
        void *device_data_;            // Allocated if device_idx ≥ 0

        bool host_dirty_;   // Host modified, needs upload
        bool device_dirty_; // Device modified, needs download

        bool sync_to_device();
        bool sync_from_device();
    };

    // Forward declare for IBlockDecoder
    class IBlockDecoder;

    /**
     * @brief IQ4_NL quantized tensor (4.5 bpw, 7.1× compression)
     *
     * Implements non-linear 4-bit quantization with lookup table.
     * Provides both full decode and fused kernel interfaces.
     *
     * Also implements IBlockDecoder to enable generic QuantizedGemmKernel.
     */
    class IQ4_NLTensor : public TensorBase, public IBlockDecoder
    {
    public:
        IQ4_NLTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~IQ4_NLTensor() override;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ4_NL; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override; // Dequantizes to temp buffer
        float *mutable_data() override;     // Not supported for quantized tensors

        std::unique_ptr<ITensorGemm> createGemm() override; // Fused dequant+GEMM
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // Shape and metadata
        size_t size() const { return shape_[0] * shape_[1]; }
        size_t ndim() const { return 2; }
        float compression_ratio() const { return 7.1f; }
        size_t logical_k() const { return shape_[1]; }
        size_t padded_k() const;
        size_t element_count() const { return shape_[0] * shape_[1]; }

        // Raw data access
        const uint8_t *raw_data() const { return raw_data_.data(); }
        const uint8_t *raw_blocks() const { return raw_data_.data(); }
        size_t raw_size() const { return raw_data_.size(); }
        size_t num_blocks() const { return raw_data_.size() / sizeof(IQ4_NLBlock); }

        // Decode API
        void decode_to_fp32(float *dst) const;
        void decodeRow(size_t row_idx, float *buffer) const;
        void decodeSpan(size_t offset, size_t count, float *buffer) const;

        // Fused kernel helpers (non-virtual versions for backward compatibility)
        const IQ4_NLBlock &get_block_at(size_t row_idx, size_t k_block_offset) const;
        void decode_tile_blocks(size_t row_start, size_t tile_n, size_t k_block_offset, float *output) const;

        // IBlockDecoder interface (inline for zero overhead in GEMM hot path)
        __attribute__((always_inline)) void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
            const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(raw_data_.data());
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            decodeBlock(blocks[block_idx], output);
        }

        __attribute__((always_inline))
        const void *
        get_raw_block_at(size_t row_idx, size_t k_block_offset) const override
        {
            const size_t blocks_per_row = (shape_[1] + IQ4_NLBlock::BLOCK_SIZE - 1) / IQ4_NLBlock::BLOCK_SIZE;
            const IQ4_NLBlock *blocks = reinterpret_cast<const IQ4_NLBlock *>(raw_data_.data());
            const size_t block_idx = row_idx * blocks_per_row + k_block_offset;
            return &blocks[block_idx];
        }

        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return IQ4_NLBlock::BLOCK_SIZE; }

    private:
        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_; // Quantized blocks on host

        int device_idx_;
        void *device_blocks_; // Quantized blocks on device (if uploaded)

        mutable std::vector<float> dequant_cache_; // Temporary for data() calls

        // Decode helpers
        static void decodeBlock(const IQ4_NLBlock &block, float *output);
        static void decode_to_fp32_microkernel(float *dst, const IQ4_NLBlock *blocks, int rows, int cols, size_t blocks_per_row);

#if defined(__AVX512F__)
        static void decodeBlockAVX512(const IQ4_NLBlock &block, float *output);
        static inline void decodeBlockVectorizedAVX512(const IQ4_NLBlock &block, float *output);
#endif

#if defined(__AVX2__)
        static void decodeBlockAVX2(const IQ4_NLBlock &block, float *output);
        static inline void decodeBlockVectorizedAVX2(const IQ4_NLBlock &block, float *output);
#endif
    };

} // namespace llaminar2
