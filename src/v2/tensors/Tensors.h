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
#include "FP16Utils.h"
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

    /** @brief Q8_0 block: 8-bit quantization (32 elements per block, 34 bytes) */
    struct Q8_0Block
    {
        uint16_t d;    ///< FP16 scale factor
        int8_t qs[32]; ///< 32 quantized int8 values
        static constexpr size_t BLOCK_SIZE = 32;
    };
    static_assert(sizeof(Q8_0Block) == 34, "Q8_0Block must be 34 bytes");

    /** @brief Q4_0 block: 4-bit quantization (32 elements per block, 18 bytes) */
    struct Q4_0Block
    {
        uint16_t d;     ///< FP16 scale factor
        uint8_t qs[16]; ///< 32 4-bit values packed (2 per byte)
        static constexpr size_t BLOCK_SIZE = 32;
    };
    static_assert(sizeof(Q4_0Block) == 18, "Q4_0Block must be 18 bytes");

    /** @brief Q4_1 block: 4-bit quantization with min (32 elements per block, 20 bytes) */
    struct Q4_1Block
    {
        uint16_t d;     ///< FP16 scale factor
        uint16_t m;     ///< FP16 minimum value
        uint8_t qs[16]; ///< 32 4-bit values packed (2 per byte)
        static constexpr size_t BLOCK_SIZE = 32;
    };
    static_assert(sizeof(Q4_1Block) == 20, "Q4_1Block must be 20 bytes");

    /** @brief Q6_K block: 6-bit K-quant (256 elements per super-block, 210 bytes) */
    struct Q6_KBlock
    {
        uint8_t ql[128];   ///< Lower 4 bits of 6-bit values
        uint8_t qh[64];    ///< Upper 2 bits of 6-bit values (packed)
        int8_t scales[16]; ///< Per-block scales
        uint16_t d;        ///< FP16 super-block scale
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(Q6_KBlock) == 210, "Q6_KBlock must be 210 bytes");

    /** @brief Q2_K block: 2-bit K-quant (256 elements per super-block, 84 bytes) */
    struct Q2_KBlock
    {
        uint8_t scales[16]; ///< Scales and mins (packed)
        uint8_t qs[64];     ///< 2-bit values packed (4 per byte)
        uint16_t d;         ///< FP16 super-block scale for scales
        uint16_t dmin;      ///< FP16 super-block scale for mins
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(Q2_KBlock) == 84, "Q2_KBlock must be 84 bytes");

    /** @brief Q5_K block: 5-bit K-quant (256 elements per super-block, 176 bytes) */
    struct Q5_KBlock
    {
        uint16_t d;         ///< FP16 super-block scale
        uint16_t dmin;      ///< FP16 super-block min scale
        uint8_t scales[12]; ///< 8 6-bit scales packed
        uint8_t qh[32];     ///< High bits (5th bit of 5-bit values)
        uint8_t qs[128];    ///< Lower 4 bits of 5-bit values
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(Q5_KBlock) == 176, "Q5_KBlock must be 176 bytes");

    /** @brief Q3_K block: 3-bit K-quant (256 elements per super-block, 110 bytes) */
    struct Q3_KBlock
    {
        uint8_t hmask[32];  ///< High bit masks (1 bit per element)
        uint8_t qs[64];     ///< Lower 2 bits of 3-bit values
        uint8_t scales[12]; ///< 16 scales packed (6 bits each)
        uint16_t d;         ///< FP16 super-block scale
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(Q3_KBlock) == 110, "Q3_KBlock must be 110 bytes");

    /** @brief Q4_K block: 4-bit K-quant (256 elements per super-block, 144 bytes) */
    struct Q4_KBlock
    {
        uint16_t d;         ///< FP16 super-block scale
        uint16_t dmin;      ///< FP16 super-block min scale
        uint8_t scales[12]; ///< 12 6-bit scales packed
        uint8_t qs[128];    ///< Lower 4 bits of 4-bit values
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(Q4_KBlock) == 144, "Q4_KBlock must be 144 bytes");

    /** @brief Q8_K block: 8-bit K-quant (256 elements per super-block, 288 bytes) */
    /** @brief Q8_K block: 8-bit K-quant super-block (256 elements, 288 bytes) */
    struct Q8_KBlock
    {
        int8_t qs[256];    ///< 8-bit quantized values
        int16_t bsums[16]; ///< Block sums for fast dot products
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(Q8_KBlock) == 288, "Q8_KBlock must be 288 bytes");

    /** @brief IQ4_XS block: 4-bit extra-small IQ (32 elements per block, 18 bytes) */
    struct IQ4_XSBlock
    {
        uint16_t d;     ///< FP16 scale factor
        uint8_t qs[16]; ///< 32 4-bit grid indices packed
        static constexpr size_t BLOCK_SIZE = 32;
    };
    static_assert(sizeof(IQ4_XSBlock) == 18, "IQ4_XSBlock must be 18 bytes");

    /** @brief IQ2_XXS block: 2-bit extra-extra-small IQ (256 elements per super-block, 66 bytes) */
    struct IQ2_XXSBlock
    {
        uint16_t d;      ///< FP16 scale factor
        uint16_t qs[32]; ///< 256 2-bit grid indices packed (16 per uint16)
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(IQ2_XXSBlock) == 66, "IQ2_XXSBlock must be 66 bytes");

    /** @brief IQ2_XS block: 2-bit extra-small IQ (256 elements per super-block, 74 bytes) */
    struct IQ2_XSBlock
    {
        uint16_t d;        ///< FP16 scale factor
        uint16_t qs[32];   ///< Grid indices
        uint8_t scales[8]; ///< Per-block scales
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(IQ2_XSBlock) == 74, "IQ2_XSBlock must be 74 bytes");

    /** @brief IQ3_XXS block: 3-bit extra-extra-small IQ (256 elements per super-block, 68 bytes) */
    struct IQ3_XXSBlock
    {
        uint16_t d;        ///< FP16 scale factor
        uint8_t qs[64];    ///< Grid indices (3 bits each, packed)
        uint8_t scales[2]; ///< Auxiliary scales
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ3_XXSBlock) == 68, "IQ3_XXSBlock must be 68 bytes");

    /** @brief IQ2_S block: 2-bit small IQ (256 elements per super-block) */
    struct IQ2_SBlock
    {
        uint16_t d;      ///< FP16 scale factor
        uint16_t qh;     ///< High bits
        uint16_t qs[32]; ///< Grid indices
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ2_SBlock) == 68, "IQ2_SBlock must be 68 bytes");

    /** @brief IQ3_S block: 3-bit small IQ (256 elements per super-block) */
    struct IQ3_SBlock
    {
        uint16_t d;        ///< FP16 scale factor
        uint8_t qs[96];    ///< Grid indices
        uint8_t scales[4]; ///< Per-block scales
        uint8_t signs[8];  ///< Sign patterns
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ3_SBlock) == 110, "IQ3_SBlock must be 110 bytes");

    /** @brief IQ1_S block: 1-bit small IQ (256 elements per super-block) */
    struct IQ1_SBlock
    {
        uint16_t d;    ///< FP16 scale factor
        uint8_t qs[8]; ///< Grid indices
        uint8_t qh[8]; ///< High bits and scales
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ1_SBlock) == 18, "IQ1_SBlock must be 18 bytes");

    /** @brief IQ1_M block: 1-bit medium IQ (256 elements per super-block, 32 bytes) */
    struct IQ1_MBlock
    {
        uint8_t qs[16];    ///< Grid indices
        uint8_t qh[8];     ///< High bits
        uint8_t scales[8]; ///< Per-block scales
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ1_MBlock) == 32, "IQ1_MBlock must be 32 bytes");

    /**
     * @brief Tensor data type
     */
    enum class TensorType
    {
        FP32,    // 32-bit float
        BF16,    // 16-bit bfloat
        FP16,    // 16-bit float
        IQ4_NL,  // 4-bit quantized (non-linear)
        IQ4_XS,  // 4-bit quantized (extra-small IQ)
        Q8_0,    // 8-bit quantized
        Q4_0,    // 4-bit quantized
        Q4_1,    // 4-bit quantized with min
        Q6_K,    // 6-bit K-quant
        Q2_K,    // 2-bit K-quant
        Q5_K,    // 5-bit K-quant
        Q3_K,    // 3-bit K-quant
        Q4_K,    // 4-bit K-quant
        Q8_K,    // 8-bit K-quant
        IQ2_XXS, // 2-bit extra-extra-small IQ
        IQ2_XS,  // 2-bit extra-small IQ
        IQ3_XXS, // 3-bit extra-extra-small IQ
        IQ2_S,   // 2-bit small IQ
        IQ3_S,   // 3-bit small IQ
        IQ1_S,   // 1-bit small IQ
        IQ1_M    // 1-bit medium IQ
    };

    /**
     * @brief Abstract tensor interface
     */
    class TensorBase : public std::enable_shared_from_this<TensorBase>
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

        // Device transfers (Phase 4.2)
        virtual bool copyFrom(const TensorBase *src) = 0; // Copy data from another tensor (handles device transfers)

        // Kernel creation (fused operations)
        virtual std::unique_ptr<ITensorGemm> createGemm() = 0;
        virtual std::unique_ptr<ITensorRoPE> createRoPE() = 0;
        virtual std::unique_ptr<ITensorSwiGLU> createSwiGLU() = 0;
        virtual std::unique_ptr<ITensorSoftmax> createSoftmax() = 0;
        virtual std::unique_ptr<ITensorRMSNorm> createRMSNorm() = 0;

        // ===== Tensor View Support =====

        /**
         * @brief Check if this tensor is a view of another tensor
         * @return true if this is a view, false if this owns its data
         */
        virtual bool is_view() const { return false; }

        /**
         * @brief Create a view into this tensor with a different shape
         *
         * Views share the underlying data buffer but present a different logical shape.
         * Useful for:
         * - Slicing tensors (e.g., first N tokens from pre-allocated buffer)
         * - Reshaping without copying
         * - Batch dimension manipulation
         *
         * @param new_shape The shape for the view
         * @param offset Offset in elements from the start of this tensor's data
         * @return Shared pointer to a view tensor, or nullptr if invalid
         *
         * @note The view borrows data from the parent tensor. The parent must outlive the view.
         * @note Total elements in new_shape must not exceed available elements from offset.
         */
        virtual std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) = 0;
    };

    // Implementation: FP32Tensor.cpp
    /**
     * @brief FP32 tensor with optional device storage
     */
    class FP32Tensor : public TensorBase
    {
    public:
        explicit FP32Tensor(const std::vector<size_t> &shape, int device_idx = -1);
        ~FP32Tensor() override;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::FP32; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Device-aware copy

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support
        bool is_view() const override { return is_view_; }
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override;

    private:
        // Private constructor for creating views
        FP32Tensor(const std::vector<size_t> &shape,
                   int device_idx,
                   std::vector<float> *parent_data,
                   size_t data_offset,
                   std::shared_ptr<FP32Tensor> parent);
        std::vector<size_t> shape_;
        int device_idx_; // -1 = host, ≥0 = device index

        // Ownership model:
        // - If is_view_ == false: owns host_data_
        // - If is_view_ == true: parent_data_ptr_ points to parent's host_data_
        bool is_view_;
        std::vector<float> host_data_;        // Owned data (only used when !is_view_)
        std::vector<float> *parent_data_ptr_; // Borrowed data pointer (only used when is_view_)
        size_t view_offset_;                  // Offset into parent data (only used when is_view_)
        std::shared_ptr<FP32Tensor> parent_;  // Keep parent alive (only used when is_view_)
                                              // Always allocated
        void *device_data_;                   // Allocated if device_idx ≥ 0

        bool host_dirty_;   // Host modified, needs upload
        bool device_dirty_; // Device modified, needs download

        bool sync_to_device();
        bool sync_from_device();
    };

    // Implementation: FP16Tensor.cpp
    /**
     * @brief FP16 tensor with optional device storage
     *
     * IEEE 754 half-precision (16-bit) floating point.
     * - 1 sign bit, 5 exponent bits, 10 mantissa bits
     * - Range: ±65,504 (narrower than FP32)
     * - Precision: ~3-4 decimal digits
     * - 2× memory reduction vs FP32
     */
    class FP16Tensor : public TensorBase
    {
    public:
        explicit FP16Tensor(const std::vector<size_t> &shape);
        FP16Tensor(const std::vector<size_t> &shape, const std::vector<uint16_t> &fp16_data);
        ~FP16Tensor() override;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::FP16; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override; // Dequantizes to cache
        float *mutable_data() override;     // Not supported

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

        // FP16-specific interface
        const uint16_t *fp16_data() const { return host_fp16_data_.data(); }
        void from_fp32(const float *fp32_data, size_t count);
        void to_fp32(float *fp32_data, size_t count) const;

    private:
        std::vector<size_t> shape_;
        int device_idx_;

        std::vector<uint16_t> host_fp16_data_;     // Native FP16 storage
        void *device_data_;                        // Device-side storage
        mutable std::vector<float> dequant_cache_; // For data() calls

        bool sync_to_device();
        bool sync_from_device();
    };

    // Implementation: BF16Tensor.cpp
    /**
     * @brief BF16 tensor with optional device storage
     *
     * Brain Float 16 (Google's reduced-precision format).
     * - 1 sign bit, 8 exponent bits, 7 mantissa bits
     * - Same range as FP32 (prevents overflow/underflow)
     * - Reduced precision (~2-3 decimal digits)
     * - 2× memory reduction vs FP32
     * - Hardware acceleration on Ice Lake+, Zen 4+
     */
    class BF16Tensor : public TensorBase
    {
    public:
        explicit BF16Tensor(const std::vector<size_t> &shape);
        BF16Tensor(const std::vector<size_t> &shape, const std::vector<uint16_t> &bf16_data);
        ~BF16Tensor() override;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::BF16; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override; // Dequantizes to cache
        float *mutable_data() override;     // Not supported

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

        // BF16-specific interface
        const uint16_t *bf16_data() const { return host_bf16_data_.data(); }
        void from_fp32(const float *fp32_data, size_t count);
        void to_fp32(float *fp32_data, size_t count) const;

    private:
        std::vector<size_t> shape_;
        int device_idx_;

        std::vector<uint16_t> host_bf16_data_;     // Native BF16 storage
        void *device_data_;                        // Device-side storage
        mutable std::vector<float> dequant_cache_; // For data() calls

        bool sync_to_device();
        bool sync_from_device();
    };

    // Forward declare for IBlockDecoder
    class IBlockDecoder;

    // Implementation: IQ4_NLTensor.cpp
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

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override; // Fused dequant+GEMM
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

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
        void decode_to_bf16(uint16_t *dst) const;
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
        static void decodeBlockVectorizedAVX2(const IQ4_NLBlock &block, float *output);
#endif
    };

    // ===== Q8_0 Tensor (8-bit quantization) =====

    // Implementation: Q8_0Tensor.cpp
    /**
     * @brief Q8_0 quantized tensor (8-bit uniform quantization)
     *
     * Block format: 32 elements per block, FP16 scale + int8[32] values
     * Compression: 4× vs FP32
     */
    class Q8_0Tensor : public TensorBase, public IBlockDecoder
    {
    public:
        Q8_0Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q8_0Tensor() override;

        // TensorBase interface
        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q8_0; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

        // IBlockDecoder interface
        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override;
        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override;
        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q8_0Block::BLOCK_SIZE; }

    private:
        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_;
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        static void decodeBlock(const Q8_0Block &block, float *output);

#if defined(__AVX512F__)
        static void decodeBlockAVX512(const Q8_0Block &block, float *output);
#endif

#if defined(__AVX2__)
        static void decodeBlockAVX2(const Q8_0Block &block, float *output);
#endif
    };

    // ===== Q4_0 Tensor (4-bit quantization) =====

    // Implementation: Q4_0Tensor.cpp
    /**
     * @brief Q4_0 quantized tensor (4-bit uniform quantization)
     *
     * Block format: 32 elements per block, FP16 scale + 4-bit packed values
     * Compression: 8× vs FP32
     */
    class Q4_0Tensor : public TensorBase, public IBlockDecoder
    {
    public:
        Q4_0Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q4_0Tensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q4_0; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override;
        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override;
        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q4_0Block::BLOCK_SIZE; }

    private:
        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_;
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        static void decodeBlock(const Q4_0Block &block, float *output);

#if defined(__AVX2__)
        static void decodeBlockAVX2(const Q4_0Block &block, float *output);
#endif
    };

    // ===== Q4_1 Tensor (4-bit with min) =====

    // Implementation: Q4_1Tensor.cpp
    /**
     * @brief Q4_1 quantized tensor (4-bit with min offset)
     *
     * Block format: 32 elements per block, FP16 scale + FP16 min + 4-bit packed values
     * Compression: ~7.1× vs FP32
     */
    class Q4_1Tensor : public TensorBase, public IBlockDecoder
    {
    public:
        Q4_1Tensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q4_1Tensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q4_1; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override;
        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override;
        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q4_1Block::BLOCK_SIZE; }

    private:
        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_;
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        static void decodeBlock(const Q4_1Block &block, float *output);

#if defined(__AVX2__)
        static void decodeBlockAVX2(const Q4_1Block &block, float *output);
#endif
    };

    // ===== K-quant Tensors =====

    // Implementation: Q6_KTensor.cpp
    /**
     * @brief Q6_K tensor (6-bit K-quant super-block)
     */
    class Q6_KTensor : public TensorBase, public IBlockDecoder
    {
    public:
        Q6_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q6_KTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q6_K; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override;
        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override;
        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q6_KBlock::BLOCK_SIZE; }

    private:
        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_;
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        static void decodeBlock(const Q6_KBlock &block, float *output);

#if defined(__AVX2__)
        static void decodeBlockAVX2(const Q6_KBlock &block, float *output);
#endif
    };

    // Implementation: Q2_KTensor.cpp
    /**
     * @brief Q2_K tensor (2-bit K-quant super-block)
     */
    class Q2_KTensor : public TensorBase, public IBlockDecoder
    {
    public:
        Q2_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q2_KTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q2_K; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override;
        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override;
        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q2_KBlock::BLOCK_SIZE; }

    private:
        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_;
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        static void decodeBlock(const Q2_KBlock &block, float *output);
    };

    // Implementation: Q5_KTensor.cpp
    /**
     * @brief Q5_K tensor (5-bit K-quant super-block)
     */
    class Q5_KTensor : public TensorBase, public IBlockDecoder
    {
    public:
        Q5_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q5_KTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q5_K; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override;
        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override;
        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q5_KBlock::BLOCK_SIZE; }

    private:
        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_;
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        static void decodeBlock(const Q5_KBlock &block, float *output);
    };

    // Implementation: Q3_KTensor.cpp
    /**
     * @brief Q3_K tensor (3-bit K-quant super-block)
     */
    class Q3_KTensor : public TensorBase, public IBlockDecoder
    {
    public:
        Q3_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q3_KTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q3_K; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override;
        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override;
        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q3_KBlock::BLOCK_SIZE; }

    private:
        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_;
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        static void decodeBlock(const Q3_KBlock &block, float *output);
    };

    // Implementation: Q4_KTensor.cpp
    /**
     * @brief Q4_K tensor (4-bit K-quant super-block)
     */
    class Q4_KTensor : public TensorBase, public IBlockDecoder
    {
    public:
        Q4_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q4_KTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q4_K; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override;
        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override;
        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q4_KBlock::BLOCK_SIZE; }

    private:
        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_;
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        static void decodeBlock(const Q4_KBlock &block, float *output);
    };

    // Implementation: Q8_KTensor.cpp
    /**
     * @brief Q8_K tensor (8-bit K-quant super-block)
     */
    class Q8_KTensor : public TensorBase, public IBlockDecoder
    {
    public:
        Q8_KTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~Q8_KTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::Q8_K; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override;
        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override;
        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return Q8_KBlock::BLOCK_SIZE; }

    private:
        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_;
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        static void decodeBlock(const Q8_KBlock &block, float *output);
    };

    // ===== IQ Tensors =====

    // Implementation: IQ4_XSTensor.cpp
    /**
     * @brief IQ4_XS tensor (4-bit extra-small IQ)
     */
    class IQ4_XSTensor : public TensorBase, public IBlockDecoder
    {
    public:
        IQ4_XSTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~IQ4_XSTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ4_XS; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override;
        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override;
        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return IQ4_XSBlock::BLOCK_SIZE; }

    private:
        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_;
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        static void decodeBlock(const IQ4_XSBlock &block, float *output);
    };

    // Implementation: IQ2_XXSTensor.cpp
    /**
     * @brief IQ2_XXS tensor (2-bit extra-extra-small IQ)
     */
    class IQ2_XXSTensor : public TensorBase, public IBlockDecoder
    {
    public:
        IQ2_XXSTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~IQ2_XXSTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ2_XXS; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override;
        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override;
        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return IQ2_XXSBlock::BLOCK_SIZE; }

    private:
        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_;
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        static void decodeBlock(const IQ2_XXSBlock &block, float *output);
    };

    // Implementation: IQ2_XSTensor.cpp
    /**
     * @brief IQ2_XS tensor (2-bit extra-small IQ)
     */
    class IQ2_XSTensor : public TensorBase, public IBlockDecoder
    {
    public:
        IQ2_XSTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~IQ2_XSTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ2_XS; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override;
        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override;
        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return IQ2_XSBlock::BLOCK_SIZE; }

    private:
        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_;
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        static void decodeBlock(const IQ2_XSBlock &block, float *output);
    };

    // Implementation: IQ3_XXSTensor.cpp
    /**
     * @brief IQ3_XXS tensor (3-bit extra-extra-small IQ)
     */
    class IQ3_XXSTensor : public TensorBase, public IBlockDecoder
    {
    public:
        IQ3_XXSTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~IQ3_XXSTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ3_XXS; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override;
        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override;
        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return IQ3_XXSBlock::BLOCK_SIZE; }

    private:
        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_;
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        static void decodeBlock(const IQ3_XXSBlock &block, float *output);
    };

    // Implementation: IQ2_STensor.cpp
    /**
     * @brief IQ2_S tensor (2-bit small IQ)
     */
    class IQ2_STensor : public TensorBase, public IBlockDecoder
    {
    public:
        IQ2_STensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~IQ2_STensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ2_S; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override;
        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override;
        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return IQ2_SBlock::BLOCK_SIZE; }

    private:
        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_;
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        static void decodeBlock(const IQ2_SBlock &block, float *output);
    };

    // Implementation: IQ3_STensor.cpp
    /**
     * @brief IQ3_S tensor (3-bit small IQ)
     */
    class IQ3_STensor : public TensorBase, public IBlockDecoder
    {
    public:
        IQ3_STensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~IQ3_STensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ3_S; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override;
        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override;
        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return IQ3_SBlock::BLOCK_SIZE; }

    private:
        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_;
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        static void decodeBlock(const IQ3_SBlock &block, float *output);
    };

    // Implementation: IQ1_STensor.cpp
    /**
     * @brief IQ1_S tensor (1-bit small IQ)
     */
    class IQ1_STensor : public TensorBase, public IBlockDecoder
    {
    public:
        IQ1_STensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~IQ1_STensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ1_S; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override;
        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override;
        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return IQ1_SBlock::BLOCK_SIZE; }

    private:
        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_;
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        static void decodeBlock(const IQ1_SBlock &block, float *output);
    };

    // Implementation: IQ1_MTensor.cpp
    /**
     * @brief IQ1_M tensor (1-bit medium IQ)
     */
    class IQ1_MTensor : public TensorBase, public IBlockDecoder
    {
    public:
        IQ1_MTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_data);
        ~IQ1_MTensor() override;

        const std::vector<size_t> &shape() const override { return shape_; }
        TensorType native_type() const override { return TensorType::IQ1_M; }

        int device_index() const override { return device_idx_; }
        bool set_device(int device_idx) override;
        bool is_on_device(int device_idx) const override { return device_idx_ == device_idx; }

        const float *data() const override;
        float *mutable_data() override;

        bool copyFrom(const TensorBase *src) override; // Phase 4.2: Stub (read-only)

        std::unique_ptr<ITensorGemm> createGemm() override;
        std::unique_ptr<ITensorRoPE> createRoPE() override;
        std::unique_ptr<ITensorSwiGLU> createSwiGLU() override;
        std::unique_ptr<ITensorSoftmax> createSoftmax() override;
        std::unique_ptr<ITensorRMSNorm> createRMSNorm() override;

        // View support (not implemented - returns nullptr)
        std::shared_ptr<TensorBase> create_view(
            const std::vector<size_t> &new_shape,
            size_t offset = 0) override
        {
            (void)new_shape;
            (void)offset;
            return nullptr; // TODO: Implement views if needed
        }

        void decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const override;
        const void *get_raw_block_at(size_t row_idx, size_t k_block_offset) const override;
        size_t decoder_rows() const override { return shape_[0]; }
        size_t decoder_cols() const override { return shape_[1]; }
        size_t block_size() const override { return IQ1_MBlock::BLOCK_SIZE; }

    private:
        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_data_;
        int device_idx_;
        void *device_blocks_;
        mutable std::vector<float> dequant_cache_;
        static void decodeBlock(const IQ1_MBlock &block, float *output);
    };

} // namespace llaminar2
