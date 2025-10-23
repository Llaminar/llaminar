/**
 * @file TensorBase.h
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

#include <vector>
#include <memory>
#include <cstddef>

namespace llaminar2
{

    // Forward declarations
    class ITensorGemm;
    class ITensorRoPE;
    class ITensorSwiGLU;
    class ITensorSoftmax;
    class ITensorRMSNorm;

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
        explicit FP32Tensor(const std::vector<size_t> &shape)
            : shape_(shape), device_idx_(-1), device_data_(nullptr),
              host_dirty_(false), device_dirty_(false)
        {
            size_t count = 1;
            for (auto dim : shape)
                count *= dim;
            host_data_.resize(count, 0.0f);
        }

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

    /**
     * @brief IQ4_NL quantized tensor (4-bit with 64-byte blocks)
     */
    class IQ4_NLTensor : public TensorBase
    {
    public:
        IQ4_NLTensor(const std::vector<size_t> &shape, const std::vector<uint8_t> &raw_blocks)
            : shape_(shape), raw_blocks_(raw_blocks), device_idx_(-1), device_blocks_(nullptr) {}

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

        // Direct access to quantized data (for kernels)
        const uint8_t *raw_blocks() const { return raw_blocks_.data(); }
        size_t num_blocks() const { return raw_blocks_.size() / 64; }

    private:
        std::vector<size_t> shape_;
        std::vector<uint8_t> raw_blocks_; // Quantized blocks on host

        int device_idx_;
        void *device_blocks_; // Quantized blocks on device (if uploaded)

        mutable std::vector<float> dequant_cache_; // Temporary for data() calls
    };

} // namespace llaminar2
