/**
 * @file FP16Tensor.cpp
 * @brief FP16 tensor implementation (16-bit IEEE 754 half-precision)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include <cstring>
#include <stdexcept>

namespace llaminar2
{
    // ========== Constructors ==========

    FP16Tensor::FP16Tensor(const std::vector<size_t> &shape)
        : shape_(shape), device_idx_(-1), device_data_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("FP16Tensor: shape cannot be empty");
        }

        size_t total = 1;
        for (size_t dim : shape)
        {
            total *= dim;
        }

        host_fp16_data_.resize(total, 0);
    }

    FP16Tensor::FP16Tensor(const std::vector<size_t> &shape, const std::vector<uint16_t> &fp16_data)
        : shape_(shape), device_idx_(-1), device_data_(nullptr), host_fp16_data_(fp16_data)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("FP16Tensor: shape cannot be empty");
        }

        size_t total = 1;
        for (size_t dim : shape)
        {
            total *= dim;
        }

        if (fp16_data.size() != total)
        {
            throw std::invalid_argument("FP16Tensor: data size mismatch");
        }
    }

    FP16Tensor::~FP16Tensor()
    {
        // TODO: Free device memory when device support is added
    }

    // ========== TensorBase Interface ==========

    bool FP16Tensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        // TODO: Upload to device when device support is added
        return true;
    }

    const float *FP16Tensor::data() const
    {
        // Lazy dequantization to FP32 cache
        if (dequant_cache_.empty())
        {
            dequant_cache_.resize(host_fp16_data_.size());
            to_fp32(dequant_cache_.data(), host_fp16_data_.size());
        }
        return dequant_cache_.data();
    }

    float *FP16Tensor::mutable_data()
    {
        throw std::runtime_error("FP16Tensor::mutable_data: FP16 tensors are immutable (use from_fp32 to update)");
    }

    bool FP16Tensor::copyFrom(const TensorBase *src)
    {
        if (!src)
        {
            return false;
        }

        // Check shape compatibility
        if (src->shape() != shape_)
        {
            return false;
        }

        // Get FP32 data from source and convert to FP16
        const float *src_data = src->data();
        if (!src_data)
        {
            return false;
        }

        // Calculate element count from shape
        size_t element_count = 1;
        for (size_t dim : shape_)
        {
            element_count *= dim;
        }

        // Convert FP32 to FP16 and store
        from_fp32(src_data, element_count);

        // Clear cache to force re-dequantization
        dequant_cache_.clear();

        return true;
    }

    std::unique_ptr<ITensorGemm> FP16Tensor::createGemm()
    {
        throw std::runtime_error("FP16Tensor: GEMM not yet implemented");
    }

    std::unique_ptr<ITensorRoPE> FP16Tensor::createRoPE()
    {
        throw std::runtime_error("FP16Tensor: RoPE not supported");
    }

    std::unique_ptr<ITensorSwiGLU> FP16Tensor::createSwiGLU()
    {
        throw std::runtime_error("FP16Tensor: SwiGLU not supported");
    }

    std::unique_ptr<ITensorSoftmax> FP16Tensor::createSoftmax()
    {
        throw std::runtime_error("FP16Tensor: Softmax not supported");
    }

    std::unique_ptr<ITensorRMSNorm> FP16Tensor::createRMSNorm()
    {
        throw std::runtime_error("FP16Tensor: RMSNorm not supported");
    }

    // ========== FP16-Specific Interface ==========

    void FP16Tensor::from_fp32(const float *fp32_data, size_t count)
    {
        if (count != host_fp16_data_.size())
        {
            throw std::invalid_argument("FP16Tensor::from_fp32: size mismatch");
        }

        // Convert FP32 → FP16
        // TODO: Add SIMD/hardware acceleration
        for (size_t i = 0; i < count; ++i)
        {
            // Round to nearest even for FP32 → FP16
            uint32_t bits;
            std::memcpy(&bits, &fp32_data[i], sizeof(float));

            uint32_t sign = (bits >> 16) & 0x8000;
            uint32_t exponent = ((bits >> 23) & 0xFF) - 112;
            uint32_t mantissa = (bits >> 13) & 0x3FF;

            if (exponent <= 0)
            {
                // Subnormal or zero
                host_fp16_data_[i] = static_cast<uint16_t>(sign);
            }
            else if (exponent >= 0x1F)
            {
                // Overflow → infinity
                host_fp16_data_[i] = static_cast<uint16_t>(sign | 0x7C00);
            }
            else
            {
                // Normal value
                host_fp16_data_[i] = static_cast<uint16_t>(sign | (exponent << 10) | mantissa);
            }
        }

        // Invalidate FP32 cache
        dequant_cache_.clear();
    }

    void FP16Tensor::to_fp32(float *fp32_data, size_t count) const
    {
        if (count != host_fp16_data_.size())
        {
            throw std::invalid_argument("FP16Tensor::to_fp32: size mismatch");
        }

        // Convert FP16 → FP32 using utility function
        for (size_t i = 0; i < count; ++i)
        {
            fp32_data[i] = fp16_to_fp32(host_fp16_data_[i]);
        }
    }

    bool FP16Tensor::sync_to_device()
    {
        // TODO: Implement device upload
        return false;
    }

    bool FP16Tensor::sync_from_device()
    {
        // TODO: Implement device download
        return false;
    }

} // namespace llaminar2
