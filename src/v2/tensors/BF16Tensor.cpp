/**
 * @file BF16Tensor.cpp
 * @brief BF16 tensor implementation (Brain Float 16)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "../utils/BFloat16.h"
#include "../kernels/cpu/BF16GemmKernel.h"
#include <cstring>
#include <stdexcept>

namespace llaminar2
{
    // ========== Constructors ==========

    BF16Tensor::BF16Tensor(const std::vector<size_t> &shape)
        : shape_(shape), device_idx_(-1), device_data_(nullptr)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("BF16Tensor: shape cannot be empty");
        }

        size_t total = 1;
        for (size_t dim : shape)
        {
            total *= dim;
        }

        host_bf16_data_.resize(total, 0);
    }

    BF16Tensor::BF16Tensor(const std::vector<size_t> &shape, const std::vector<uint16_t> &bf16_data)
        : shape_(shape), device_idx_(-1), device_data_(nullptr), host_bf16_data_(bf16_data)
    {
        if (shape.empty())
        {
            throw std::invalid_argument("BF16Tensor: shape cannot be empty");
        }

        size_t total = 1;
        for (size_t dim : shape)
        {
            total *= dim;
        }

        if (bf16_data.size() != total)
        {
            throw std::invalid_argument("BF16Tensor: data size mismatch");
        }
    }

    BF16Tensor::~BF16Tensor()
    {
        // TODO: Free device memory when device support is added
    }

    // ========== TensorBase Interface ==========

    bool BF16Tensor::set_device(int device_idx)
    {
        device_idx_ = device_idx;
        // TODO: Upload to device when device support is added
        return true;
    }

    const float *BF16Tensor::data() const
    {
        // Lazy dequantization to FP32 cache
        if (dequant_cache_.empty())
        {
            dequant_cache_.resize(host_bf16_data_.size());
            to_fp32(dequant_cache_.data(), host_bf16_data_.size());
        }
        return dequant_cache_.data();
    }

    float *BF16Tensor::mutable_data()
    {
        throw std::runtime_error("BF16Tensor::mutable_data: BF16 tensors are immutable (use from_fp32 to update)");
    }

    bool BF16Tensor::copyFrom(const TensorBase *src)
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

        // Get FP32 data from source and convert to BF16
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

        // Convert FP32 to BF16 and store
        from_fp32(src_data, element_count);

        // Clear cache to force re-dequantization
        dequant_cache_.clear();

        return true;
    }

    std::unique_ptr<ITensorGemm> BF16Tensor::createGemm()
    {
        return std::make_unique<BF16GemmKernel>(this);
    }

    std::unique_ptr<ITensorRoPE> BF16Tensor::createRoPE()
    {
        throw std::runtime_error("BF16Tensor: RoPE not supported");
    }

    std::unique_ptr<ITensorSwiGLU> BF16Tensor::createSwiGLU()
    {
        throw std::runtime_error("BF16Tensor: SwiGLU not supported");
    }

    std::unique_ptr<ITensorSoftmax> BF16Tensor::createSoftmax()
    {
        throw std::runtime_error("BF16Tensor: Softmax not supported");
    }

    std::unique_ptr<ITensorRMSNorm> BF16Tensor::createRMSNorm()
    {
        throw std::runtime_error("BF16Tensor: RMSNorm not supported");
    }

    // ========== BF16-Specific Interface ==========

    void BF16Tensor::from_fp32(const float *fp32_data, size_t count)
    {
        if (count != host_bf16_data_.size())
        {
            throw std::invalid_argument("BF16Tensor::from_fp32: size mismatch");
        }

        // Convert FP32 → BF16 using utility
        for (size_t i = 0; i < count; ++i)
        {
            bfloat16 bf = bfloat16::from_float(fp32_data[i]);
            host_bf16_data_[i] = bf.data;
        }

        // Invalidate FP32 cache
        dequant_cache_.clear();
    }

    void BF16Tensor::to_fp32(float *fp32_data, size_t count) const
    {
        if (count != host_bf16_data_.size())
        {
            throw std::invalid_argument("BF16Tensor::to_fp32: size mismatch");
        }

        // Convert BF16 → FP32
        for (size_t i = 0; i < count; ++i)
        {
            bfloat16 bf;
            bf.data = host_bf16_data_[i];
            fp32_data[i] = bf.to_float();
        }
    }

    bool BF16Tensor::sync_to_device()
    {
        // TODO: Implement device upload
        return false;
    }

    bool BF16Tensor::sync_from_device()
    {
        // TODO: Implement device download
        return false;
    }

} // namespace llaminar2
