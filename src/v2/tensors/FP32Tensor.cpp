/**
 * @file FP32Tensor.cpp
 * @brief FP32 tensor implementation
 *
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "TensorKernels.h"
#include "../backends/ComputeBackend.h"
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace llaminar2
{

    FP32Tensor::FP32Tensor(const std::vector<size_t> &shape)
        : shape_(shape), device_idx_(-1), device_data_(nullptr),
          host_dirty_(false), device_dirty_(false)
    {
        size_t count = 1;
        for (auto dim : shape)
        {
            count *= dim;
        }
        host_data_.resize(count, 0.0f);
    }

    FP32Tensor::~FP32Tensor()
    {
        // TODO: Free device_data_ if allocated
        if (device_data_)
        {
            std::cerr << "[FP32Tensor] TODO: Free device data in destructor\n";
        }
    }

    bool FP32Tensor::set_device(int device_idx)
    {
        if (device_idx == device_idx_)
        {
            return true; // Already on target device
        }

        // TODO: Implement actual device transfer
        std::cerr << "[FP32Tensor] set_device not yet fully implemented\n";
        device_idx_ = device_idx;
        return true;
    }

    const float *FP32Tensor::data() const
    {
        // If on device, sync to host
        // TODO: Implement lazy sync when device support is added
        return host_data_.data();
    }

    float *FP32Tensor::mutable_data()
    {
        // Mark as dirty if on device
        // TODO: Implement dirty flag when device support is added
        host_dirty_ = true;
        return host_data_.data();
    }

    std::unique_ptr<ITensorGemm> FP32Tensor::createGemm()
    {
        // TODO: Implement GEMM kernel creation
        std::cerr << "[FP32Tensor] createGemm not yet implemented\n";
        return nullptr;
    }

    std::unique_ptr<ITensorRoPE> FP32Tensor::createRoPE()
    {
        // TODO: Implement RoPE kernel creation
        std::cerr << "[FP32Tensor] createRoPE not yet implemented\n";
        return nullptr;
    }

    std::unique_ptr<ITensorSwiGLU> FP32Tensor::createSwiGLU()
    {
        // TODO: Implement SwiGLU kernel creation
        std::cerr << "[FP32Tensor] createSwiGLU not yet implemented\n";
        return nullptr;
    }

    std::unique_ptr<ITensorSoftmax> FP32Tensor::createSoftmax()
    {
        // TODO: Implement Softmax kernel creation
        std::cerr << "[FP32Tensor] createSoftmax not yet implemented\n";
        return nullptr;
    }

    std::unique_ptr<ITensorRMSNorm> FP32Tensor::createRMSNorm()
    {
        // TODO: Implement RMSNorm kernel creation
        std::cerr << "[FP32Tensor] createRMSNorm not yet implemented\n";
        return nullptr;
    }

    bool FP32Tensor::sync_to_device()
    {
        // TODO: Implement sync_to_device
        std::cerr << "[FP32Tensor] sync_to_device not yet implemented\n";
        return false;
    }

    bool FP32Tensor::sync_from_device()
    {
        // TODO: Implement sync_from_device
        std::cerr << "[FP32Tensor] sync_from_device not yet implemented\n";
        return false;
    }

} // namespace llaminar2
