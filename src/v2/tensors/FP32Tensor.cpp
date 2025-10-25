/**
 * @file FP32Tensor.cpp
 * @brief FP32 tensor implementation
 *
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "TensorKernels.h"
#include "../backends/ComputeBackend.h"
#include "../kernels/cpu/FP32GemmKernel.h"
#include "../kernels/cpu/CPUSoftmaxKernel.h"
#include "../kernels/cpu/CPURMSNormKernel.h"
#include "../kernels/cpu/CPUSwiGLUKernel.h"
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace llaminar2
{

    FP32Tensor::FP32Tensor(const std::vector<size_t> &shape, int device_idx)
        : shape_(shape), device_idx_(device_idx), device_data_(nullptr),
          host_dirty_(false), device_dirty_(false)
    {
        size_t count = 1;
        for (auto dim : shape)
        {
            count *= dim;
        }
        host_data_.resize(count, 0.0f);

        // TODO Phase 4: Allocate device_data_ if device_idx >= 0
        if (device_idx_ >= 0)
        {
            std::cerr << "[FP32Tensor] GPU allocation not yet implemented (device " << device_idx_ << ")\n";
        }
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
        return std::make_unique<FP32GemmKernel>(this);
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

    bool FP32Tensor::copyFrom(const TensorBase *src)
    {
        if (!src)
        {
            std::cerr << "[FP32Tensor::copyFrom] ERROR: Source tensor is null\n";
            return false;
        }

        // Validate shape compatibility
        const auto &src_shape = src->shape();
        if (src_shape != shape_)
        {
            std::cerr << "[FP32Tensor::copyFrom] ERROR: Shape mismatch - src: [";
            for (size_t i = 0; i < src_shape.size(); ++i)
            {
                std::cerr << src_shape[i];
                if (i + 1 < src_shape.size())
                    std::cerr << ", ";
            }
            std::cerr << "], dst: [";
            for (size_t i = 0; i < shape_.size(); ++i)
            {
                std::cerr << shape_[i];
                if (i + 1 < shape_.size())
                    std::cerr << ", ";
            }
            std::cerr << "]\n";
            return false;
        }

        size_t count = 1;
        for (auto dim : shape_)
        {
            count *= dim;
        }

        int src_device = src->device_index();
        int dst_device = device_idx_;

        // Determine transfer type
        bool cpu_to_cpu = (src_device == -1 && dst_device == -1);
        bool cpu_to_gpu = (src_device == -1 && dst_device >= 0);
        bool gpu_to_cpu = (src_device >= 0 && dst_device == -1);
        bool gpu_to_gpu = (src_device >= 0 && dst_device >= 0);

        std::cout << "[FP32Tensor::copyFrom] Transfer: device " << src_device
                  << " → device " << dst_device << " (" << count << " elements)\n";

        if (cpu_to_cpu)
        {
            // CPU → CPU: Simple memcpy
            const float *src_data = src->data();
            std::memcpy(host_data_.data(), src_data, count * sizeof(float));
            host_dirty_ = true; // Mark host as authoritative
            return true;
        }
        else if (cpu_to_gpu)
        {
            // CPU → GPU: Phase 4 CUDA
            std::cerr << "[FP32Tensor::copyFrom] CPU → GPU transfer not yet implemented (Phase 4 CUDA)\n";
            std::cerr << "                         Would copy " << count << " floats from CPU to GPU device " << dst_device << "\n";
            return false;
        }
        else if (gpu_to_cpu)
        {
            // GPU → CPU: Phase 4 CUDA
            std::cerr << "[FP32Tensor::copyFrom] GPU → CPU transfer not yet implemented (Phase 4 CUDA)\n";
            std::cerr << "                         Would copy " << count << " floats from GPU device " << src_device << " to CPU\n";
            return false;
        }
        else if (gpu_to_gpu)
        {
            // GPU → GPU: Phase 4 CUDA (peer-to-peer copy)
            if (src_device == dst_device)
            {
                std::cerr << "[FP32Tensor::copyFrom] Same GPU device (" << src_device << "), no transfer needed\n";
                return true;
            }
            std::cerr << "[FP32Tensor::copyFrom] GPU → GPU transfer not yet implemented (Phase 4 CUDA)\n";
            std::cerr << "                         Would copy " << count << " floats from GPU " << src_device << " to GPU " << dst_device << "\n";
            return false;
        }

        // Should never reach here
        std::cerr << "[FP32Tensor::copyFrom] ERROR: Unknown transfer type\n";
        return false;
    }

} // namespace llaminar2
