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
          host_dirty_(false), device_dirty_(false),
          is_view_(false), parent_data_ptr_(nullptr), view_offset_(0), parent_(nullptr)
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

    FP32Tensor::FP32Tensor(const std::vector<size_t> &shape,
                           int device_idx,
                           std::vector<float> *parent_data,
                           size_t data_offset,
                           std::shared_ptr<FP32Tensor> parent)
        : shape_(shape), device_idx_(device_idx), device_data_(nullptr),
          host_dirty_(false), device_dirty_(false),
          is_view_(true), parent_data_ptr_(parent_data), view_offset_(data_offset),
          parent_(parent)
    {
        // Views don't allocate their own host_data_
        // They borrow from the parent via parent_data_ptr_
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

        if (is_view_)
        {
            return parent_data_ptr_->data() + view_offset_;
        }
        return host_data_.data();
    }

    float *FP32Tensor::mutable_data()
    {
        // Mark as dirty if on device
        // TODO: Implement dirty flag when device support is added
        host_dirty_ = true;

        if (is_view_)
        {
            return parent_data_ptr_->data() + view_offset_;
        }
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
        // FP32 tensors use CPU RMSNorm kernel
        return std::make_unique<CPURMSNormKernel>();
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

    std::shared_ptr<TensorBase> FP32Tensor::create_view(
        const std::vector<size_t> &new_shape,
        size_t offset)
    {
        // Calculate total elements in the new view
        size_t view_elements = 1;
        for (auto dim : new_shape)
        {
            view_elements *= dim;
        }

        // Calculate available elements from offset
        size_t parent_elements = 1;
        for (auto dim : shape_)
        {
            parent_elements *= dim;
        }

        if (offset >= parent_elements)
        {
            std::cerr << "[FP32Tensor::create_view] ERROR: offset " << offset
                      << " >= parent size " << parent_elements << "\n";
            return nullptr;
        }

        size_t available_elements = parent_elements - offset;
        if (view_elements > available_elements)
        {
            std::cerr << "[FP32Tensor::create_view] ERROR: view size " << view_elements
                      << " > available elements " << available_elements
                      << " (offset=" << offset << ", parent_size=" << parent_elements << ")\n";
            return nullptr;
        }

        // Determine the actual parent tensor and data pointer
        // If this is already a view, chain to the root parent
        std::shared_ptr<FP32Tensor> root_parent;
        if (is_view_)
        {
            root_parent = std::dynamic_pointer_cast<FP32Tensor>(parent_);
            if (!root_parent)
            {
                std::cerr << "[FP32Tensor::create_view] ERROR: Failed to cast parent to FP32Tensor (is_view=true)\n";
                return nullptr;
            }
        }
        else
        {
            // Get a proper shared_ptr to this object (increments ref count)
            try
            {
                auto self_ptr = shared_from_this();
                root_parent = std::dynamic_pointer_cast<FP32Tensor>(self_ptr);
                if (!root_parent)
                {
                    std::cerr << "[FP32Tensor::create_view] ERROR: Failed to cast shared_from_this to FP32Tensor\n";
                    return nullptr;
                }
            }
            catch (const std::bad_weak_ptr &e)
            {
                std::cerr << "[FP32Tensor::create_view] ERROR: shared_from_this() failed - object not managed by shared_ptr!\n";
                std::cerr << "[FP32Tensor::create_view] Exception: " << e.what() << "\n";
                return nullptr;
            }
        }

        std::vector<float> *root_data = is_view_ ? parent_data_ptr_ : &host_data_;
        size_t root_offset = is_view_ ? (view_offset_ + offset) : offset;

        // Create view using private constructor
        auto view_tensor = std::shared_ptr<FP32Tensor>(new FP32Tensor(
            new_shape,
            device_idx_,
            root_data,
            root_offset,
            root_parent));

        return view_tensor;
    }

} // namespace llaminar2
