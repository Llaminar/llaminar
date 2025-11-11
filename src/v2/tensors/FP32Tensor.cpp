/**
 * @file FP32Tensor.cpp
 * @brief FP32 tensor implementation
 *
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "../utils/Logger.h"
#include "TensorKernels.h"
#include "SIMDHelpers.h"
#include "FP16Utils.h"
#include "../backends/ComputeBackend.h"
#include "../kernels/cpu/gemm/GemmAutoTuner.h"
#include "../kernels/cpu/CPUSoftmaxKernel.h"
#include "../kernels/cpu/CPURMSNormKernel.h"
#include "../kernels/cpu/CPUSwiGLUKernel.h"
#include "../kernels/cpu/CPUAttention.h"
#include "../kernels/cpu/CPURoPEKernel.h"
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cmath>

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
            LOG_DEBUG("[FP32Tensor] GPU allocation not yet implemented (device " << device_idx_ << ")");
        }
    }

    FP32Tensor::FP32Tensor(const std::vector<size_t> &shape,
                           int device_idx,
                           AlignedVector<float> *parent_data,
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
            LOG_ERROR("[FP32Tensor] TODO: Free device data in destructor");
        }
    }

    bool FP32Tensor::set_device(int device_idx)
    {
        if (device_idx == device_idx_)
        {
            return true; // Already on target device
        }

        // TODO: Implement actual device transfer
        LOG_ERROR("[FP32Tensor] set_device not yet fully implemented");
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
        // FP32 tensors use the auto-tuned GEMM kernel for optimal performance
        // FP32Tensor implements ITensorGemmTileDataProvider with block_size() = 32
        return llaminar::v2::kernels::createAutoTunedGemm(this);
    }

    std::unique_ptr<ITensorRoPE> FP32Tensor::createRoPE()
    {
        return std::make_unique<CPURoPEKernel>();
    }

    std::unique_ptr<ITensorSwiGLU> FP32Tensor::createSwiGLU()
    {
        // TODO: Implement SwiGLU kernel creation
        LOG_ERROR("[FP32Tensor] createSwiGLU not yet implemented");
        return nullptr;
    }

    std::unique_ptr<ITensorSoftmax> FP32Tensor::createSoftmax()
    {
        // TODO: Implement Softmax kernel creation
        LOG_ERROR("[FP32Tensor] createSoftmax not yet implemented");
        return nullptr;
    }

    std::unique_ptr<ITensorRMSNorm> FP32Tensor::createRMSNorm()
    {
        // FP32 tensors use CPU RMSNorm kernel
        return std::make_unique<CPURMSNormKernel>();
    }

    std::unique_ptr<ITensorAttention> FP32Tensor::createAttention()
    {
        // FP32 tensors use CPU Attention kernel
        return std::make_unique<CPUAttention>();
    }

    bool FP32Tensor::sync_to_device()
    {
        // TODO: Implement sync_to_device
        LOG_ERROR("[FP32Tensor] sync_to_device not yet implemented");
        return false;
    }

    bool FP32Tensor::sync_from_device()
    {
        // TODO: Implement sync_from_device
        LOG_ERROR("[FP32Tensor] sync_from_device not yet implemented");
        return false;
    }

    bool FP32Tensor::copyFrom(const TensorBase *src)
    {
        if (!src)
        {
            LOG_ERROR("[FP32Tensor::copyFrom] ERROR: Source tensor is null");
            return false;
        }

        // Validate shape compatibility
        const auto &src_shape = src->shape();
        if (src_shape != shape_)
        {
            std::string src_str = "[";
            for (size_t i = 0; i < src_shape.size(); ++i)
            {
                src_str += std::to_string(src_shape[i]);
                if (i + 1 < src_shape.size())
                    src_str += ", ";
            }
            src_str += "]";

            std::string dst_str = "[";
            for (size_t i = 0; i < shape_.size(); ++i)
            {
                dst_str += std::to_string(shape_[i]);
                if (i + 1 < shape_.size())
                    dst_str += ", ";
            }
            dst_str += "]";

            LOG_ERROR("[FP32Tensor::copyFrom] ERROR: Shape mismatch - src: " << src_str << ", dst: " << dst_str);
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

        LOG_DEBUG("[FP32Tensor::copyFrom] Transfer: device " << src_device
                                                             << " → device " << dst_device << " (" << count << " elements)");

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
            LOG_ERROR("[FP32Tensor::copyFrom] CPU → GPU transfer not yet implemented (Phase 4 CUDA)");
            LOG_DEBUG("                         Would copy " << count << " floats from CPU to GPU device " << dst_device);
            return false;
        }
        else if (gpu_to_cpu)
        {
            // GPU → CPU: Phase 4 CUDA
            LOG_ERROR("[FP32Tensor::copyFrom] GPU → CPU transfer not yet implemented (Phase 4 CUDA)");
            LOG_DEBUG("                         Would copy " << count << " floats from GPU device " << src_device << " to CPU");
            return false;
        }
        else if (gpu_to_gpu)
        {
            // GPU → GPU: Phase 4 CUDA (peer-to-peer copy)
            if (src_device == dst_device)
            {
                LOG_DEBUG("[FP32Tensor::copyFrom] Same GPU device (" << src_device << "), no transfer needed");
                return true;
            }
            LOG_ERROR("[FP32Tensor::copyFrom] GPU → GPU transfer not yet implemented (Phase 4 CUDA)");
            LOG_DEBUG("                         Would copy " << count << " floats from GPU " << src_device << " to GPU " << dst_device);
            return false;
        }

        // Should never reach here
        LOG_ERROR("[FP32Tensor::copyFrom] ERROR: Unknown transfer type");
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
            LOG_ERROR("[FP32Tensor::create_view] ERROR: offset " << offset << " >= parent size " << parent_elements);
            return nullptr;
        }

        size_t available_elements = parent_elements - offset;
        if (view_elements > available_elements)
        {
            LOG_ERROR("[FP32Tensor::create_view] ERROR: view size " << view_elements
                                                                    << " > available elements " << available_elements
                                                                    << " (offset=" << offset << ", parent_size=" << parent_elements << ")");
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
                LOG_ERROR("[FP32Tensor::create_view] ERROR: Failed to cast parent to FP32Tensor (is_view=true)");
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
                    LOG_ERROR("[FP32Tensor::create_view] ERROR: Failed to cast shared_from_this to FP32Tensor");
                    return nullptr;
                }
            }
            catch (const std::bad_weak_ptr &e)
            {
                LOG_ERROR("[FP32Tensor::create_view] ERROR: shared_from_this() failed - object not managed by shared_ptr!");
                LOG_ERROR("[FP32Tensor::create_view] Exception: " << e.what());
                return nullptr;
            }
        }

        AlignedVector<float> *root_data = is_view_ ? parent_data_ptr_ : &host_data_;
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

    // ===== Format Conversion Methods =====

    void FP32Tensor::to_fp32(float *dst) const
    {
        const size_t count = element_count();
        const float *src = data(); // Handles view offset if needed
        std::memcpy(dst, src, count * sizeof(float));
    }

    void FP32Tensor::to_bf16(uint16_t *dst) const
    {
        const size_t count = element_count();
        const float *src = data();

#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = simd::fp32_to_bf16(src[i]);
        }
    }

    void FP32Tensor::to_fp16(uint16_t *dst) const
    {
        const size_t count = element_count();
        const float *src = data();

#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = fp32_to_fp16(src[i]);
        }
    }

    void FP32Tensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
    {
        const size_t total_elements = element_count();
        const size_t num_blocks = (total_elements + block_size - 1) / block_size;
        const float *src = data();

#pragma omp parallel for
        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const size_t offset = block_idx * block_size;
            const size_t count = std::min(block_size, total_elements - offset);

            // Find max absolute value in block
            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[offset + i]));
            }

            // Compute scale factor (avoid division by zero)
            const float scale = (max_abs > 1e-10f) ? (127.0f / max_abs) : 0.0f;
            dst_scales[block_idx] = (scale > 0.0f) ? (1.0f / scale) : 0.0f; // Store inverse for faster dequant

            // Quantize block to int8 with rounding
            for (size_t i = 0; i < count; ++i)
            {
                const float val = src[offset + i] * scale;
                const float clamped = std::max(-127.0f, std::min(127.0f, val));
                dst_int8[offset + i] = static_cast<int8_t>(std::round(clamped));
            }

            // Zero-fill partial block tail (if any)
            for (size_t i = count; i < block_size; ++i)
            {
                dst_int8[offset + i] = 0;
            }
        }
    }

    bool FP32Tensor::to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales) const
    {
        const auto &shp = shape();
        if (shp.size() != 2)
        {
            LOG_ERROR("[FP32Tensor] to_int8_perchannel() requires 2D tensor, got " << shp.size() << "D");
            return false;
        }

        const size_t rows = shp[0];
        const size_t cols = shp[1];
        const float *src = data();

        // Compute per-column scales
        for (size_t j = 0; j < cols; ++j)
        {
            float max_abs = 0.0f;
            for (size_t i = 0; i < rows; ++i)
            {
                float abs_val = std::fabs(src[i * cols + j]);
                if (abs_val > max_abs)
                    max_abs = abs_val;
            }
            dst_col_scales[j] = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        }

        // Compute per-row scales (if requested)
        if (dst_row_scales != nullptr)
        {
            for (size_t i = 0; i < rows; ++i)
            {
                float max_abs = 0.0f;
                for (size_t j = 0; j < cols; ++j)
                {
                    float abs_val = std::fabs(src[i * cols + j]);
                    if (abs_val > max_abs)
                        max_abs = abs_val;
                }
                dst_row_scales[i] = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
            }
        }

        // Quantize to INT8 using per-column scales
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                const size_t idx = i * cols + j;
                const float inv_scale = 1.0f / dst_col_scales[j];
                float scaled = src[idx] * inv_scale;
                int32_t quantized = static_cast<int32_t>(std::round(scaled));

                // Clamp to INT8 range
                if (quantized > 127)
                    quantized = 127;
                else if (quantized < -127)
                    quantized = -127;

                dst_int8[idx] = static_cast<int8_t>(quantized);
            }
        }

        return true;
    }

    void FP32Tensor::to_fp32_row(size_t row_idx, float *buffer) const
    {
        const auto &shp = shape();
        if (shp.size() != 2)
        {
            throw std::runtime_error("to_fp32_row() requires 2D tensor");
        }
        if (row_idx >= shp[0])
        {
            throw std::out_of_range("Row index out of bounds");
        }

        const size_t cols = shp[1];
        const float *src = data();
        std::memcpy(buffer, src + row_idx * cols, cols * sizeof(float));
    }

    void FP32Tensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
    {
        if (offset + count > element_count())
        {
            throw std::out_of_range("Span exceeds tensor bounds");
        }

        const float *src = data();
        std::memcpy(buffer, src + offset, count * sizeof(float));
    }

    bool FP32Tensor::applyRMSNorm(
        const float *gamma,
        int seq_len,
        int d_model,
        float eps,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        auto kernel = createRMSNorm();
        if (!kernel)
        {
            LOG_ERROR("[FP32Tensor::applyRMSNorm] Failed to create RMSNorm kernel");
            return false;
        }

        // FP32 path: apply() with FP32 buffers (in-place)
        return kernel->apply(
            this->data(),
            gamma,
            this->mutable_data(),
            seq_len, d_model, eps,
            false, // normalize_gamma
            mpi_ctx,
            device_idx);
    }

    bool FP32Tensor::applyRoPE(
        float *K,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        bool use_bf16,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        auto kernel = createRoPE();
        if (!kernel)
        {
            LOG_ERROR("[FP32Tensor::applyRoPE] Failed to create RoPE kernel");
            return false;
        }

        // FP32 path: apply() with FP32 buffers
        // Q is this tensor, K is passed as parameter
        return kernel->apply(
            this->mutable_data(), // Q
            K,                    // K
            position_ids,
            seq_len, n_heads, n_kv_heads, head_dim,
            rope_theta,
            use_bf16,
            mpi_ctx,
            device_idx);
    }

    void FP32Tensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const
    {
        // Calculate source offset in FP32 data
        const size_t cols = shape_[1];
        const size_t k_start = k_block_offset * Q8_0Block::BLOCK_SIZE;

        // Bounds check
        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("FP32Tensor::decode_to_q8_0: row_idx out of range");
        }
        if (k_start + Q8_0Block::BLOCK_SIZE > cols)
        {
            throw std::out_of_range("FP32Tensor::decode_to_q8_0: k_block_offset exceeds tensor width");
        }

        // Get pointer to source FP32 data
        const float *fp32_data = data() + row_idx * cols + k_start;

        // Use vectorized quantization (auto-dispatches to AVX512/AVX2/scalar)
        simd::decode_fp32_to_q8_0(
            fp32_data,   // Input: FP32 values
            output->qs,  // Output: Q8_0 int8 values
            &output->d); // Output: Q8_0 FP16 scale
    }

} // namespace llaminar2
