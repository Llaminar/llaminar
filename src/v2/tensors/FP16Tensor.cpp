/**
 * @file FP16Tensor.cpp
 * @brief FP16 tensor implementation (16-bit IEEE 754 half-precision)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "../utils/Logger.h"
#include "../kernels/cpu/CPURMSNormKernel.h"
#include "../kernels/cpu/CPURoPEKernel.h"
#include "../kernels/cpu/gemm/GemmAutoTuner.h"
#include "../backends/ComputeBackend.h"
#ifdef HAVE_CUDA
#include "../kernels/cuda/CudaGemmFactory.h"
#endif
#include <cstring>
#include <stdexcept>
#include "SIMDHelpers.h"
#include "FP16Utils.h"
#include <algorithm>
#include <cmath>

namespace llaminar2
{
    // ========== Constructors ==========

    FP16Tensor::FP16Tensor(const std::vector<size_t> &shape)
        : shape_(shape), device_idx_(-1), device_data_(nullptr),
          is_view_(false), parent_data_ptr_(nullptr), view_offset_(0), parent_(nullptr)
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
        : shape_(shape), device_idx_(-1), device_data_(nullptr), host_fp16_data_(fp16_data.size()),
          is_view_(false), parent_data_ptr_(nullptr), view_offset_(0), parent_(nullptr)
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

        // Copy std::vector data into AlignedVector
        std::copy(fp16_data.begin(), fp16_data.end(), host_fp16_data_.begin());
    }

    // Private view constructor
    FP16Tensor::FP16Tensor(const std::vector<size_t> &shape,
                           int device_idx,
                           AlignedVector<uint16_t> *parent_data,
                           size_t data_offset,
                           std::shared_ptr<FP16Tensor> parent)
        : shape_(shape), device_idx_(device_idx), device_data_(nullptr),
          is_view_(true), parent_data_ptr_(parent_data), view_offset_(data_offset),
          parent_(parent)
    {
        // Views don't allocate their own host_fp16_data_
        // They borrow from the parent via parent_data_ptr_
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
            // Get the actual data pointer (owned or borrowed)
            const uint16_t *fp16_ptr = is_view_
                                           ? (parent_data_ptr_->data() + view_offset_)
                                           : host_fp16_data_.data();

            // Calculate element count from shape
            size_t element_count = 1;
            for (size_t dim : shape_)
            {
                element_count *= dim;
            }

            dequant_cache_.resize(element_count);
            to_fp32(dequant_cache_.data(), element_count);
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
        // Route to appropriate backend based on tensor's device placement
        if (device_idx_ >= 0)
        {
            // Tensor is on a GPU device - get device type from DeviceManager
            auto &dm = DeviceManager::instance();
            const auto &devices = dm.devices();

            if (static_cast<size_t>(device_idx_) >= devices.size())
            {
                LOG_ERROR("[FP16Tensor] Invalid device_idx: " << device_idx_);
                throw std::runtime_error("FP16Tensor::createGemm: invalid device index");
            }

            const auto &device = devices[device_idx_];

            // Route based on backend type
            switch (device.type)
            {
#ifdef HAVE_CUDA
            case ComputeBackendType::GPU_CUDA:
                LOG_DEBUG("[FP16Tensor] Creating CUDA GEMM kernel for device " << device_idx_);
                return llaminar::v2::kernels::cuda::createCudaGemm(this);
#endif
#ifdef HAVE_ROCM
            case ComputeBackendType::GPU_ROCM:
                LOG_ERROR("[FP16Tensor] ROCm GEMM not yet implemented");
                throw std::runtime_error("ROCm GEMM not implemented");
#endif
            default:
                LOG_ERROR("[FP16Tensor] Unsupported GPU backend type: " << static_cast<int>(device.type));
                throw std::runtime_error("Unsupported GPU backend type");
            }
        }
        else
        {
            // Tensor is on CPU - use auto-tuned CPU kernel
            // FP16 implements ITensorGemmTileDataProvider interface (used generically for auto-tuner)
            LOG_DEBUG("[FP16Tensor] Creating CPU GEMM kernel with auto-tuner");
            return llaminar::v2::kernels::createAutoTunedGemm(this);
        }
    }

    std::unique_ptr<ITensorRoPE> FP16Tensor::createRoPE()
    {
        return std::make_unique<CPURoPEKernel>();
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
        // FP16 tensors use native FP16 RMSNorm kernel (no conversion to FP32)
        return std::make_unique<CPURMSNormKernel>();
    }

    std::unique_ptr<ITensorAttention> FP16Tensor::createAttention()
    {
        LOG_ERROR("[FP16Tensor] createAttention not supported for quantized tensors");
        return nullptr;
    }

    // ========== FP16-Specific Interface ==========

    void FP16Tensor::from_fp32(const float *fp32_data, size_t count)
    {
        // Get the actual data pointer (owned or borrowed)
        uint16_t *fp16_ptr = is_view_
                                 ? (parent_data_ptr_->data() + view_offset_)
                                 : host_fp16_data_.data();

        // Calculate expected element count from shape
        size_t expected_count = 1;
        for (size_t dim : shape_)
        {
            expected_count *= dim;
        }

        if (count != expected_count)
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
            int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 112; // Signed!
            uint32_t mantissa = (bits >> 13) & 0x3FF;

            if (exponent <= 0)
            {
                // Subnormal or zero
                fp16_ptr[i] = static_cast<uint16_t>(sign);
            }
            else if (exponent >= 0x1F)
            {
                // Overflow → infinity
                fp16_ptr[i] = static_cast<uint16_t>(sign | 0x7C00);
            }
            else
            {
                // Normal value
                fp16_ptr[i] = static_cast<uint16_t>(sign | (exponent << 10) | mantissa);
            }
        }

        // Invalidate FP32 cache
        dequant_cache_.clear();
    }

    void FP16Tensor::to_fp32(float *fp32_data, size_t count) const
    {
        // Get the actual data pointer (owned or borrowed)
        const uint16_t *fp16_ptr = is_view_
                                       ? (parent_data_ptr_->data() + view_offset_)
                                       : host_fp16_data_.data();

        // Calculate expected element count from shape
        size_t expected_count = 1;
        for (size_t dim : shape_)
        {
            expected_count *= dim;
        }

        if (count != expected_count)
        {
            throw std::invalid_argument("FP16Tensor::to_fp32: size mismatch");
        }

        // Convert FP16 → FP32 using utility function
        for (size_t i = 0; i < count; ++i)
        {
            fp32_data[i] = fp16_to_fp32(fp16_ptr[i]);
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

    std::shared_ptr<TensorBase> FP16Tensor::create_view(
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
            LOG_ERROR("[FP16Tensor::create_view] ERROR: offset " << offset
                                                                 << " >= parent size " << parent_elements);
            return nullptr;
        }

        size_t available_elements = parent_elements - offset;
        if (view_elements > available_elements)
        {
            LOG_ERROR("[FP16Tensor::create_view] ERROR: view size " << view_elements
                                                                    << " > available elements " << available_elements
                                                                    << " (offset=" << offset << ", parent_size=" << parent_elements << ")");
            return nullptr;
        }

        // Determine the actual parent tensor and data pointer
        // If this is already a view, chain to the root parent
        std::shared_ptr<FP16Tensor> root_parent;
        if (is_view_)
        {
            root_parent = std::dynamic_pointer_cast<FP16Tensor>(parent_);
            if (!root_parent)
            {
                LOG_ERROR("[FP16Tensor::create_view] ERROR: Failed to cast parent to FP16Tensor (is_view=true)");
                return nullptr;
            }
        }
        else
        {
            // Get a proper shared_ptr to this object (increments ref count)
            try
            {
                auto self_ptr = shared_from_this();
                root_parent = std::dynamic_pointer_cast<FP16Tensor>(self_ptr);
                if (!root_parent)
                {
                    LOG_ERROR("[FP16Tensor::create_view] ERROR: Failed to cast shared_from_this to FP16Tensor");
                    return nullptr;
                }
            }
            catch (const std::bad_weak_ptr &e)
            {
                LOG_ERROR("[FP16Tensor::create_view] ERROR: shared_from_this() failed - object not managed by shared_ptr!");
                LOG_ERROR("[FP16Tensor::create_view] Exception: " << e.what());
                return nullptr;
            }
        }

        AlignedVector<uint16_t> *root_data = is_view_ ? parent_data_ptr_ : &host_fp16_data_;
        size_t root_offset = is_view_ ? (view_offset_ + offset) : offset;

        // Create view using private constructor
        auto view_tensor = std::shared_ptr<FP16Tensor>(new FP16Tensor(
            new_shape,
            device_idx_,
            root_data,
            root_offset,
            root_parent));

        return view_tensor;
    }

    // ===== Format Conversion Methods (TensorBase interface) =====

    // ===== Format Conversion Methods (TensorBase interface) =====

    void FP16Tensor::to_fp32(float *dst) const
    {
        const size_t count = element_count();
        const uint16_t *src = fp16_data();

#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = fp16_to_fp32(src[i]);
        }
    }

    void FP16Tensor::to_bf16(uint16_t *dst) const
    {
        // Decode to FP32 first, then convert to BF16
        const size_t count = element_count();
        std::vector<float> temp_fp32(count);
        to_fp32(temp_fp32.data());

#pragma omp parallel for
        for (size_t i = 0; i < count; ++i)
        {
            dst[i] = simd::fp32_to_bf16(temp_fp32[i]);
        }
    }

    void FP16Tensor::to_fp16(uint16_t *dst) const
    {
        // FP16 → FP16 is just a memcpy
        const size_t count = element_count();
        const uint16_t *src = fp16_data();
        std::memcpy(dst, src, count * sizeof(uint16_t));
    }

    void FP16Tensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
    {
        // Decode to FP32 first, then quantize to int8
        const size_t total_elements = element_count();
        std::vector<float> temp_fp32(total_elements);
        to_fp32(temp_fp32.data());

        const size_t num_blocks = (total_elements + block_size - 1) / block_size;

#pragma omp parallel for
        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
        {
            const size_t offset = block_idx * block_size;
            const size_t count = std::min(block_size, total_elements - offset);

            // Find max absolute value in block
            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(temp_fp32[offset + i]));
            }

            // Compute scale factor (avoid division by zero)
            const float scale = (max_abs > 1e-10f) ? (127.0f / max_abs) : 0.0f;
            dst_scales[block_idx] = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

            // Quantize block to int8 with rounding
            for (size_t i = 0; i < count; ++i)
            {
                const float val = temp_fp32[offset + i] * scale;
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

    bool FP16Tensor::to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales) const
    {
        // FP16 → FP32 → INT8 per-channel quantization
        // Use the generic conversion path since FP16 doesn't implement ITensorGemmTileDataProvider

        if (shape_.size() != 2)
        {
            LOG_ERROR("[FP16Tensor::to_int8_perchannel] Only 2D tensors supported");
            return false;
        }

        const size_t rows = shape_[0];
        const size_t cols = shape_[1];

        // Decode entire tensor to FP32
        std::vector<float> temp_fp32(rows * cols);
        to_fp32(temp_fp32.data());

        // Compute per-column scales
        std::fill(dst_col_scales, dst_col_scales + cols, 0.0f);
        for (size_t j = 0; j < cols; ++j)
        {
            float max_abs = 0.0f;
            for (size_t i = 0; i < rows; ++i)
            {
                max_abs = std::max(max_abs, std::abs(temp_fp32[i * cols + j]));
            }
            dst_col_scales[j] = (max_abs > 1e-10f) ? (max_abs / 127.0f) : 1.0f;
        }

        // Compute per-row scales if requested
        if (dst_row_scales)
        {
            for (size_t i = 0; i < rows; ++i)
            {
                float max_abs = 0.0f;
                for (size_t j = 0; j < cols; ++j)
                {
                    max_abs = std::max(max_abs, std::abs(temp_fp32[i * cols + j]));
                }
                dst_row_scales[i] = (max_abs > 1e-10f) ? (max_abs / 127.0f) : 1.0f;
            }
        }

        // Quantize using per-column scales
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                const size_t idx = i * cols + j;
                const float inv_scale = 1.0f / dst_col_scales[j];
                const float scaled = temp_fp32[idx] * inv_scale;
                const int32_t quantized = static_cast<int32_t>(std::round(scaled));

                // Clamp to INT8 range
                dst_int8[idx] = static_cast<int8_t>(std::max(-127, std::min(127, quantized)));
            }
        }

        return true;
    }

    void FP16Tensor::to_fp32_row(size_t row_idx, float *buffer) const
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
        const uint16_t *src = fp16_data();
        const uint16_t *row_src = src + row_idx * cols;

        for (size_t i = 0; i < cols; ++i)
        {
            buffer[i] = fp16_to_fp32(row_src[i]);
        }
    }

    void FP16Tensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
    {
        if (offset + count > element_count())
        {
            throw std::out_of_range("Span exceeds tensor bounds");
        }

        const uint16_t *src = fp16_data();
        for (size_t i = 0; i < count; ++i)
        {
            buffer[i] = fp16_to_fp32(src[offset + i]);
        }
    }

    bool FP16Tensor::applyRMSNorm(
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
            LOG_ERROR("[FP16Tensor::applyRMSNorm] Failed to create RMSNorm kernel");
            return false;
        }

        // FP16 path: apply_fp16() with FP16 buffers (in-place)
        return kernel->apply_fp16(
            this->fp16_data(),
            gamma,
            this->mutable_fp16_data(),
            seq_len, d_model, eps,
            device_idx);
    }

    bool FP16Tensor::applyRoPE(
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
            LOG_ERROR("[FP16Tensor::applyRoPE] Failed to create RoPE kernel");
            return false;
        }

        // FP16 path: apply_fp16() with FP16 buffers
        // Q is this tensor, K must be FP16 as well
        // Note: K is passed as float* but should actually be FP16
        return kernel->apply_fp16(
            this->mutable_fp16_data(),       // Q (FP16)
            reinterpret_cast<uint16_t *>(K), // K (FP16)
            position_ids,
            seq_len, n_heads, n_kv_heads, head_dim,
            rope_theta,
            device_idx);
    }

    void FP16Tensor::decode_to_q8_0(size_t row_idx, size_t k_block_offset, Q8_0Block *output) const
    {
        // Calculate source offset in FP16 data
        const size_t cols = shape_[1];
        const size_t k_start = k_block_offset * Q8_0Block::BLOCK_SIZE;

        // Bounds check
        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("FP16Tensor::decode_to_q8_0: row_idx out of range");
        }
        if (k_start + Q8_0Block::BLOCK_SIZE > cols)
        {
            throw std::out_of_range("FP16Tensor::decode_to_q8_0: k_block_offset exceeds tensor width");
        }

        // Get pointer to source FP16 data
        const uint16_t *fp16_ptr = host_fp16_data_.data() + row_idx * cols + k_start;

        // Use vectorized decode + quantize (auto-dispatches to AVX512/AVX2/scalar)
        simd::decode_fp16_to_q8_0(
            fp16_ptr,    // Input: FP16 values
            output->qs,  // Output: Q8_0 int8 values
            &output->d); // Output: Q8_0 FP16 scale
    }

    const Q8_1Block *FP16Tensor::decode_to_q8_1(size_t row_idx, size_t k_block_offset) const
    {
        // Calculate source offset in FP16 data
        const size_t cols = shape_[1];
        const size_t k_start = k_block_offset * Q8_1Block::BLOCK_SIZE;

        // Bounds check
        if (row_idx >= shape_[0])
        {
            throw std::out_of_range("FP16Tensor::decode_to_q8_1: row_idx out of range");
        }
        if (k_start + Q8_1Block::BLOCK_SIZE > cols)
        {
            throw std::out_of_range("FP16Tensor::decode_to_q8_1: k_block_offset exceeds tensor width");
        }

        // Get pointer to source FP16 data
        const uint16_t *fp16_ptr = host_fp16_data_.data() + row_idx * cols + k_start;

        // Use thread-local storage to avoid heap allocation
        thread_local Q8_1Block q8_1_block;

        // Use vectorized decode + quantize with pre-computed sum (auto-dispatches to AVX512/AVX2/scalar)
        simd::decode_fp16_to_q8_1(
            fp16_ptr,       // Input: FP16 values
            q8_1_block.qs,  // Output: Q8_1 int8 values
            &q8_1_block.d,  // Output: Q8_1 FP16 scale
            &q8_1_block.s); // Output: Q8_1 FP16 pre-computed sum

        return &q8_1_block;
    }

} // namespace llaminar2
