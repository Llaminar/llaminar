/**
 * @file FP16Tensor.cpp
 * @brief FP16 tensor implementation (16-bit IEEE 754 half-precision)
 * @author David Sanftenberg
 */

#include "Tensors.h"
#include "../kernels/KernelFactory.h"
#include "../utils/Logger.h"
#include "../utils/DebugEnv.h"
#include "../kernels/cpu/ops/CPURMSNormTypedKernel.h"
#include "../kernels/cpu/attention/CpuAttentionKernelT.h"
#include "../kernels/cpu/ops/CPURoPEKernelT.h"
#include "../kernels/cpu/ops/CPUSwiGLUKernelT.h"
#include "../kernels/cpu/ops/CPUSoftmaxKernelT.h"
#include "../backends/ComputeBackend.h"
#include <cstring>
#include <stdexcept>
#include "SIMDHelpers.h"
#include "FP16Utils.h"
#include <algorithm>
#include <cmath>
#include <omp.h>

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

    // =========================================================================
    // Lazy Transfer Accessors (Phase 3)
    // =========================================================================

    void *FP16Tensor::raw_host_data_ptr()
    {
        if (is_view_)
        {
            return parent_data_ptr_->data() + view_offset_;
        }
        return host_fp16_data_.data();
    }

    const void *FP16Tensor::raw_host_data_ptr() const
    {
        if (is_view_)
        {
            return parent_data_ptr_->data() + view_offset_;
        }
        return host_fp16_data_.data();
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
        // Use centralized KernelFactory for device-aware dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(device_idx_);
        return llaminar::v2::kernels::KernelFactory::createGemm(this, dev_type);
    }

    std::unique_ptr<ITensorRoPE> FP16Tensor::createRoPE()
    {
        return std::make_unique<CPURoPEKernelT<FP16Tensor>>();
    }

    std::unique_ptr<ITensorSwiGLU> FP16Tensor::createSwiGLU()
    {
        return std::make_unique<CPUSwiGLUKernelT<FP16Tensor>>();
    }

    std::unique_ptr<ITensorSoftmax> FP16Tensor::createSoftmax()
    {
        return std::make_unique<CPUSoftmaxKernelT<FP16Tensor>>();
    }

    std::unique_ptr<ITensorRMSNorm> FP16Tensor::createRMSNorm()
    {
        // FP16 tensors use typed RMSNorm kernel
        return std::make_unique<CPURMSNormTypedKernel<ActivationPrecision::FP16>>();
    }

    std::unique_ptr<ITensorAttention> FP16Tensor::createAttention()
    {
        // FP16 tensors use templated CPU attention kernel
        return std::make_unique<CpuAttentionKernelT<FP16Tensor>>();
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

        // Convert FP16 → FP32 using utility function with OpenMP parallelization for large tensors
#pragma omp parallel for schedule(static) if (count > 10000)
        for (size_t i = 0; i < count; ++i)
        {
            fp32_data[i] = fp16_to_fp32(fp16_ptr[i]);
        }
    }

    bool FP16Tensor::from_int32_with_scales(
        const int32_t *accum,
        int rows,
        int cols,
        const float *row_scales,
        const float *col_scales,
        const float *bias)
    {
        if (!accum)
        {
            LOG_ERROR("[FP16Tensor::from_int32_with_scales] accum buffer is null");
            return false;
        }

        if (shape_.size() != 2)
        {
            LOG_ERROR("[FP16Tensor::from_int32_with_scales] tensor must be 2D, got " << shape_.size() << "D");
            return false;
        }
        if (static_cast<int>(shape_[0]) != rows || static_cast<int>(shape_[1]) != cols)
        {
            LOG_ERROR("[FP16Tensor::from_int32_with_scales] shape mismatch: tensor=[" << shape_[0]
                                                                                      << ", " << shape_[1] << "] input=[" << rows << ", " << cols << "]");
            return false;
        }

        uint16_t *dst = is_view_ ? (parent_data_ptr_->data() + view_offset_) : host_fp16_data_.data();

#pragma omp parallel
        {
            std::vector<float> row_buffer(static_cast<size_t>(cols));

#pragma omp for
            for (int r = 0; r < rows; ++r)
            {
                const float row_scale = row_scales ? row_scales[r] : 1.0f;
                const size_t offset = static_cast<size_t>(r) * static_cast<size_t>(cols);

                simd::requantize_int32_row_to_fp32(
                    accum + offset,
                    row_buffer.data(),
                    cols,
                    row_scale,
                    col_scales,
                    bias);

                simd::convert_fp32_to_fp16(row_buffer.data(), dst + offset, static_cast<size_t>(cols));
            }
        }

        dequant_cache_.clear();
        return true;
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

        // Use SIMD-optimized conversion with OpenMP
        // Process in chunks to allow parallel execution of vectorized code
        const size_t chunk_size = 4096; // Large enough to amortize thread overhead
        const size_t num_chunks = (count + chunk_size - 1) / chunk_size;

#pragma omp parallel for
        for (size_t i = 0; i < num_chunks; ++i)
        {
            size_t start = i * chunk_size;
            size_t current_chunk = std::min(chunk_size, count - start);
            simd::convert_fp16_to_fp32(src + start, dst + start, current_chunk);
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

        // Quantize using per-column scales (or per-row if requested)
        const bool use_row_scales = (dst_row_scales != nullptr);

        for (size_t i = 0; i < rows; ++i)
        {
            const float row_inv_scale = use_row_scales ? (1.0f / dst_row_scales[i]) : 1.0f;

            for (size_t j = 0; j < cols; ++j)
            {
                const size_t idx = i * cols + j;
                float inv_scale;

                if (use_row_scales)
                {
                    inv_scale = row_inv_scale;
                }
                else
                {
                    inv_scale = 1.0f / dst_col_scales[j];
                }

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

    ActivationPack FP16Tensor::to_int8_activation_pack(int rows, int cols) const
    {
        if (rows <= 0 || cols <= 0)
        {
            LOG_ERROR("[FP16Tensor] to_int8_activation_pack requires positive dimensions");
            return {};
        }

        const auto &shp = shape();
        if (shp.size() != 2)
        {
            LOG_ERROR("[FP16Tensor] to_int8_activation_pack requires 2D tensor, got " << shp.size() << "D");
            return {};
        }
        if (static_cast<size_t>(rows) > shp[0] || static_cast<size_t>(cols) != shp[1])
        {
            LOG_ERROR("[FP16Tensor] to_int8_activation_pack dimension mismatch: tensor is ["
                      << shp[0] << ", " << shp[1] << "], requested " << rows << "x" << cols);
            return {};
        }

        ActivationPack pack;
        pack.rows = rows;
        pack.cols = cols;
        const size_t row_stride = static_cast<size_t>(cols);
        const size_t total = row_stride * static_cast<size_t>(rows);
        pack.data.resize(total);
        pack.row_scales.resize(static_cast<size_t>(rows));

        const uint16_t *src_data = fp16_data();

#pragma omp parallel
        {
            // Per-thread buffer to avoid repeated allocations
            std::vector<float> row_buffer(cols);

#pragma omp for
            for (int m = 0; m < rows; ++m)
            {
                const uint16_t *row_src = src_data + static_cast<size_t>(m) * row_stride;
                int8_t *row_dst = pack.data.data() + static_cast<size_t>(m) * row_stride;

                // Convert FP16 row to FP32
                simd::convert_fp16_to_fp32(row_src, row_buffer.data(), cols);

                // Calculate max abs from FP32 buffer
                float max_abs = simd::activation_row_max_abs(row_buffer.data(), cols);

                const float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
                pack.row_scales[static_cast<size_t>(m)] = scale;
                const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

                simd::quantize_activation_row(row_buffer.data(), cols, inv_scale, row_dst);
            }
        }

        return pack;
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
            fp16_ptr,                                          // Input: FP16 values
            q8_1_block.qs,                                     // Output: Q8_1 int8 values
            &q8_1_block.d,                                     // Output: Q8_1 FP16 scale
            reinterpret_cast<uint16_t *>(&q8_1_block.sum_qs)); // Output: Q8_1 INT16 pre-computed sum (Nov 2024: changed from FP16 's')

        return &q8_1_block;
    }

    // ===== Bulk Q8_1 Quantization =====

    bool FP16Tensor::quantize_to_q8_1(void *q8_1_buffer, int m, int k) const
    {
        if (!q8_1_buffer || m <= 0 || k <= 0)
        {
            return false;
        }

        // Validate dimensions against tensor shape
        const size_t cols = shape_[1];
        const size_t rows = shape_[0];
        if (static_cast<size_t>(m) > rows || static_cast<size_t>(k) > cols)
        {
            return false;
        }

        const int k_blocks = (k + 31) / 32;
        Q8_1Block *all_blocks = reinterpret_cast<Q8_1Block *>(q8_1_buffer);
        const uint16_t *fp16_data = host_fp16_data_.data();

#pragma omp parallel
        {
            // Parallelize over rows for large M, or collapse(2) for small M
            int quant_thresh = debugEnv().gemm.gemm_quant_parallel_threshold;
            if (quant_thresh == 0)
                quant_thresh = omp_get_num_threads();

            // Thread-local buffer for FP16→FP32 conversion
            alignas(64) float fp32_block[32];

            if (m < quant_thresh)
            {
#pragma omp for collapse(2) schedule(static)
                for (int i = 0; i < m; ++i)
                {
                    for (int k_blk = 0; k_blk < k_blocks; ++k_blk)
                    {
                        const uint16_t *fp16_row = fp16_data + i * cols;
                        Q8_1Block *row_blocks = all_blocks + i * k_blocks;

                        // Convert FP16 block to FP32
                        const int k_start = k_blk * 32;
                        for (int j = 0; j < 32; ++j)
                        {
                            int col_idx = k_start + j;
                            fp32_block[j] = (col_idx < k) ? fp16_to_fp32(fp16_row[col_idx]) : 0.0f;
                        }

                        // Find max absolute value in block
                        float max_abs = 0.0f;
                        for (int j = 0; j < 32; ++j)
                        {
                            float val = std::abs(fp32_block[j]);
                            if (val > max_abs)
                                max_abs = val;
                        }

                        // Compute scale
                        float d = max_abs / 127.0f;
                        if (d < 1e-10f)
                            d = 1e-10f;
                        float id = 1.0f / d;

                        row_blocks[k_blk].d = fp32_to_fp16(d);

                        // Quantize values and compute sum
                        int32_t sum_qs = 0;
                        for (int j = 0; j < 32; ++j)
                        {
                            int8_t q = static_cast<int8_t>(std::round(fp32_block[j] * id));
                            row_blocks[k_blk].qs[j] = q;
                            sum_qs += q;
                        }

                        row_blocks[k_blk].sum_qs = static_cast<int16_t>(sum_qs);
                    }
                }
            }
            else
            {
#pragma omp for schedule(static)
                for (int i = 0; i < m; ++i)
                {
                    const uint16_t *fp16_row = fp16_data + i * cols;
                    Q8_1Block *row_blocks = all_blocks + i * k_blocks;

                    for (int k_blk = 0; k_blk < k_blocks; ++k_blk)
                    {
                        // Convert FP16 block to FP32
                        const int k_start = k_blk * 32;
                        for (int j = 0; j < 32; ++j)
                        {
                            int col_idx = k_start + j;
                            fp32_block[j] = (col_idx < k) ? fp16_to_fp32(fp16_row[col_idx]) : 0.0f;
                        }

                        // Find max absolute value in block
                        float max_abs = 0.0f;
                        for (int j = 0; j < 32; ++j)
                        {
                            float val = std::abs(fp32_block[j]);
                            if (val > max_abs)
                                max_abs = val;
                        }

                        // Compute scale
                        float d = max_abs / 127.0f;
                        if (d < 1e-10f)
                            d = 1e-10f;
                        float id = 1.0f / d;

                        row_blocks[k_blk].d = fp32_to_fp16(d);

                        // Quantize values and compute sum
                        int32_t sum_qs = 0;
                        for (int j = 0; j < 32; ++j)
                        {
                            int8_t q = static_cast<int8_t>(std::round(fp32_block[j] * id));
                            row_blocks[k_blk].qs[j] = q;
                            sum_qs += q;
                        }

                        row_blocks[k_blk].sum_qs = static_cast<int16_t>(sum_qs);
                    }
                }
            }
        }

        return true;
    }

} // namespace llaminar2
