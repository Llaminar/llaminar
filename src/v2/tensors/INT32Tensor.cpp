/**
 * @file INT32Tensor.cpp
 * @brief INT32 tensor implementation for accumulator storage
 * @author David Sanftenberg
 * @date 2025-11-05
 */

#include "Tensors.h"
#include "../kernels/cpu/gemm_v4/OneDNNGemmKernel.h"
#include "../utils/Logger.h"
#include "../kernels/cpu/CPURMSNormKernel.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>

namespace llaminar2
{
    // =============================================================================
    // QUANTIZATION HELPERS
    // =============================================================================

    /**
     * @brief Requantize INT32 to INT8 with per-row dynamic scaling
     *
     * This is the KEY function for full INT8 pipelines. It converts INT32
     * accumulator results back to INT8 for the next layer, computing per-row
     * scales to fit the INT32 range into INT8 [-127, 127].
     *
     * Per-row quantization (vs per-tensor) maintains better accuracy:
     * - Each row has independent dynamic range
     * - Prevents outliers in one row from reducing precision in others
     *
     * @param int32_data Input INT32 accumulator [m, n]
     * @param int8_data Output INT8 data [m, n]
     * @param row_scales Output per-row scales [m]
     * @param m Number of rows
     * @param n Number of columns
     */
    static void requantizeINT32ToINT8_PerRow(
        const int32_t *int32_data,
        int8_t *int8_data,
        float *row_scales,
        size_t m, size_t n)
    {
        for (size_t i = 0; i < m; ++i)
        {
            const int32_t *row = int32_data + i * n;

            // Find max absolute value in this row
            int32_t max_abs = 0;
            for (size_t j = 0; j < n; ++j)
            {
                max_abs = std::max(max_abs, std::abs(row[j]));
            }

            // Compute scale to fit into INT8 range [-127, 127]
            // (Reserve -128 to avoid asymmetry issues)
            float scale = (max_abs > 0) ? static_cast<float>(max_abs) / 127.0f : 1.0f;
            row_scales[i] = scale;

            // Quantize row
            float inv_scale = 1.0f / scale;
            for (size_t j = 0; j < n; ++j)
            {
                float scaled = static_cast<float>(row[j]) * inv_scale;
                int8_data[i * n + j] = static_cast<int8_t>(std::round(
                    std::clamp(scaled, -127.0f, 127.0f)));
            }
        }
    }

    /**
     * @brief Dequantize INT32 to FP32 using global scale
     *
     * @param int32_data Input INT32 data
     * @param fp32_data Output FP32 data
     * @param scale Scale factor
     * @param count Number of elements
     */
    static void dequantizeINT32ToFP32(
        const int32_t *int32_data,
        float *fp32_data,
        float scale,
        size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            fp32_data[i] = static_cast<float>(int32_data[i]) * scale;
        }
    }

    /**
     * @brief Quantize FP32 to INT32 with scale factor
     *
     * @param fp32_data Input FP32 data
     * @param int32_data Output INT32 data
     * @param scale Input scale factor
     * @param count Number of elements
     */
    static void quantizeFP32ToINT32(
        const float *fp32_data,
        int32_t *int32_data,
        float scale,
        size_t count)
    {
        float inv_scale = 1.0f / scale;
        for (size_t i = 0; i < count; ++i)
        {
            int32_data[i] = static_cast<int32_t>(std::round(fp32_data[i] * inv_scale));
        }
    }

    // =============================================================================
    // CONSTRUCTORS
    // =============================================================================

    INT32Tensor::INT32Tensor(const std::vector<size_t> &shape)
        : shape_(shape), device_idx_(-1), scale_(1.0f), device_data_(nullptr)
    {
        size_t total_elements = 1;
        for (auto dim : shape)
        {
            total_elements *= dim;
        }
        host_int32_data_.resize(total_elements, 0);
    }

    INT32Tensor::INT32Tensor(const std::vector<size_t> &shape,
                             const std::vector<int32_t> &data)
        : shape_(shape), device_idx_(-1), scale_(1.0f), device_data_(nullptr)
    {
        size_t total_elements = 1;
        for (auto dim : shape)
        {
            total_elements *= dim;
        }

        if (data.size() != total_elements)
        {
            LOG_ERROR("[INT32Tensor] Data size mismatch: got " << data.size()
                                                               << ", expected " << total_elements);
            host_int32_data_.resize(total_elements, 0);
        }
        else
        {
            host_int32_data_.resize(data.size());
            std::copy(data.begin(), data.end(), host_int32_data_.begin());
        }
    }

    INT32Tensor::INT32Tensor(const std::vector<size_t> &shape,
                             const std::vector<float> &fp32_data,
                             float scale)
        : shape_(shape), device_idx_(-1), scale_(scale), device_data_(nullptr)
    {
        size_t total_elements = 1;
        for (auto dim : shape)
        {
            total_elements *= dim;
        }

        if (fp32_data.size() != total_elements)
        {
            LOG_ERROR("[INT32Tensor] FP32 data size mismatch: got " << fp32_data.size()
                                                                    << ", expected " << total_elements);
            host_int32_data_.resize(total_elements, 0);
        }
        else
        {
            host_int32_data_.resize(total_elements);
            quantizeFP32ToINT32(fp32_data.data(), host_int32_data_.data(), scale, total_elements);
        }
    }

    // =============================================================================
    // DEVICE MANAGEMENT
    // =============================================================================

    bool INT32Tensor::set_device(int device_idx)
    {
        if (device_idx == -1)
        {
            return true; // CPU always supported
        }
        LOG_WARN("[INT32Tensor] Device upload not yet implemented");
        return false;
    }

    bool INT32Tensor::sync_to_device()
    {
        LOG_WARN("[INT32Tensor] sync_to_device not implemented");
        return false;
    }

    bool INT32Tensor::sync_from_device()
    {
        LOG_WARN("[INT32Tensor] sync_from_device not implemented");
        return false;
    }

    // =============================================================================
    // DATA ACCESS (TensorBase interface)
    // =============================================================================

    const float *INT32Tensor::data() const
    {
        // Dequantize to cache on demand
        size_t total_elements = 1;
        for (auto dim : shape_)
        {
            total_elements *= dim;
        }

        if (dequant_cache_.size() != total_elements)
        {
            dequant_cache_.resize(total_elements);
            dequantizeINT32ToFP32(host_int32_data_.data(), dequant_cache_.data(),
                                  scale_, total_elements);
        }

        return dequant_cache_.data();
    }

    float *INT32Tensor::mutable_data()
    {
        LOG_ERROR("[INT32Tensor] mutable_data() not supported (read-only tensor)");
        return nullptr;
    }

    bool INT32Tensor::copyFrom(const TensorBase *src)
    {
        if (!src)
        {
            LOG_ERROR("[INT32Tensor] copyFrom: null source");
            return false;
        }

        // Check shape compatibility
        if (src->shape() != shape_)
        {
            LOG_ERROR("[INT32Tensor] copyFrom: shape mismatch");
            return false;
        }

        // Copy from FP32 source
        if (src->native_type() == TensorType::FP32)
        {
            const float *src_data = src->data();
            size_t count = 1;
            for (auto dim : shape_)
            {
                count *= dim;
            }
            quantizeFP32ToINT32(src_data, host_int32_data_.data(), scale_, count);
            dequant_cache_.clear(); // Invalidate cache
            return true;
        }

        // Copy from INT32 source
        if (src->native_type() == TensorType::INT32)
        {
            const INT32Tensor *src_int32 = dynamic_cast<const INT32Tensor *>(src);
            if (src_int32)
            {
                host_int32_data_ = src_int32->host_int32_data_;
                scale_ = src_int32->scale_;
                row_scales_ = src_int32->row_scales_;
                dequant_cache_.clear();
                return true;
            }
        }

        LOG_ERROR("[INT32Tensor] copyFrom: unsupported source type");
        return false;
    }

    // =============================================================================
    // KERNEL CREATION
    // =============================================================================

    std::unique_ptr<ITensorGemm> INT32Tensor::createGemm()
    {
        LOG_ERROR("[INT32Tensor] GEMM not supported (INT32 is output-only from INT8 GEMM)");
        return nullptr;
    }

    std::unique_ptr<ITensorRoPE> INT32Tensor::createRoPE()
    {
        LOG_ERROR("[INT32Tensor] RoPE not supported for INT32");
        return nullptr;
    }

    std::unique_ptr<ITensorSwiGLU> INT32Tensor::createSwiGLU()
    {
        LOG_ERROR("[INT32Tensor] SwiGLU not supported for INT32");
        return nullptr;
    }

    std::unique_ptr<ITensorSoftmax> INT32Tensor::createSoftmax()
    {
        LOG_ERROR("[INT32Tensor] Softmax not supported for INT32");
        return nullptr;
    }

    std::unique_ptr<ITensorRMSNorm> INT32Tensor::createRMSNorm()
    {
        // INT32 tensors use INT32→INT8 RMSNorm kernel with requantization
        return std::make_unique<CPURMSNormKernel>();
    }

    std::unique_ptr<ITensorAttention> INT32Tensor::createAttention()
    {
        LOG_ERROR("[INT32Tensor] Attention not supported for INT32");
        return nullptr;
    }

    // =============================================================================
    // FORMAT CONVERSION
    // =============================================================================

    void INT32Tensor::to_fp32(float *dst) const
    {
        size_t count = 1;
        for (auto dim : shape_)
        {
            count *= dim;
        }
        dequantizeINT32ToFP32(host_int32_data_.data(), dst, scale_, count);
    }

    bool INT32Tensor::from_int32_with_scales(
        const int32_t *accum,
        int rows,
        int cols,
        const float *row_scales,
        const float *col_scales,
        const float *bias)
    {
        (void)col_scales;
        (void)bias;

        if (!accum)
        {
            LOG_ERROR("[INT32Tensor::from_int32_with_scales] accum buffer is null");
            return false;
        }

        if (shape_.size() != 2)
        {
            LOG_ERROR("[INT32Tensor::from_int32_with_scales] tensor must be 2D, got " << shape_.size() << "D");
            return false;
        }
        if (static_cast<int>(shape_[0]) != rows || static_cast<int>(shape_[1]) != cols)
        {
            LOG_ERROR("[INT32Tensor::from_int32_with_scales] shape mismatch: tensor=[" << shape_[0]
                                                                                       << ", " << shape_[1] << "] input=[" << rows << ", " << cols << "]");
            return false;
        }

        const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        std::memcpy(host_int32_data_.data(), accum, total * sizeof(int32_t));

        if (row_scales)
        {
            row_scales_.assign(row_scales, row_scales + rows);
        }
        else
        {
            row_scales_.clear();
        }

        dequant_cache_.clear();
        return true;
    }

    void INT32Tensor::to_bf16(uint16_t *dst) const
    {
        LOG_ERROR("[INT32Tensor] to_bf16 not yet implemented");
    }

    void INT32Tensor::to_fp16(uint16_t *dst) const
    {
        LOG_ERROR("[INT32Tensor] to_fp16 not yet implemented");
    }

    void INT32Tensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
    {
        LOG_ERROR("[INT32Tensor] to_int8_blocked not yet implemented");
    }

    bool INT32Tensor::to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales) const
    {
        // Use requantize_to_int8 for per-row quantization
        if (!dst_row_scales)
        {
            LOG_ERROR("[INT32Tensor] to_int8_perchannel requires dst_row_scales");
            return false;
        }

        if (shape_.size() != 2)
        {
            LOG_ERROR("[INT32Tensor] to_int8_perchannel requires 2D tensor");
            return false;
        }

        return requantize_to_int8(dst_int8, dst_row_scales);
    }

    void INT32Tensor::to_fp32_row(size_t row_idx, float *buffer) const
    {
        if (shape_.size() != 2)
        {
            LOG_ERROR("[INT32Tensor] to_fp32_row requires 2D tensor");
            return;
        }

        size_t cols = shape_[1];
        const int32_t *row = host_int32_data_.data() + row_idx * cols;
        dequantizeINT32ToFP32(row, buffer, scale_, cols);
    }

    void INT32Tensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
    {
        if (offset + count > host_int32_data_.size())
        {
            LOG_ERROR("[INT32Tensor] to_fp32_span: out of range");
            return;
        }

        dequantizeINT32ToFP32(host_int32_data_.data() + offset, buffer, scale_, count);
    }

    ActivationPack INT32Tensor::to_int8_activation_pack(int rows, int cols) const
    {
        return pack_activation_rows_to_int8(rows, cols);
    }

    std::shared_ptr<TensorBase> INT32Tensor::create_view(
        const std::vector<size_t> &new_shape,
        size_t offset)
    {
        LOG_ERROR("[INT32Tensor] create_view not yet implemented");
        return nullptr;
    }

    // =============================================================================
    // INT32→INT8 REQUANTIZATION (Core function for full INT8 pipeline)
    // =============================================================================

    bool INT32Tensor::requantize_to_int8(int8_t *dst_int8, float *dst_row_scales) const
    {
        if (shape_.size() != 2)
        {
            LOG_ERROR("[INT32Tensor] requantize_to_int8 requires 2D tensor [m, n]");
            return false;
        }

        size_t m = shape_[0];
        size_t n = shape_[1];

        requantizeINT32ToINT8_PerRow(
            host_int32_data_.data(),
            dst_int8,
            dst_row_scales,
            m, n);

        return true;
    }

    bool INT32Tensor::applyRMSNorm(
        const float *gamma,
        int seq_len,
        int d_model,
        float eps,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        // NOTE: INT32→INT8 RMSNorm path requires an INT8 output tensor.
        // The current signature doesn't support this (in-place operation).
        //
        // For now, log an error and return false. Future work:
        // 1. Add applyRMSNorm() variant that accepts output tensor pointer
        // 2. Create INT8Tensor output in pipeline before calling
        // 3. Use kernel->apply_int32_to_int8() with proper output buffer
        //
        // Alternative: Convert to FP32, normalize, convert back (not ideal for full INT8 path)

        LOG_ERROR("[INT32Tensor::applyRMSNorm] INT32→INT8 RMSNorm not yet supported");
        LOG_ERROR("  Requires INT8 output tensor - signature needs extension");
        LOG_ERROR("  For now, convert to FP32 for normalization or add INT8 output parameter");
        return false;
    }

    bool INT32Tensor::applyRoPE(
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
        // RoPE is an activation operation, not applicable to quantized INT32 accumulators
        LOG_ERROR("[INT32Tensor::applyRoPE] RoPE not supported on INT32 accumulator tensors");
        LOG_ERROR("  RoPE is an activation operation - convert to FP32 first");
        return false;
    }

} // namespace llaminar2
