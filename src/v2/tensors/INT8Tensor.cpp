/**
 * @file INT8Tensor.cpp
 * @brief INT8 tensor implementation
 * @author David Sanftenberg
 * @date 2025-11-05
 */

#include "Tensors.h"
#include "../kernels/cpu/gemm_v4/OneDNNGemmKernel.h"
#include "../kernels/cpu/fused/FusedRMSNormQuantize.h"
#include "../utils/Logger.h"
// #include "../kernels/cpu/gemm/int8/INT8PackedGemm.h"  // DEPRECATED: Now using IntegerGemm via createGemm()
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>

namespace llaminar2
{
    // =============================================================================
    // QUANTIZATION HELPERS (Forward declarations)
    // =============================================================================

    /**
     * @brief Quantize FP32 data to INT8 with scale factor
     *
     * Computes:
     *   scale = max_abs / 127.0
     *   int8_val = round(fp32_val / scale)
     *
     * @param fp32_data Input FP32 data
     * @param int8_data Output INT8 data
     * @param scale Output scale factor
     * @param count Number of elements
     */
    static void quantizeFP32ToINT8(const float *fp32_data, int8_t *int8_data,
                                   float &scale, size_t count);

    /**
     * @brief Quantize FP32 2D matrix to INT8 with per-column and per-row scales
     *
     * For weight matrices [rows, cols], computes:
     * - Per-column scales: Used for normal (transpose_B=false) operations
     * - Per-row scales: Used for transposed (transpose_B=true) operations
     *
     * This ensures transpose operations maintain per-channel quantization accuracy.
     *
     * @param fp32_data Input FP32 data (row-major)
     * @param int8_data Output INT8 data (row-major)
     * @param col_scales Output per-column scales [cols]
     * @param row_scales Output per-row scales [rows]
     * @param rows Number of rows
     * @param cols Number of columns
     */
    static void quantizeFP32ToINT8_PerColumn(const float *fp32_data, int8_t *int8_data,
                                             std::vector<float> &col_scales,
                                             std::vector<float> &row_scales,
                                             size_t rows, size_t cols);

    INT8Tensor::INT8Tensor(const std::vector<size_t> &shape)
        : shape_(shape), device_idx_(-1), scale_(1.0f), device_data_(nullptr)
    {
        size_t total = 1;
        for (auto dim : shape)
            total *= dim;
        host_int8_data_.resize(total, 0);
    }

    INT8Tensor::INT8Tensor(const std::vector<size_t> &shape,
                           const std::vector<int8_t> &data,
                           float scale)
        : shape_(shape), device_idx_(-1), scale_(scale), device_data_(nullptr)
    {
        size_t total = 1;
        for (auto dim : shape)
            total *= dim;

        if (data.size() != total)
        {
            LOG_ERROR("[INT8Tensor] Data size mismatch: got " << data.size()
                                                              << ", expected " << total);
            host_int8_data_.resize(total, 0);
        }
        else
        {
            host_int8_data_.resize(data.size());
            std::copy(data.begin(), data.end(), host_int8_data_.begin());
        }
    }

    INT8Tensor::INT8Tensor(const std::vector<size_t> &shape,
                           const std::vector<float> &fp32_data)
        : shape_(shape), device_idx_(-1), device_data_(nullptr)
    {
        size_t total = 1;
        for (auto dim : shape)
            total *= dim;

        if (fp32_data.size() != total)
        {
            LOG_ERROR("[INT8Tensor] FP32 data size mismatch: got " << fp32_data.size()
                                                                   << ", expected " << total);
            host_int8_data_.resize(total, 0);
            scale_ = 1.0f;
            return;
        }

        host_int8_data_.resize(total);

        // For 2D tensors (weight matrices), use per-column quantization
        // Also compute per-row scales for transpose operations
        if (shape_.size() == 2)
        {
            quantizeFP32ToINT8_PerColumn(fp32_data.data(), host_int8_data_.data(),
                                         col_scales_, row_scales_cache_, shape_[0], shape_[1]);
            // Set global scale to max of column scales for backward compatibility
            scale_ = col_scales_.empty() ? 1.0f : *std::max_element(col_scales_.begin(), col_scales_.end());
        }
        else
        {
            // Non-2D tensors use global quantization
            quantizeFP32ToINT8(fp32_data.data(), host_int8_data_.data(), scale_, total);
        }
    }

    bool INT8Tensor::set_device(int device_idx)
    {
        // TODO: Implement device upload for INT8
        LOG_WARN("[INT8Tensor] Device upload not yet implemented");
        device_idx_ = device_idx;
        return false;
    }

    const float *INT8Tensor::data() const
    {
        // Dequantize to cache on demand
        size_t total = 1;
        for (auto dim : shape_)
            total *= dim;

        if (dequant_cache_.size() != total)
        {
            dequant_cache_.resize(total);
        }

        // Dequantize with per-row scales if available, otherwise use global scale
        if (!row_scales_cache_.empty())
        {
            // Per-row quantization (used by FusedRMSNormQuantize)
            // CRITICAL: Only dequantize rows for which we have scales (effective_seq_len)
            // Buffer may be larger (effective_max), but only first N rows contain valid data
            const size_t rows_to_dequantize = row_scales_cache_.size();
            const size_t cols = shape_.size() > 1 ? shape_[1] : 1;

            // Log warning if shape doesn't match scales (expected for batched execution)
            if (rows_to_dequantize != shape_[0])
            {
                LOG_DEBUG("[INT8Tensor::data] Dequantizing " << rows_to_dequantize
                                                             << " rows (have scales) out of " << shape_[0] << " total rows (buffer size)");
            }

            for (size_t r = 0; r < rows_to_dequantize; ++r)
            {
                const float row_scale = row_scales_cache_[r];
                for (size_t c = 0; c < cols; ++c)
                {
                    const size_t idx = r * cols + c;
                    dequant_cache_[idx] = static_cast<float>(host_int8_data_[idx]) * row_scale;
                }
            }

            // Zero out remaining rows (if buffer is larger than data)
            const size_t total_rows = shape_[0];
            if (rows_to_dequantize < total_rows)
            {
                for (size_t r = rows_to_dequantize; r < total_rows; ++r)
                {
                    for (size_t c = 0; c < cols; ++c)
                    {
                        const size_t idx = r * cols + c;
                        dequant_cache_[idx] = 0.0f;
                    }
                }
            }
        }
        else
        {
            // Global scale (default path for weight tensors)
            for (size_t i = 0; i < total; ++i)
            {
                dequant_cache_[i] = static_cast<float>(host_int8_data_[i]) * scale_;
            }
        }

        return dequant_cache_.data();
    }

    float *INT8Tensor::mutable_data()
    {
        // Invalidate dequant cache since we're allowing mutation
        dequant_cache_.clear();

        // Return pointer to INT8 buffer (reinterpreted as float* for interface compatibility)
        // Callers must reinterpret_cast back to int8_t* for actual INT8 writes
        return reinterpret_cast<float *>(host_int8_data_.data());
    }

    bool INT8Tensor::copyFrom(const TensorBase *src)
    {
        // TODO: Implement cross-tensor copy
        LOG_ERROR("[INT8Tensor] copyFrom not yet implemented");
        return false;
    }

    std::unique_ptr<ITensorGemm> INT8Tensor::createGemm()
    {
        // CPU: Use new IntegerGemm system (replaces deprecated INT8PackedGemm)
        // This tensor serves as the weight tensor (B parameter) in INT8×INT8→INT8 operations
        LOG_ERROR("[INT8Tensor] createGemm() not yet implemented for new IntegerGemm system");
        return nullptr; // TODO: Integrate with IntegerGemmAdapter
    }

    std::unique_ptr<ITensorRMSNorm> INT8Tensor::createRMSNorm()
    {
        // INT8 tensors can't be normalized in-place (would lose quantization)
        // Return FusedRMSNormQuantize kernel for INT8 activation quantization
        return std::make_unique<FusedRMSNormQuantize>();
    }

    std::unique_ptr<ITensorRoPE> INT8Tensor::createRoPE()
    {
        // RoPE not applicable to INT8 activations (needs FP32 for trigonometric ops)
        LOG_ERROR("[INT8Tensor] RoPE not supported on INT8 activations");
        return nullptr;
    }

    std::unique_ptr<ITensorAttention> INT8Tensor::createAttention()
    {
        // Attention operates on dequantized FP32 data
        LOG_ERROR("[INT8Tensor] Attention not supported on INT8 activations");
        return nullptr;
    }

    std::unique_ptr<ITensorSwiGLU> INT8Tensor::createSwiGLU()
    {
        // SwiGLU operates on dequantized FP32 data
        LOG_ERROR("[INT8Tensor] SwiGLU not supported on INT8 activations");
        return nullptr;
    }

    std::unique_ptr<ITensorSoftmax> INT8Tensor::createSoftmax()
    {
        // Softmax operates on dequantized FP32 data
        LOG_ERROR("[INT8Tensor] Softmax not supported on INT8 activations");
        return nullptr;
    }

    ActivationPack INT8Tensor::to_int8_activation_pack(int rows, int cols) const
    {
        // INT8Tensor is already quantized - copy existing data to pack
        ActivationPack pack;
        pack.data.assign(host_int8_data_.begin(), host_int8_data_.end()); // Copy INT8 data
        pack.row_scales.assign(rows, scale_);                             // Use per-tensor scale for all rows
        pack.rows = rows;
        pack.cols = cols;
        return pack;
    }

    bool INT8Tensor::applyRoPE(
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
        // RoPE requires floating-point operations - not supported on INT8
        LOG_ERROR("[INT8Tensor] applyRoPE not supported on INT8 activations");
        return false;
    }

    bool INT8Tensor::from_int32_with_scales(
        const int32_t *accum,
        int rows,
        int cols,
        const float *row_scales,
        const float *col_scales,
        const float *bias)
    {
        if (!accum)
        {
            LOG_ERROR("[INT8Tensor::from_int32_with_scales] accum buffer is null");
            return false;
        }

        if (shape_.size() != 2)
        {
            LOG_ERROR("[INT8Tensor::from_int32_with_scales] tensor must be 2D, got " << shape_.size() << "D");
            return false;
        }
        if (static_cast<int>(shape_[0]) != rows || static_cast<int>(shape_[1]) != cols)
        {
            LOG_ERROR("[INT8Tensor::from_int32_with_scales] shape mismatch: tensor=[" << shape_[0]
                                                                                      << ", " << shape_[1] << "] input=[" << rows << ", " << cols << "]");
            return false;
        }

        // For INT8 output, requantize from INT32 accumulator
        // The INT32 accumulator contains values in the range:
        //   accum[i,j] = sum_k(A_int8[i,k] * B_int8[k,j])
        // We need to scale this back to INT8 range: accum * row_scales[i] * col_scales[j]
        // Then requantize to INT8 with a new per-tensor scale

        // First pass: find max absolute value for global quantization
        float max_abs = 0.0f;
        for (int i = 0; i < rows; ++i)
        {
            const float row_scale = row_scales ? row_scales[i] : 1.0f;
            for (int j = 0; j < cols; ++j)
            {
                const float col_scale = col_scales ? col_scales[j] : 1.0f;
                const float bias_val = bias ? bias[j] : 0.0f;
                float val = static_cast<float>(accum[i * cols + j]) * row_scale * col_scale + bias_val;
                max_abs = std::max(max_abs, std::abs(val));
            }
        }

        // Compute quantization scale
        scale_ = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;

        // Second pass: quantize to INT8
        for (int i = 0; i < rows; ++i)
        {
            const float row_scale = row_scales ? row_scales[i] : 1.0f;
            for (int j = 0; j < cols; ++j)
            {
                const float col_scale = col_scales ? col_scales[j] : 1.0f;
                const float bias_val = bias ? bias[j] : 0.0f;
                float val = static_cast<float>(accum[i * cols + j]) * row_scale * col_scale + bias_val;
                int8_t quantized = static_cast<int8_t>(std::round(val / scale_));
                host_int8_data_[i * cols + j] = quantized;
            }
        }

        return true;
    }

    void INT8Tensor::to_fp32(float *dst) const
    {
        size_t total = 1;
        for (auto dim : shape_)
            total *= dim;

        for (size_t i = 0; i < total; ++i)
        {
            dst[i] = static_cast<float>(host_int8_data_[i]) * scale_;
        }
    }

    void INT8Tensor::to_bf16(uint16_t *dst) const
    {
        // TODO: Implement INT8 → BF16 conversion
        LOG_ERROR("[INT8Tensor] to_bf16 not yet implemented");
    }

    void INT8Tensor::to_fp16(uint16_t *dst) const
    {
        // TODO: Implement INT8 → FP16 conversion
        LOG_ERROR("[INT8Tensor] to_fp16 not yet implemented");
    }

    void INT8Tensor::to_int8_blocked(int8_t *dst_int8, float *dst_scales, size_t block_size) const
    {
        // Already in INT8 format, just copy
        size_t total = 1;
        for (auto dim : shape_)
            total *= dim;

        std::memcpy(dst_int8, host_int8_data_.data(), total * sizeof(int8_t));

        // Store scale factors (one per block)
        size_t num_blocks = (total + block_size - 1) / block_size;
        for (size_t i = 0; i < num_blocks; ++i)
        {
            dst_scales[i] = scale_;
        }
    }

    bool INT8Tensor::to_int8_perchannel(int8_t *dst_int8, float *dst_col_scales, float *dst_row_scales) const
    {
        // Already in INT8 format with per-channel scales
        const auto &shp = shape();
        if (shp.size() != 2)
        {
            LOG_ERROR("[INT8Tensor] to_int8_perchannel() requires 2D tensor, got " << shp.size() << "D");
            return false;
        }

        const size_t rows = shp[0];
        const size_t cols = shp[1];

        // Copy INT8 data directly
        std::memcpy(dst_int8, host_int8_data_.data(), rows * cols * sizeof(int8_t));

        // Copy or generate per-column scales
        if (!col_scales_.empty())
        {
            std::memcpy(dst_col_scales, col_scales_.data(), cols * sizeof(float));
        }
        else
        {
            // Use global scale for all columns
            for (size_t j = 0; j < cols; ++j)
            {
                dst_col_scales[j] = scale_;
            }
        }

        // Copy or generate per-row scales
        if (dst_row_scales != nullptr)
        {
            if (!row_scales_cache_.empty())
            {
                std::memcpy(dst_row_scales, row_scales_cache_.data(), rows * sizeof(float));
            }
            else
            {
                // Use global scale for all rows
                for (size_t i = 0; i < rows; ++i)
                {
                    dst_row_scales[i] = scale_;
                }
            }
        }

        return true;
    }

    void INT8Tensor::to_fp32_row(size_t row_idx, float *buffer) const
    {
        if (shape_.size() != 2)
        {
            LOG_ERROR("[INT8Tensor] to_fp32_row requires 2D tensor");
            return;
        }

        size_t cols = shape_[1];
        size_t offset = row_idx * cols;

        for (size_t i = 0; i < cols; ++i)
        {
            buffer[i] = static_cast<float>(host_int8_data_[offset + i]) * scale_;
        }
    }

    void INT8Tensor::to_fp32_span(size_t offset, size_t count, float *buffer) const
    {
        for (size_t i = 0; i < count; ++i)
        {
            buffer[i] = static_cast<float>(host_int8_data_[offset + i]) * scale_;
        }
    }

    std::shared_ptr<TensorBase> INT8Tensor::create_view(
        const std::vector<size_t> &new_shape,
        size_t offset)
    {
        // TODO: Implement INT8 view support
        LOG_ERROR("[INT8Tensor] View not yet implemented");
        return nullptr;
    }

    bool INT8Tensor::sync_to_device()
    {
        // TODO: Implement device sync
        return false;
    }

    bool INT8Tensor::sync_from_device()
    {
        // TODO: Implement device sync
        return false;
    }

    // =============================================================================
    // QUANTIZATION HELPERS
    // =============================================================================

    void quantizeFP32ToINT8(const float *fp32_data,
                            int8_t *int8_data,
                            float &scale,
                            size_t count)
    {
        if (count == 0)
        {
            scale = 1.0f;
            return;
        }

        // Find max absolute value for scale calculation
        float max_abs = 0.0f;
        for (size_t i = 0; i < count; ++i)
        {
            float abs_val = std::fabs(fp32_data[i]);
            if (abs_val > max_abs)
                max_abs = abs_val;
        }

        // Compute scale factor
        // int8 range: [-127, 127] (we avoid -128 for symmetry)
        if (max_abs == 0.0f)
        {
            scale = 1.0f;
            std::fill_n(int8_data, count, 0);
            return;
        }

        scale = max_abs / 127.0f;
        float inv_scale = 1.0f / scale;

        // Quantize with rounding
        for (size_t i = 0; i < count; ++i)
        {
            float scaled = fp32_data[i] * inv_scale;
            int32_t quantized = static_cast<int32_t>(std::round(scaled));

            // Clamp to int8 range
            if (quantized > 127)
                quantized = 127;
            else if (quantized < -127)
                quantized = -127;

            int8_data[i] = static_cast<int8_t>(quantized);
        }
    }

    void quantizeFP32ToINT8_PerColumn(const float *fp32_data,
                                      int8_t *int8_data,
                                      std::vector<float> &col_scales,
                                      std::vector<float> &row_scales,
                                      size_t rows,
                                      size_t cols)
    {
        if (rows == 0 || cols == 0)
        {
            col_scales.clear();
            row_scales.clear();
            return;
        }

        col_scales.resize(cols);
        row_scales.resize(rows);

        // Compute per-column scales
        for (size_t j = 0; j < cols; ++j)
        {
            float max_abs = 0.0f;
            for (size_t i = 0; i < rows; ++i)
            {
                float abs_val = std::fabs(fp32_data[i * cols + j]);
                if (abs_val > max_abs)
                    max_abs = abs_val;
            }

            col_scales[j] = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        }

        // Compute per-row scales (for transpose operations)
        for (size_t i = 0; i < rows; ++i)
        {
            float max_abs = 0.0f;
            for (size_t j = 0; j < cols; ++j)
            {
                float abs_val = std::fabs(fp32_data[i * cols + j]);
                if (abs_val > max_abs)
                    max_abs = abs_val;
            }

            row_scales[i] = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        }

        // Quantize each element using its column's scale
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                const size_t idx = i * cols + j;
                const float inv_scale = 1.0f / col_scales[j];
                float scaled = fp32_data[idx] * inv_scale;
                int32_t quantized = static_cast<int32_t>(std::round(scaled));

                // Clamp to int8 range
                if (quantized > 127)
                    quantized = 127;
                else if (quantized < -127)
                    quantized = -127;

                int8_data[idx] = static_cast<int8_t>(quantized);
            }
        }
    }

    /**
     * @brief Get per-row scales (computed during quantization)
     *
     * For transpose operations (transpose_B=true), we need per-row scales instead of per-column.
     * These are computed from the original FP32 data during quantization and cached.
     *
     * @return Reference to cached per-row scales
     */
    const std::vector<float> &INT8Tensor::get_row_scales() const
    {
        if (row_scales_cache_.empty())
        {
            LOG_WARN("[INT8Tensor] get_row_scales() called but row scales not computed during quantization");
        }
        return row_scales_cache_;
    }

    // ============================================================================
    // ITensorGemmTileDataProvider Interface - For INT8 GEMM Kernels
    // ============================================================================

    /**
     * @brief Decode a row of INT8 data to FP32 for GEMM computation
     *
     * Used by INT8PackedGemm to dequantize weight matrix rows during GEMM.
     * Applies appropriate scaling: per-row (for transpose), per-column, or global.
     *
     * @param row_idx Row index in the tensor
     * @param k_block_offset Block offset within row (currently unused - full row decoded)
     * @param output Output buffer for dequantized FP32 data [cols elements]
     */
    void INT8Tensor::decode_block_at(size_t row_idx, size_t k_block_offset, float *output) const
    {
        const size_t cols = shape_[1];
        const int8_t *int8_row = host_int8_data_.data() + row_idx * cols;

        // Use per-row scale if available (for transpose_B=true operations)
        if (!row_scales_cache_.empty())
        {
            const float row_scale = row_scales_cache_[row_idx];
            for (size_t i = 0; i < cols; ++i)
            {
                output[i] = static_cast<float>(int8_row[i]) * row_scale;
            }
        }
        // Use per-column scales if available (for normal operations)
        else if (!col_scales_.empty())
        {
            for (size_t i = 0; i < cols; ++i)
            {
                output[i] = static_cast<float>(int8_row[i]) * col_scales_[i];
            }
        }
        // Fallback to global scale
        else
        {
            for (size_t i = 0; i < cols; ++i)
            {
                output[i] = static_cast<float>(int8_row[i]) * scale_;
            }
        }
    }

    /**
     * @brief Get pointer to raw INT8 data for a specific row
     *
     * Returns pointer to the start of a row for direct INT8 operations.
     *
     * @param row_idx Row index in the tensor
     * @param k_block_offset Block offset within row (currently unused)
     * @return Const pointer to raw INT8 data
     */
    const void *
    INT8Tensor::get_raw_block_at(size_t row_idx, size_t k_block_offset) const
    {
        return host_int8_data_.data() + row_idx * shape_[1];
    }

} // namespace llaminar2
