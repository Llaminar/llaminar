/**
 * @file INT8SwiGLUKernel.cpp
 * @brief INT8 SwiGLU activation kernel implementation
 * @author David Sanftenberg
 * @date 2025-11-06
 */

#include "INT8SwiGLUKernel.h"
#include "../../utils/Logger.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace llaminar2
{

    INT8SwiGLUKernel::INT8SwiGLUKernel(int device_idx)
        : device_idx_(device_idx)
    {
    }

    bool INT8SwiGLUKernel::forward(
        const int8_t *gate_int8,
        const float *gate_row_scales,
        const int8_t *up_int8,
        const float *up_row_scales,
        int8_t *output_int8,
        float *output_row_scales,
        int batch,
        int seq_len,
        int d_ff)
    {
        // Validate device compatibility - CPU kernel only supports CPU devices
        if (device_idx_ >= 0)
        {
            LOG_ERROR("INT8SwiGLUKernel: CPU kernel does not support GPU devices (device_idx="
                      << device_idx_ << ")");
            return false;
        }

        // Validate pointers
        if (!gate_int8 || !gate_row_scales || !up_int8 || !up_row_scales ||
            !output_int8 || !output_row_scales)
        {
            LOG_ERROR("INT8SwiGLUKernel: null pointer(s) detected");
            return false;
        }

        // Validate dimensions
        if (batch <= 0 || seq_len <= 0 || d_ff <= 0)
        {
            LOG_ERROR("INT8SwiGLUKernel: invalid dimensions: batch=" << batch
                                                                     << " seq_len=" << seq_len
                                                                     << " d_ff=" << d_ff);
            return false;
        }

        size_t total_size = static_cast<size_t>(batch) * seq_len * d_ff;

        // Allocate temporary buffers
        gate_fp32_buffer_.resize(total_size);
        up_fp32_buffer_.resize(total_size);
        output_fp32_buffer_.resize(total_size);

        // Step 1: Dequantize gate and up projections
        for (int b = 0; b < batch; ++b)
        {
            for (int i = 0; i < seq_len; ++i)
            {
                int row_idx = b * seq_len + i;
                float gate_scale = gate_row_scales[row_idx];
                float up_scale = up_row_scales[row_idx];

                int row_offset = row_idx * d_ff;

                for (int d = 0; d < d_ff; ++d)
                {
                    int idx = row_offset + d;
                    gate_fp32_buffer_[idx] = static_cast<float>(gate_int8[idx]) * gate_scale;
                    up_fp32_buffer_[idx] = static_cast<float>(up_int8[idx]) * up_scale;
                }
            }
        }

        // Step 2: Compute SwiGLU = gate × SiLU(up)
        for (size_t idx = 0; idx < total_size; ++idx)
        {
            float gate_val = gate_fp32_buffer_[idx];
            float up_val = up_fp32_buffer_[idx];
            float silu_up = silu(up_val);
            output_fp32_buffer_[idx] = gate_val * silu_up;
        }

        // Step 3: Requantize output to INT8 (per-row quantization)
        for (int b = 0; b < batch; ++b)
        {
            for (int i = 0; i < seq_len; ++i)
            {
                int row_idx = b * seq_len + i;
                int row_offset = row_idx * d_ff;

                // Find max abs value in row
                float max_abs = 0.0f;
                for (int d = 0; d < d_ff; ++d)
                {
                    max_abs = std::max(max_abs, std::abs(output_fp32_buffer_[row_offset + d]));
                }

                // Compute scale for this row
                float scale = (max_abs > 1e-6f) ? (max_abs / 127.0f) : 1.0f;
                output_row_scales[row_idx] = scale;

                float inv_scale = 1.0f / scale;

                // Quantize row
                for (int d = 0; d < d_ff; ++d)
                {
                    int idx = row_offset + d;
                    float scaled = output_fp32_buffer_[idx] * inv_scale;
                    int quantized = static_cast<int>(std::round(scaled));
                    output_int8[idx] = static_cast<int8_t>(
                        std::clamp(quantized, -127, 127));
                }
            }
        }

        return true;
    }

    float INT8SwiGLUKernel::sigmoid(float x)
    {
        // Numerically stable sigmoid implementation
        if (x >= 0.0f)
        {
            // For positive x: sigmoid(x) = 1 / (1 + exp(-x))
            return 1.0f / (1.0f + std::exp(-x));
        }
        else
        {
            // For negative x: sigmoid(x) = exp(x) / (1 + exp(x))
            // This avoids exp(large_positive) which can overflow
            float exp_x = std::exp(x);
            return exp_x / (1.0f + exp_x);
        }
    }

    float INT8SwiGLUKernel::silu(float x)
    {
        // SiLU(x) = x × sigmoid(x)
        // Also known as Swish activation
        return x * sigmoid(x);
    }

} // namespace llaminar2
