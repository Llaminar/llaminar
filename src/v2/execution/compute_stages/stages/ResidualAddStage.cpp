/**
 * @file ResidualAddStage.cpp
 * @brief Implementation of ResidualAddStage
 */

#include "ResidualAddStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../tensors/SIMDHelpers.h"

namespace llaminar2
{

    // =============================================================================
    // ResidualAddStage Implementation (Type-Safe via TensorBase)
    // =============================================================================

    ResidualAddStage::ResidualAddStage(Params params) : params_(std::move(params)) {}

    bool ResidualAddStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[ResidualAddStage] Null device context");
            return false;
        }

        if (!params_.input || !params_.residual || !params_.output)
        {
            LOG_ERROR("[ResidualAddStage] Null tensor(s): input=" << params_.input
                                                                  << " residual=" << params_.residual
                                                                  << " output=" << params_.output);
            return false;
        }

        // Use explicit num_elements if provided, otherwise fall back to input->numel()
        // CRITICAL: For decode mode with pre-allocated buffers, params_.num_elements must be set
        // to avoid reading/writing beyond the actual sequence data.
        const size_t num_elements = params_.num_elements > 0 ? params_.num_elements : params_.input->numel();

        TensorType input_type = params_.input->native_type();
        TensorType residual_type = params_.residual->native_type();
        TensorType output_type = params_.output->native_type();

        LOG_DEBUG("[ResidualAddStage] Execute: num_elements=" << num_elements
                                                              << " input_type=" << params_.input->dtype_name()
                                                              << " residual_type=" << params_.residual->dtype_name()
                                                              << " output_type=" << params_.output->dtype_name());

        // Handle mixed-type case: FP32 input + Q8_1 residual → Q8_1 output
        // This occurs when fused attention outputs FP32 but activations are Q8_1
        if (input_type == TensorType::FP32 && residual_type == TensorType::Q8_1 && output_type == TensorType::Q8_1)
        {
            return executeFP32_Q8_1_to_Q8_1(ctx, num_elements);
        }

        // Handle HybridQ16 case: Q8_1 input + Q16_1 residual → Q16_1 output
        // This is THE key operation for typed residual connections
        if (input_type == TensorType::Q8_1 && residual_type == TensorType::Q16_1 && output_type == TensorType::Q16_1)
        {
            return executeQ8_1_Q16_1_to_Q16_1(ctx, num_elements);
        }

        // Handle HybridQ16 FFN residual: FP32 input + Q16_1 residual → Q16_1 output
        // This is used when the FFN down_proj outputs FP32 but the residual stream is Q16_1
        if (input_type == TensorType::FP32 && residual_type == TensorType::Q16_1 && output_type == TensorType::Q16_1)
        {
            return executeFP32_Q16_1_to_Q16_1(ctx, num_elements);
        }

        // Handle Q8_1 input + FP32 residual → FP32 output
        // This is used in HybridQ16 mode FFN residual where delta is Q8_1 but residual stream starts as FP32
        if (input_type == TensorType::Q8_1 && residual_type == TensorType::FP32 && output_type == TensorType::FP32)
        {
            return executeQ8_1_FP32_to_FP32(ctx, num_elements);
        }

        // Dispatch based on tensor type, passing num_elements through
        // Note: This assumes all three tensors have the same type
        switch (input_type)
        {
        case TensorType::FP32:
            return executeFP32(ctx, num_elements);
        case TensorType::BF16:
            return executeBF16(ctx, num_elements);
        case TensorType::FP16:
            return executeFP16(ctx, num_elements);
        case TensorType::Q8_1:
            return executeQ8_1(ctx, num_elements);
        default:
            LOG_ERROR("[ResidualAddStage] Unsupported tensor type: " << params_.input->dtype_name());
            return false;
        }
    }

    bool ResidualAddStage::executeFP32_Q8_1_to_Q8_1(IDeviceContext *ctx, size_t num_elements)
    {
        // Mixed-type case: FP32 input + Q8_1 residual → Q8_1 output
        // This is used when fused attention outputs FP32 (attn_proj) but the hidden state is Q8_1
        const auto *input_fp32 = dynamic_cast<const FP32Tensor *>(params_.input);
        const auto *residual_q8 = dynamic_cast<const Q8_1Tensor *>(params_.residual);
        auto *output_q8 = dynamic_cast<Q8_1Tensor *>(params_.output);

        if (!input_fp32 || !residual_q8 || !output_q8)
        {
            LOG_ERROR("[ResidualAddStage::FP32_Q8_1_to_Q8_1] Failed to cast tensors");
            return false;
        }

        const float *input_data = input_fp32->data();
        const float *residual_data = residual_q8->fp32_data(); // Explicit dequantization acknowledgment

        LOG_DEBUG("[ResidualAddStage::FP32_Q8_1_to_Q8_1] Adding " << num_elements << " elements");

        // Add in FP32 and quantize result back to Q8_1
        // Use block-wise approach for efficiency
        const size_t num_blocks = num_elements / 32;
        Q8_1Block *output_blocks = output_q8->mutable_typed_data();

        // Process block by block
        for (size_t b = 0; b < num_blocks; ++b)
        {
            alignas(32) float temp[32];
            const size_t offset = b * 32;

            // Add FP32 input + dequantized Q8_1 residual
            for (size_t i = 0; i < 32; ++i)
            {
                temp[i] = input_data[offset + i] + residual_data[offset + i];
            }

            // Quantize result block
            simd::quantize_single_block(temp, output_blocks[b], 32);
        }

        return true;
    }

    bool ResidualAddStage::executeFP32(IDeviceContext *ctx, size_t num_elements)
    {
        const float *input = params_.input->data();
        const float *residual = params_.residual->data();
        float *output = params_.output->mutable_data();
        const size_t n = num_elements;

        LOG_DEBUG("[ResidualAddStage::FP32] input[0:4]="
                  << input[0] << "," << input[1] << "," << input[2] << "," << input[3]
                  << " residual[0:4]="
                  << residual[0] << "," << residual[1] << "," << residual[2] << "," << residual[3]);

        ctx->runFor(0, n, [=](size_t i)
                    { output[i] = input[i] + residual[i]; });

        LOG_DEBUG("[ResidualAddStage::FP32] output[0:4]="
                  << output[0] << "," << output[1] << "," << output[2] << "," << output[3]);

        return true;
    }

    bool ResidualAddStage::executeBF16(IDeviceContext *ctx, size_t num_elements)
    {
        const uint16_t *input = static_cast<const uint16_t *>(params_.input->raw_data());
        const uint16_t *residual = static_cast<const uint16_t *>(params_.residual->raw_data());
        uint16_t *output = static_cast<uint16_t *>(params_.output->raw_mutable_data());
        const size_t n = num_elements;

        LOG_DEBUG("[ResidualAddStage::BF16] Converting and adding " << n << " elements");

        ctx->runFor(0, n, [=](size_t i)
                    {
            float in_f = simd::bf16_to_fp32(input[i]);
            float res_f = simd::bf16_to_fp32(residual[i]);
            output[i] = simd::fp32_to_bf16(in_f + res_f); });

        return true;
    }

    bool ResidualAddStage::executeFP16(IDeviceContext *ctx, size_t num_elements)
    {
        const uint16_t *input = static_cast<const uint16_t *>(params_.input->raw_data());
        const uint16_t *residual = static_cast<const uint16_t *>(params_.residual->raw_data());
        uint16_t *output = static_cast<uint16_t *>(params_.output->raw_mutable_data());
        const size_t n = num_elements;

        LOG_DEBUG("[ResidualAddStage::FP16] Converting and adding " << n << " elements");

        ctx->runFor(0, n, [=](size_t i)
                    {
            float in_f = simd::fp16_to_fp32(input[i]);
            float res_f = simd::fp16_to_fp32(residual[i]);
            output[i] = simd::fp32_to_fp16(in_f + res_f); });

        return true;
    }

    bool ResidualAddStage::executeQ8_1(IDeviceContext *ctx, size_t num_elements)
    {
        // Cast to Q8_1Tensor to access block storage
        const auto *input_q8 = dynamic_cast<const Q8_1Tensor *>(params_.input);
        const auto *residual_q8 = dynamic_cast<const Q8_1Tensor *>(params_.residual);
        auto *output_q8 = dynamic_cast<Q8_1Tensor *>(params_.output);

        if (!input_q8 || !residual_q8 || !output_q8)
        {
            LOG_ERROR("[ResidualAddStage::Q8_1] Failed to cast tensors to Q8_1Tensor");
            return false;
        }

        const size_t numel = num_elements;
        if (numel % 32 != 0)
        {
            LOG_ERROR("[ResidualAddStage::Q8_1] Element count " << numel << " is not a multiple of 32");
            return false;
        }

        const Q8_1Block *input_blocks = input_q8->typed_data();
        const Q8_1Block *residual_blocks = residual_q8->typed_data();
        Q8_1Block *output_blocks = output_q8->mutable_typed_data();

        LOG_DEBUG("[ResidualAddStage::Q8_1] Adding " << numel << " elements ("
                                                     << (numel / 32) << " blocks)");

        // Use the SIMD-optimized Q8_1 addition (AVX512/AVX2/scalar)
        simd::q8_1_add_q8_1(input_blocks, residual_blocks, output_blocks, numel);

        return true;
    }

    bool ResidualAddStage::executeQ8_1_Q16_1_to_Q16_1(IDeviceContext *ctx, size_t num_elements)
    {
        // HybridQ16 mode: Q8_1 delta + Q16_1 residual → Q16_1 output (in-place)
        // This is THE key operation for typed residual connections.
        //
        // Note: params_.input is the delta (Q8_1), params_.residual is the accumulator (Q16_1)
        // The output tensor should be the same as residual for in-place operation.
        // Q16_1 provides 266× better precision than Q8_1 for the residual stream.

        const auto *delta_q8 = dynamic_cast<const Q8_1Tensor *>(params_.input);
        auto *residual_q16 = dynamic_cast<Q16_1Tensor *>(params_.output);

        if (!delta_q8 || !residual_q16)
        {
            LOG_ERROR("[ResidualAddStage::Q8_1_Q16_1] Failed to cast tensors");
            return false;
        }

        if (num_elements % 32 != 0)
        {
            LOG_ERROR("[ResidualAddStage::Q8_1_Q16_1] Element count " << num_elements
                                                                      << " is not a multiple of 32");
            return false;
        }

        const Q8_1Block *delta_blocks = delta_q8->typed_data();
        Q16_1Block *residual_blocks = residual_q16->mutable_typed_data();

        LOG_DEBUG("[ResidualAddStage::Q8_1_Q16_1] Adding " << num_elements
                                                           << " elements (" << (num_elements / 32) << " blocks)");

        // Use SIMD-optimized Q16_1 += Q8_1 addition
        // This function adds Q8_1 delta to Q16_1 residual in-place
        simd::q16_1_add_q8_1(residual_blocks, delta_blocks, num_elements);

        return true;
    }

    bool ResidualAddStage::executeFP32_Q16_1_to_Q16_1(IDeviceContext *ctx, size_t num_elements)
    {
        // HybridQ16 FFN residual: FP32 delta + Q16_1 residual → Q16_1 output (in-place)
        // This is used when the FFN down_proj outputs FP32 (via GEMM) but the residual
        // stream is maintained in Q16_1 for better precision.
        //
        // Note: params_.input is the delta (FP32), params_.output is the Q16_1 residual.
        // The operation is in-place on the residual buffer.

        const auto *delta_fp32 = dynamic_cast<const FP32Tensor *>(params_.input);
        auto *residual_q16 = dynamic_cast<Q16_1Tensor *>(params_.output);

        if (!delta_fp32 || !residual_q16)
        {
            LOG_ERROR("[ResidualAddStage::FP32_Q16_1] Failed to cast tensors");
            return false;
        }

        if (num_elements % 32 != 0)
        {
            LOG_ERROR("[ResidualAddStage::FP32_Q16_1] Element count " << num_elements
                                                                      << " is not a multiple of 32");
            return false;
        }

        const float *delta_data = delta_fp32->data();
        Q16_1Block *residual_blocks = residual_q16->mutable_typed_data();

        LOG_DEBUG("[ResidualAddStage::FP32_Q16_1] Adding " << num_elements
                                                           << " elements (" << (num_elements / 32) << " blocks)");

        // Use SIMD-optimized Q16_1 += FP32 addition
        // This function adds FP32 delta to Q16_1 residual in-place
        simd::q16_1_add_fp32(residual_blocks, delta_data, num_elements);

        return true;
    }

    bool ResidualAddStage::executeQ8_1_FP32_to_FP32(IDeviceContext *ctx, size_t num_elements)
    {
        // Q8_1 delta + FP32 residual → FP32 output (in-place on residual)
        // This is used in HybridQ16 mode when the FFN down_proj outputs Q8_1
        // but the residual stream hasn't been converted to Q16_1 yet.
        //
        // Note: params_.input is the delta (Q8_1), params_.output is the FP32 residual.
        // The operation is in-place on the residual buffer.

        const auto *delta_q8_1 = dynamic_cast<const Q8_1Tensor *>(params_.input);
        auto *residual_fp32 = dynamic_cast<FP32Tensor *>(params_.output);

        if (!delta_q8_1 || !residual_fp32)
        {
            LOG_ERROR("[ResidualAddStage::Q8_1_FP32_to_FP32] Failed to cast tensors");
            return false;
        }

        if (num_elements % 32 != 0)
        {
            LOG_ERROR("[ResidualAddStage::Q8_1_FP32_to_FP32] Element count " << num_elements
                                                                             << " is not a multiple of 32");
            return false;
        }

        const Q8_1Block *delta_blocks = delta_q8_1->typed_data();
        float *residual_data = residual_fp32->mutable_data();

        LOG_DEBUG("[ResidualAddStage::Q8_1_FP32_to_FP32] Adding " << num_elements
                                                                  << " elements (" << (num_elements / 32) << " blocks)");

        // Use SIMD-optimized FP32 += Q8_1 addition (dequant-and-add in one pass)
        simd::fp32_add_q8_1(residual_data, delta_blocks, num_elements);

        return true;
    }

    size_t ResidualAddStage::estimatedFlops() const
    {
        if (!params_.input)
            return 0;

        // For BF16/FP16: convert (1) + add + convert (1) = ~3 ops/elem
        // For FP32: 1 add per element
        // For Q8_1: dequant (2) + add + requant (2) = ~5 ops/elem
        TensorType ttype = params_.input->native_type();
        if (ttype == TensorType::BF16 || ttype == TensorType::FP16)
        {
            return params_.input->numel() * 3;
        }
        if (ttype == TensorType::Q8_1)
        {
            return params_.input->numel() * 5;
        }
        return params_.input->numel();
    }

    size_t ResidualAddStage::estimatedMemoryBytes() const
    {
        if (!params_.input)
            return 0;

        // Memory depends on tensor type
        TensorType ttype = params_.input->native_type();

        // Q8_1: 1 byte per element + 4 bytes per 32-element block for scale+sum
        // ~1.125 bytes/element on average
        if (ttype == TensorType::Q8_1)
        {
            // Q8_1Block: 32 bytes qs + 2 bytes d + 2 bytes sum_qs = 36 bytes per 32 elements
            size_t num_blocks = (params_.input->numel() + 31) / 32;
            return 3 * num_blocks * sizeof(Q8_1Block); // input + residual + output
        }

        size_t bytes_per_element;
        switch (ttype)
        {
        case TensorType::BF16:
        case TensorType::FP16:
            bytes_per_element = 2;
            break;
        case TensorType::FP32:
        default:
            bytes_per_element = 4;
            break;
        }
        return 3 * params_.input->numel() * bytes_per_element; // input + residual + output
    }

    bool ResidualAddStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
        default:
            return false;
        }
    }

    StageDumpInfo ResidualAddStage::getDumpInfo() const
    {
        StageDumpInfo info;

        if (!params_.input || !params_.residual || !params_.output)
            return info;

        // Determine dtype string based on tensor type
        const char *dtype = params_.input->dtype_name();
        size_t elem_size = sizeof(float);
        TensorType ttype = params_.input->native_type();
        switch (ttype)
        {
        case TensorType::BF16:
        case TensorType::FP16:
            elem_size = 2;
            break;
        case TensorType::Q8_1:
            // Q8_1Block: 36 bytes per 32 elements = 1.125 bytes/elem
            // For dump purposes, use 1 (closest integer approximation)
            elem_size = 1;
            break;
        default:
            break;
        }

        // Use explicit num_elements if provided (for pre-allocated buffers)
        // Otherwise derive from tensor dimensions
        size_t actual_elements = (params_.num_elements > 0) ? params_.num_elements : params_.input->numel();
        int cols = static_cast<int>(params_.input->cols());
        int rows = (cols > 0) ? static_cast<int>(actual_elements / cols) : 1;

        // Input tensors - use safe FP32 accessor
        const float *input_data = getSafeFp32Data(params_.input);
        const float *residual_data = getSafeFp32Data(params_.residual);
        const float *output_data = getSafeFp32Data(params_.output);

        if (input_data)
        {
            info.inputs.push_back({"input", input_data,
                                   static_cast<size_t>(rows), static_cast<size_t>(cols),
                                   dtype, elem_size});
        }
        if (residual_data)
        {
            info.inputs.push_back({"residual", residual_data,
                                   static_cast<size_t>(rows), static_cast<size_t>(cols),
                                   dtype, elem_size});
        }

        // Output
        if (output_data)
        {
            info.outputs.push_back({"output", output_data,
                                    static_cast<size_t>(rows), static_cast<size_t>(cols),
                                    dtype, elem_size});
        }

        // Scalar params
        info.addScalarInt("num_elements", static_cast<int>(actual_elements));
        info.addScalarInt("rows", rows);
        info.addScalarInt("cols", cols);

        return info;
    }

    StageBufferRequirements ResidualAddStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.input || !params_.residual || !params_.output)
            return reqs; // Empty if tensors not set

        // Get dimensions from tensors
        const size_t rows = params_.input->rows();
        const size_t cols = params_.input->cols();

        // Convert tensor type to buffer tensor type
        BufferTensorType buf_type = toBufferTensorType(params_.input->native_type());

        // INPUT buffers (read-only)
        reqs.addInput("input", {rows, cols}, buf_type);
        reqs.addInput("residual", {rows, cols}, buf_type);

        // OUTPUT buffer (may alias residual for in-place operation)
        reqs.addOutput("output", {rows, cols}, buf_type);

        return reqs;
    }

} // namespace llaminar2
