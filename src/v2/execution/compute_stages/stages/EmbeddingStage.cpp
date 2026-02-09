/**
 * @file EmbeddingStage.cpp
 * @brief Implementation of EmbeddingStage
 */

#include "EmbeddingStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../kernels/KernelFactory.h"
#include <cstring>

namespace llaminar2
{

    // =============================================================================
    // EmbeddingStage Implementation
    // =============================================================================

    EmbeddingStage::EmbeddingStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    ITensorEmbedding *EmbeddingStage::getOrCreateKernel()
    {
        // Return cached kernel if available
        if (cached_kernel_)
        {
            return cached_kernel_.get();
        }

        // Create kernel using stage's device_id (NOT output tensor's device_ member)
        // This ensures GPU kernels are created when the stage is assigned to GPU,
        // even if the output tensor was allocated with a CPU device initially.
        auto *output_base = dynamic_cast<TensorBase *>(params_.output);
        if (!output_base)
        {
            LOG_ERROR("[EmbeddingStage::getOrCreateKernel] Output not TensorBase");
            return nullptr;
        }

        // Get device type from stage's device_id (set by GraphOrchestrator based on placement)
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(params_.device_id);

        // Dispatch based on output tensor type
        if (auto *fp32 = dynamic_cast<FP32Tensor *>(output_base))
        {
            cached_kernel_ = llaminar::v2::kernels::KernelFactory::createEmbedding(fp32, dev_type);
        }
        else if (auto *bf16 = dynamic_cast<BF16Tensor *>(output_base))
        {
            cached_kernel_ = llaminar::v2::kernels::KernelFactory::createEmbedding(bf16, dev_type);
        }
        else if (auto *fp16 = dynamic_cast<FP16Tensor *>(output_base))
        {
            cached_kernel_ = llaminar::v2::kernels::KernelFactory::createEmbedding(fp16, dev_type);
        }
        else if (auto *q8_1 = dynamic_cast<Q8_1Tensor *>(output_base))
        {
            cached_kernel_ = llaminar::v2::kernels::KernelFactory::createEmbedding(q8_1, dev_type);
        }
        else
        {
            LOG_ERROR("[EmbeddingStage::getOrCreateKernel] Unsupported output tensor type: "
                      << output_base->dtype_name());
            return nullptr;
        }

        LOG_DEBUG("[EmbeddingStage::getOrCreateKernel] Created "
                  << (dev_type == DeviceType::CUDA   ? "CUDA"
                      : dev_type == DeviceType::ROCm ? "ROCm"
                                                     : "CPU")
                  << " embedding kernel for " << output_base->dtype_name() << " output");

        return cached_kernel_.get();
    }

    IWorkspaceConsumer *EmbeddingStage::getKernelAsWorkspaceConsumer()
    {
        auto *kernel = getOrCreateKernel();
        if (!kernel)
        {
            return nullptr;
        }
        return dynamic_cast<IWorkspaceConsumer *>(kernel);
    }

    bool EmbeddingStage::execute(IDeviceContext *ctx)
    {
        LOG_DEBUG("[EmbeddingStage] Execute: num_tokens=" << params_.num_tokens
                                                          << " d_model=" << params_.d_model
                                                          << " vocab_size=" << params_.vocab_size);

        // Validate inputs
        if (!params_.embed_table || !params_.output)
        {
            LOG_ERROR("[EmbeddingStage] Null tensor pointers");
            return false;
        }

        if (!params_.token_ids && !params_.token_batches)
        {
            LOG_ERROR("[EmbeddingStage] No token input provided");
            return false;
        }

        if (params_.num_tokens <= 0 || params_.d_model <= 0)
        {
            LOG_ERROR("[EmbeddingStage] Invalid dimensions");
            return false;
        }

        // Check if output is Q16_1 - needs special handling since there's no native kernel
        if (params_.output->native_type() == TensorType::Q16_1)
        {
            return executeQ16_1Output();
        }

        // Use the same approach as EmbeddingOp in Qwen2Pipeline:
        // Get the embedding kernel from the output tensor (IActivationTensor::createEmbedding)
        // This handles automatic type dispatch for FP32, Q8_1, etc.

        // Cast ITensor* to TensorBase* for CPU operations
        auto *embed_table_base = requireTensorBase(params_.embed_table, "embed_table");
        auto *output_base = requireTensorBase(params_.output, "output");
        if (!embed_table_base || !output_base)
        {
            LOG_ERROR("[EmbeddingStage] GPU tensors not yet supported");
            return false;
        }

        auto *activation = dynamic_cast<IActivationTensor *>(output_base);
        if (!activation)
        {
            LOG_ERROR("[EmbeddingStage] Output tensor must implement IActivationTensor");
            return false;
        }

        // Use cached kernel (enables workspace binding for GPU kernels)
        auto *kernel = getOrCreateKernel();
        if (!kernel)
        {
            LOG_ERROR("[EmbeddingStage] Failed to create embedding kernel");
            return false;
        }

        // Thread GPU stream for graph capture
        kernel->setGPUStream(gpuStream());

        // Handle batched vs single sequence input
        if (params_.token_batches)
        {
            // Batched input: flatten token batches with padding
            const int batch_size = static_cast<int>(params_.token_batches->size());
            const int padded_len = params_.padded_seq_len;
            const int total_positions = batch_size * padded_len;

            std::vector<int> flat_token_ids(total_positions, 0); // Zero-initialized for padding

            int global_idx = 0;
            for (int b = 0; b < batch_size; ++b)
            {
                const auto &tokens = (*params_.token_batches)[b];
                const int seq_len = static_cast<int>(tokens.size());

                // Copy actual tokens
                for (int i = 0; i < seq_len && i < padded_len; ++i)
                {
                    flat_token_ids[global_idx + i] = tokens[i];
                }
                global_idx += padded_len;
            }

            // Delegate to kernel's apply_tensor - handles all type dispatch internally
            if (!kernel->apply_tensor(embed_table_base, flat_token_ids.data(), total_positions,
                                      params_.d_model, output_base, params_.mpi_ctx, params_.device_id.toKernelDeviceIndex()))
            {
                LOG_ERROR("[EmbeddingStage] Kernel apply_tensor failed (batched)");
                return false;
            }

            // Zero out padding positions in output
            // This matches what EmbeddingOp does in its execute_batched
            global_idx = 0;
            for (int b = 0; b < batch_size; ++b)
            {
                const auto &tokens = (*params_.token_batches)[b];
                const int seq_len = static_cast<int>(tokens.size());

                for (int i = seq_len; i < padded_len; ++i)
                {
                    zero_output_row(params_.output, global_idx + i, params_.d_model);
                }
                global_idx += padded_len;
            }
        }
        else
        {
            // Single sequence input: delegate directly to kernel
            if (!kernel->apply_tensor(embed_table_base, params_.token_ids, params_.num_tokens,
                                      params_.d_model, output_base, params_.mpi_ctx, params_.device_id.toKernelDeviceIndex()))
            {
                LOG_ERROR("[EmbeddingStage] Kernel apply_tensor failed");
                return false;
            }
        }

        // Mark output as device-dirty WITH EVENT immediately after GPU kernel launch.
        // This ensures any subsequent read (e.g., DEBUG logging below) will properly
        // sync via event wait rather than reading stale data. The GraphExecutor's
        // markOutputsDirty() call is too late for reads inside execute().
        if (params_.device_id.type != DeviceType::CPU)
        {
            auto *output_base_tb = dynamic_cast<TensorBase *>(params_.output);
            if (output_base_tb)
            {
                output_base_tb->mark_device_dirty_with_event();
            }
        }

        // DEBUG: Log embedding output for parity debugging (guard expensive fp32_data() call)
        if (Logger::getInstance().shouldLog(LogLevel::VERBOSITY_DEBUG))
        {
            const float *out_data = params_.output->fp32_data();
            if (out_data && params_.num_tokens > 0)
            {
                LOG_DEBUG("[EmbeddingStage] output[0:8]=" << std::setprecision(6)
                                                          << out_data[0] << "," << out_data[1] << "," << out_data[2] << "," << out_data[3] << ","
                                                          << out_data[4] << "," << out_data[5] << "," << out_data[6] << "," << out_data[7]
                                                          << " device_id=" << params_.device_id.to_string());
            }
        }

        return true;
    }

    bool EmbeddingStage::executeQ16_1Output()
    {
        // Q16_1 output: Do FP32 embedding lookup, then quantize to Q16_1
        // This is used for HybridQ16 mode where the residual stream is Q16_1
        LOG_DEBUG("[EmbeddingStage] Q16_1 output path: FP32 lookup + quantization");

        auto *q16_output = dynamic_cast<Q16_1Tensor *>(params_.output);
        if (!q16_output)
        {
            LOG_ERROR("[EmbeddingStage] Failed to cast output to Q16_1Tensor");
            return false;
        }

        // Get embedding table data (always FP32)
        const float *embed_data = params_.embed_table->data();
        if (!embed_data)
        {
            LOG_ERROR("[EmbeddingStage] Null embedding table data");
            return false;
        }

        const int num_tokens = params_.token_ids ? params_.num_tokens
                                                 : static_cast<int>(params_.token_batches->size() * params_.padded_seq_len);
        const int d_model = params_.d_model;

        // Allocate temporary FP32 buffer for embedding lookup
        std::vector<float> fp32_buffer(static_cast<size_t>(num_tokens) * d_model);

        // Handle batched vs single sequence input
        if (params_.token_batches)
        {
            // Batched input: flatten token batches with padding
            const int batch_size = static_cast<int>(params_.token_batches->size());
            const int padded_len = params_.padded_seq_len;

            int global_idx = 0;
            for (int b = 0; b < batch_size; ++b)
            {
                const auto &tokens = (*params_.token_batches)[b];
                const int seq_len = static_cast<int>(tokens.size());

                // Copy actual embeddings
                for (int i = 0; i < seq_len && i < padded_len; ++i)
                {
                    const int token_id = tokens[i];
                    std::memcpy(fp32_buffer.data() + (global_idx + i) * d_model,
                                embed_data + token_id * d_model,
                                d_model * sizeof(float));
                }
                // Zero out padding positions
                for (int i = seq_len; i < padded_len; ++i)
                {
                    std::memset(fp32_buffer.data() + (global_idx + i) * d_model, 0, d_model * sizeof(float));
                }
                global_idx += padded_len;
            }
        }
        else
        {
            // Single sequence: direct embedding lookup
            for (int t = 0; t < num_tokens; ++t)
            {
                const int token_id = params_.token_ids[t];
                std::memcpy(fp32_buffer.data() + t * d_model,
                            embed_data + token_id * d_model,
                            d_model * sizeof(float));
            }
        }

        // Quantize FP32 → Q16_1 using the partial copy method
        // Only quantize the rows we actually filled, not the entire tensor
        if (!q16_output->copyFrom_fp32_rows(fp32_buffer.data(), static_cast<size_t>(num_tokens)))
        {
            LOG_ERROR("[EmbeddingStage] Failed to quantize FP32 to Q16_1");
            return false;
        }

        LOG_DEBUG("[EmbeddingStage] Q16_1 output: " << num_tokens << " tokens quantized");
        return true;
    }

    /**
     * @brief Zero out a row of the output tensor (for padding)
     *
     * Handles FP32, Q8_1, and Q16_1 output formats.
     */
    void EmbeddingStage::zero_output_row(ITensor *output, int row_idx, int d_model)
    {
        // Cast to TensorBase for CPU operations
        auto *output_base = dynamic_cast<TensorBase *>(output);
        if (!output_base)
        {
            LOG_ERROR("[EmbeddingStage::zero_output_row] GPU tensors not yet supported");
            return;
        }

        TensorType output_type = output_base->native_type();

        if (output_type == TensorType::FP32)
        {
            float *data = const_cast<float *>(output_base->fp32_data());
            if (data)
            {
                std::memset(data + row_idx * d_model, 0, d_model * sizeof(float));
            }
        }
        else if (output_type == TensorType::Q8_1)
        {
            // For Q8_1, zero out the blocks for this row
            auto *q8_1_output = dynamic_cast<Q8_1Tensor *>(output_base);
            if (q8_1_output)
            {
                // Q8_1 has 32 elements per block
                const int block_size = 32;
                const int blocks_per_row = (d_model + block_size - 1) / block_size;
                Q8_1Block *blocks = q8_1_output->mutable_typed_data();

                if (blocks)
                {
                    for (int b = 0; b < blocks_per_row; ++b)
                    {
                        Q8_1Block &block = blocks[row_idx * blocks_per_row + b];
                        block.d = 0.0f;
                        std::memset(block.qs, 0, 32);
                    }
                }
            }
        }
        else if (output_type == TensorType::Q16_1)
        {
            // For Q16_1, zero out the blocks for this row
            auto *q16_1_output = dynamic_cast<Q16_1Tensor *>(output_base);
            if (q16_1_output)
            {
                // Q16_1 has 32 elements per block
                const int block_size = 32;
                const int blocks_per_row = (d_model + block_size - 1) / block_size;
                Q16_1Block *blocks = q16_1_output->mutable_typed_data();

                if (blocks)
                {
                    for (int b = 0; b < blocks_per_row; ++b)
                    {
                        Q16_1Block &block = blocks[row_idx * blocks_per_row + b];
                        block.d = 0.0f;
                        block.sum_qs = 0;
                        std::memset(block.qs, 0, sizeof(block.qs));
                    }
                }
            }
        }
    }

    size_t EmbeddingStage::estimatedFlops() const
    {
        // Embedding is pure memory ops, no FLOPs
        return 0;
    }

    size_t EmbeddingStage::estimatedMemoryBytes() const
    {
        // Read: embed_table[token_id] for each token
        // Write: output[num_tokens, d_model]
        return params_.num_tokens * params_.d_model * sizeof(float) * 2;
    }

    bool EmbeddingStage::supportsBackend(ComputeBackendType backend) const
    {
        // EmbeddingStage supports GPU backends when CUDA is enabled
        // KernelFactory::createEmbedding() provides CUDA implementation
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
#ifdef HAVE_CUDA
        case ComputeBackendType::GPU_CUDA:
            return true; // CUDAEmbeddingKernelT is available via KernelFactory
#endif
#ifdef HAVE_ROCM
        case ComputeBackendType::GPU_ROCM:
            return false; // ROCm embedding not yet implemented
#endif
        default:
            return false;
        }
    }

    StageDumpInfo EmbeddingStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Weight: embedding table (must be added for coherence to upload to GPU!)
        if (params_.embed_table)
        {
            info.addWeight("embed_table", params_.embed_table);
        }

        if (params_.output)
        {
            info.addOutput("embeddings", params_.output,
                           params_.num_tokens, params_.d_model);
        }

        info.addScalarInt("num_tokens", params_.num_tokens);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt("vocab_size", params_.vocab_size);
        info.addScalarInt("device_id", params_.device_id.toKernelDeviceIndex());

        return info;
    }

    StageBufferRequirements EmbeddingStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.embed_table || !params_.output)
            return reqs; // Empty if tensors not set

        // WEIGHT buffer (embedding table - read-only)
        BufferTensorType embed_type = toBufferTensorType(params_.embed_table->native_type());
        reqs.addWeight("embed_table",
                       {static_cast<size_t>(params_.vocab_size), static_cast<size_t>(params_.d_model)},
                       embed_type);

        // OUTPUT buffer (embeddings)
        BufferTensorType out_type = toBufferTensorType(params_.output->native_type());
        reqs.addOutput("output",
                       {static_cast<size_t>(params_.num_tokens), static_cast<size_t>(params_.d_model)},
                       out_type);

        // Note: token_ids is a raw int* pointer, not a TensorBase, so we don't
        // declare it as a buffer (it's typically a small CPU array)

        return reqs;
    }

} // namespace llaminar2
