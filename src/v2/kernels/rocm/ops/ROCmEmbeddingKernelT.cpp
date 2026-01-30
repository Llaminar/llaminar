/**
 * @file ROCmEmbeddingKernelT.cpp
 * @brief ROCm embedding kernel host-side implementation
 */

#include "ROCmEmbeddingKernelT.h"
#include "utils/Logger.h"
#include "utils/KernelProfiler.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"

#include <hip/hip_runtime.h>

// Forward declarations for HIP kernels (defined in ROCmEmbeddingKernels.hip)
extern "C"
{
    hipError_t hipOps_embedding_fp32(
        const float *embed_data,
        const int *token_ids,
        float *output,
        int num_tokens,
        int d_model,
        hipStream_t stream);

    hipError_t hipOps_embedding_bf16(
        const float *embed_data,
        const int *token_ids,
        uint16_t *output,
        int num_tokens,
        int d_model,
        hipStream_t stream);

    hipError_t hipOps_embedding_fp16(
        const float *embed_data,
        const int *token_ids,
        uint16_t *output,
        int num_tokens,
        int d_model,
        hipStream_t stream);

    hipError_t hipOps_embedding_q8_1(
        const float *embed_data,
        const int *token_ids,
        void *output,
        int num_tokens,
        int d_model,
        hipStream_t stream);
}

namespace llaminar2
{

    bool ROCmEmbeddingKernelT::apply(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        float *output,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        int dev = (device_idx >= 0) ? device_idx : device_idx_;

        hipError_t set_err = hipSetDevice(dev);
        if (set_err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Failed to set device " << dev << ": " << hipGetErrorString(set_err));
            return false;
        }

        hipError_t err = hipOps_embedding_fp32(embed_data, token_ids, output, num_tokens, d_model, nullptr);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] FP32 kernel failed: " << hipGetErrorString(err));
            return false;
        }

        return true;
    }

    bool ROCmEmbeddingKernelT::apply_bf16(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        uint16_t *output,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        int dev = (device_idx >= 0) ? device_idx : device_idx_;

        hipError_t set_err = hipSetDevice(dev);
        if (set_err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Failed to set device " << dev << ": " << hipGetErrorString(set_err));
            return false;
        }

        hipError_t err = hipOps_embedding_bf16(embed_data, token_ids, output, num_tokens, d_model, nullptr);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] BF16 kernel failed: " << hipGetErrorString(err));
            return false;
        }

        return true;
    }

    bool ROCmEmbeddingKernelT::apply_fp16(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        uint16_t *output,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        int dev = (device_idx >= 0) ? device_idx : device_idx_;

        hipError_t set_err = hipSetDevice(dev);
        if (set_err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Failed to set device " << dev << ": " << hipGetErrorString(set_err));
            return false;
        }

        hipError_t err = hipOps_embedding_fp16(embed_data, token_ids, output, num_tokens, d_model, nullptr);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] FP16 kernel failed: " << hipGetErrorString(err));
            return false;
        }

        return true;
    }

    bool ROCmEmbeddingKernelT::apply_q8_1(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        void *output,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        int dev = (device_idx >= 0) ? device_idx : device_idx_;

        hipError_t set_err = hipSetDevice(dev);
        if (set_err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Failed to set device " << dev << ": " << hipGetErrorString(set_err));
            return false;
        }

        if (d_model % 32 != 0)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Q8_1 requires d_model to be multiple of 32, got " << d_model);
            return false;
        }

        hipError_t err = hipOps_embedding_q8_1(embed_data, token_ids, output, num_tokens, d_model, nullptr);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Q8_1 kernel failed: " << hipGetErrorString(err));
            return false;
        }

        return true;
    }

    bool ROCmEmbeddingKernelT::apply_tensor(
        const TensorBase *embed_table,
        const int *token_ids,
        int num_tokens,
        int d_model,
        TensorBase *output,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        KERNEL_PROFILE_SCOPE(KernelType::EMBEDDING);
        (void)mpi_ctx;

        if (!embed_table || !output)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] apply_tensor: null tensor pointer");
            return false;
        }

        // Output must be FP32 for now
        if (output->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Output must be FP32 tensor, got " << static_cast<int>(output->native_type()));
            return false;
        }

        auto *output_fp32 = dynamic_cast<FP32Tensor *>(output);
        if (!output_fp32)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Output tensor cast to FP32 failed");
            return false;
        }

        // Get dequantized FP32 data from any tensor type via data()
        // This handles Q4_0, Q8_0, FP32, etc. automatically
        const float *embed_data = embed_table->data();
        if (!embed_data)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Failed to get embedding data");
            return false;
        }

        // Determine target ROCm device
        int dev = (device_idx >= 0) ? device_idx : device_idx_;

        hipError_t set_err = hipSetDevice(dev);
        if (set_err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Failed to set device " << dev << ": " << hipGetErrorString(set_err));
            return false;
        }

        // =====================================================================
        // Step 1: Get token_ids buffer from workspace and copy data
        // =====================================================================
        int *d_token_ids = nullptr;
        size_t token_bytes = static_cast<size_t>(num_tokens) * sizeof(int);

        if (!hasWorkspace())
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Workspace not bound - hot-path allocation disabled. "
                      "Call bindWorkspace() before apply_tensor()");
            return false;
        }

        // Use pre-allocated workspace buffer (workspace is required)
        d_token_ids = static_cast<int *>(workspace_->getBuffer(EmbeddingWorkspaceBuffers::TOKEN_IDS));
        if (!d_token_ids)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Workspace buffer '" << EmbeddingWorkspaceBuffers::TOKEN_IDS << "' not found");
            return false;
        }

        hipError_t err = hipMemcpy(d_token_ids, token_ids, token_bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Failed to copy token_ids to GPU: " << hipGetErrorString(err));
            return false;
        }

        // =====================================================================
        // Step 2: Get GPU pointer for output (coherence handled by GraphExecutor)
        // =====================================================================
        float *d_output = static_cast<float *>(output_fp32->gpu_data_ptr());
        if (!d_output)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Output GPU pointer is null");
            return false;
        }

        // =====================================================================
        // Step 3: Get or upload embedding table to GPU
        // =====================================================================
        float *d_embed = nullptr;

        // Check if embed_table is FP32 and on GPU
        auto *embed_fp32 = dynamic_cast<const FP32Tensor *>(embed_table);
        if (embed_fp32 && embed_fp32->isOnGPU())
        {
            // Fast path: FP32 tensor already on GPU
            d_embed = const_cast<float *>(static_cast<const float *>(embed_fp32->gpu_data_ptr()));
        }
        else
        {
            // Workspace-cached path: Upload dequantized embedding table once, reuse across calls
            // This is critical for performance with quantized embeddings (Q4_0, etc.)
            // The embedding table (~500MB) should only be uploaded once per model load
            size_t vocab_size = embed_table->rows();
            size_t embed_dim = static_cast<size_t>(d_model);
            size_t embed_bytes = vocab_size * embed_dim * sizeof(float);

            // Use pre-allocated workspace buffer (workspace is required)
            d_embed = static_cast<float *>(workspace_->getBuffer(EmbeddingWorkspaceBuffers::EMBED_TABLE));
            if (!d_embed)
            {
                LOG_ERROR("[ROCmEmbeddingKernelT] Workspace buffer '" << EmbeddingWorkspaceBuffers::EMBED_TABLE << "' not found");
                return false;
            }

            // Check if we need to upload (first call or different embedding table for THIS workspace)
            auto it = s_workspace_embed_cache_.find(workspace_);
            bool needs_upload = (it == s_workspace_embed_cache_.end()) || (it->second != embed_table);
            if (needs_upload)
            {
                err = hipMemcpy(d_embed, embed_data, embed_bytes, hipMemcpyHostToDevice);
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ROCmEmbeddingKernelT] Failed to copy embeddings to GPU: " << hipGetErrorString(err));
                    return false;
                }

                // Cache the tensor pointer for THIS workspace to avoid re-upload on subsequent calls
                s_workspace_embed_cache_[workspace_] = embed_table;
                LOG_INFO("[ROCmEmbeddingKernelT] Uploaded dequantized embedding table: "
                         << vocab_size << "x" << embed_dim << " (" << embed_bytes / (1024 * 1024) << " MB)"
                         << " workspace=" << static_cast<void *>(workspace_));
            }
            // else: embedding already uploaded to workspace buffer, reuse d_embed
        }

        // =====================================================================
        // Step 4: Launch kernel with GPU pointers
        // =====================================================================
        LOG_DEBUG("[ROCmEmbeddingKernelT] Launching kernel: d_embed=" << static_cast<void *>(d_embed)
                                                                      << " d_token_ids=" << static_cast<void *>(d_token_ids)
                                                                      << " d_output=" << static_cast<void *>(d_output)
                                                                      << " num_tokens=" << num_tokens << " d_model=" << d_model
                                                                      << " dev=" << dev);

        err = hipOps_embedding_fp32(d_embed, d_token_ids, d_output, num_tokens, d_model, nullptr);

        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] FP32 kernel failed: " << hipGetErrorString(err)
                                                                    << " (d_embed=" << static_cast<void *>(d_embed)
                                                                    << " d_token_ids=" << static_cast<void *>(d_token_ids)
                                                                    << " d_output=" << static_cast<void *>(d_output)
                                                                    << " num_tokens=" << num_tokens << " d_model=" << d_model << ")");
            return false;
        }

        return true;
    }

    // =============================================================================
    // IWorkspaceConsumer Interface Implementation
    // =============================================================================

    WorkspaceRequirements ROCmEmbeddingKernelT::getWorkspaceRequirements(
        int m, int n, int k) const
    {
        (void)n; // Unused for embedding

        WorkspaceRequirements reqs;

        // Buffer 1: Token IDs [max_seq_len × sizeof(int)]
        // m is the maximum sequence length
        size_t token_ids_bytes = static_cast<size_t>(m) * sizeof(int);
        reqs.buffers.push_back({
            EmbeddingWorkspaceBuffers::TOKEN_IDS,
            token_ids_bytes,
            256, // Alignment for HIP
            true // Required
        });

        // Buffer 2: Embedding table temp [vocab_size × d_model × sizeof(float)]
        // Used when embedding table is not already on GPU (quantized tables)
        // Use conservative estimates: vocab_size = 151936 (Qwen2), d_model = k
        // If k not provided, use 896 (Qwen2.5-0.5B d_model)
        constexpr size_t DEFAULT_VOCAB_SIZE = 151936;
        size_t d_model = (k > 0) ? static_cast<size_t>(k) : 896;
        size_t embed_table_bytes = DEFAULT_VOCAB_SIZE * d_model * sizeof(float);
        reqs.buffers.push_back({
            EmbeddingWorkspaceBuffers::EMBED_TABLE,
            embed_table_bytes,
            256, // Alignment for HIP
            true // Required - needed for quantized embedding tables
        });

        return reqs;
    }

    void ROCmEmbeddingKernelT::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        workspace_ = workspace;
    }

    bool ROCmEmbeddingKernelT::hasWorkspace() const
    {
        return workspace_ != nullptr && workspace_->isAllocated();
    }

    DeviceWorkspaceManager *ROCmEmbeddingKernelT::getWorkspace() const
    {
        return workspace_;
    }

} // namespace llaminar2
