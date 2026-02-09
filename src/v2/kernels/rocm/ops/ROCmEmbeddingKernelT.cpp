/**
 * @file ROCmEmbeddingKernelT.cpp
 * @brief ROCm embedding kernel host-side implementation
 */

#include "ROCmEmbeddingKernelT.h"
#include "utils/Logger.h"
#include "utils/ROCmKernelProfiler.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../common/EmbedQ8Repack.h"
#include "../../../backends/rocm/HipDeviceGuard.h"

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

    hipError_t hipOps_embedding_q8(
        const void *embed_q8,
        const int *token_ids,
        float *output,
        int num_tokens,
        int d_model,
        int blocks_per_row,
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

        hipError_t set_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(dev));
        if (set_err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Failed to set device " << dev << ": " << hipGetErrorString(set_err));
            return false;
        }

        hipError_t err = hipOps_embedding_fp32(embed_data, token_ids, output, num_tokens, d_model, static_cast<hipStream_t>(getStream()));
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

        hipError_t set_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(dev));
        if (set_err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Failed to set device " << dev << ": " << hipGetErrorString(set_err));
            return false;
        }

        hipError_t err = hipOps_embedding_bf16(embed_data, token_ids, output, num_tokens, d_model, static_cast<hipStream_t>(getStream()));
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

        hipError_t set_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(dev));
        if (set_err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Failed to set device " << dev << ": " << hipGetErrorString(set_err));
            return false;
        }

        hipError_t err = hipOps_embedding_fp16(embed_data, token_ids, output, num_tokens, d_model, static_cast<hipStream_t>(getStream()));
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

        hipError_t set_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(dev));
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

        hipError_t err = hipOps_embedding_q8_1(embed_data, token_ids, output, num_tokens, d_model, static_cast<hipStream_t>(getStream()));
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
        ROCM_KERNEL_PROFILE_SCOPE_STREAM(ROCmKernelType::EMBEDDING_LOOKUP, static_cast<hipStream_t>(getStream()));
        (void)mpi_ctx;

        if (!embed_table || !output)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] apply_tensor: null tensor pointer");
            return false;
        }

        // Output must be FP32
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

        // Set target ROCm device
        int dev = (device_idx >= 0) ? device_idx : device_idx_;
        hipError_t set_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(dev));
        if (set_err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Failed to set device " << dev << ": " << hipGetErrorString(set_err));
            return false;
        }

        // =====================================================================
        // Step 1: Get token_ids buffer from workspace and copy data
        // =====================================================================
        if (!hasWorkspace())
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Workspace not bound - call bindWorkspace() before apply_tensor()");
            return false;
        }

        int *d_token_ids = static_cast<int *>(workspace_->getBuffer(EmbeddingWorkspaceBuffers::TOKEN_IDS));
        if (!d_token_ids)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Workspace buffer '" << EmbeddingWorkspaceBuffers::TOKEN_IDS << "' not found");
            return false;
        }

        size_t token_bytes = static_cast<size_t>(num_tokens) * sizeof(int);
        // Use async copy on the device stream so this operation is compatible
        // with HIP/CUDA stream capture (GPU graph recording). Synchronous
        // hipMemcpy uses the legacy stream which would create a dependency on
        // a capturing stream, causing capture to fail.
        hipStream_t stream = static_cast<hipStream_t>(getStream());
        hipError_t err = hipMemcpyAsync(d_token_ids, token_ids, token_bytes, hipMemcpyHostToDevice, stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Failed to copy token_ids to GPU: " << hipGetErrorString(err));
            return false;
        }

        // =====================================================================
        // Step 2: Get GPU pointer for output
        // =====================================================================
        float *d_output = static_cast<float *>(output_fp32->gpu_data_ptr());
        if (!d_output)
        {
            LOG_ERROR("[ROCmEmbeddingKernelT] Output GPU pointer is null");
            return false;
        }

        // =====================================================================
        // Step 3: Route by embedding table format
        // =====================================================================

        // --- Fast path: FP32 tensor already on GPU ---
        auto *embed_fp32 = dynamic_cast<const FP32Tensor *>(embed_table);
        if (embed_fp32 && embed_fp32->isOnGPU())
        {
            float *d_embed = const_cast<float *>(static_cast<const float *>(embed_fp32->gpu_data_ptr()));
            LOG_DEBUG("[ROCmEmbeddingKernelT] FP32 fast path: d_embed=" << static_cast<void *>(d_embed)
                                                                        << " num_tokens=" << num_tokens << " d_model=" << d_model);
            err = hipOps_embedding_fp32(d_embed, d_token_ids, d_output, num_tokens, d_model, stream);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmEmbeddingKernelT] FP32 kernel failed: " << hipGetErrorString(err));
                return false;
            }
            return true;
        }

        // --- Quantized path: repack to EmbedQ8 via IINT8Unpackable ---
        const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(embed_table);
        if (unpackable)
        {
            void *d_embed_q8 = workspace_->getBuffer(EmbeddingWorkspaceBuffers::EMBED_TABLE);
            if (!d_embed_q8)
            {
                LOG_ERROR("[ROCmEmbeddingKernelT] Workspace buffer '" << EmbeddingWorkspaceBuffers::EMBED_TABLE << "' not found");
                return false;
            }

            auto it = s_workspace_embed_cache_.find(workspace_);
            bool needs_upload = (it == s_workspace_embed_cache_.end()) || (it->second != embed_table);
            if (needs_upload)
            {
                auto repacked = repackEmbeddingToQ8(embed_table, d_model);

                // Use sync copy for the large one-time embedding upload.
                // This only happens on the first forward pass, not during
                // cached decode, so it won't conflict with graph capture.
                err = hipMemcpy(d_embed_q8, repacked.data.data(), repacked.byte_size,
                                hipMemcpyHostToDevice);
                if (err != hipSuccess)
                {
                    LOG_ERROR("[ROCmEmbeddingKernelT] Failed to upload EmbedQ8 data: " << hipGetErrorString(err));
                    return false;
                }

                s_workspace_embed_cache_[workspace_] = embed_table;
                LOG_INFO("[ROCmEmbeddingKernelT] Uploaded EmbedQ8 embedding: "
                         << tensorTypeName(embed_table->native_type()) << " "
                         << repacked.vocab_size << "x" << d_model
                         << " \u2192 " << (repacked.byte_size / (1024 * 1024)) << " MB"
                         << " (" << repacked.blocks_per_row << " blocks/row)"
                         << " workspace=" << static_cast<void *>(workspace_));
            }

            size_t blocks_per_row = (static_cast<size_t>(d_model) + 31) / 32;
            err = hipOps_embedding_q8(d_embed_q8, d_token_ids, d_output,
                                      num_tokens, d_model,
                                      static_cast<int>(blocks_per_row), stream);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmEmbeddingKernelT] EmbedQ8 kernel failed: " << hipGetErrorString(err));
                return false;
            }
            return true;
        }

        // No FP32 fallback — embedding table must be either FP32-on-GPU or IINT8Unpackable
        LOG_ERROR("[ROCmEmbeddingKernelT] Embedding table type "
                  << tensorTypeName(embed_table->native_type())
                  << " is not FP32-on-GPU and does not implement IINT8Unpackable");
        return false;
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

        // Buffer 2: Embedding table temp [vocab_size × blocks_per_row × sizeof(EmbedQ8Block)]
        // Used when embedding table is not already on GPU (quantized → EmbedQ8 repack)
        // Use conservative estimates: vocab_size = 151936 (Qwen2), d_model = k
        // If k not provided, use 896 (Qwen2.5-0.5B d_model)
        constexpr size_t DEFAULT_VOCAB_SIZE = 151936;
        size_t d_model = (k > 0) ? static_cast<size_t>(k) : 896;
        size_t blocks_per_row = (d_model + 31) / 32;
        size_t embed_table_bytes = DEFAULT_VOCAB_SIZE * blocks_per_row * sizeof(EmbedQ8Block);
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
