/**
 * @file CUDAEmbeddingKernelT.h
 * @brief CUDA implementation of embedding lookup kernel
 *
 * Handles embedding table lookup from device-owned weights. Quantized tables
 * must be prepared and uploaded before execution; FP32 tables must already be
 * resident on the target CUDA device.
 *
 * Supports FP32 embedding tables with FP32 output.
 *
 * ## Workspace Support (REQUIRED)
 *
 * Implements IWorkspaceConsumer for allocation-free hot-path execution.
 * **Workspace MUST be bound via bindWorkspace() before calling apply_tensor().**
 * The workspace provides pre-allocated buffers from DeviceWorkspaceManager.
 *
 * Without a bound workspace, apply_tensor() will return false with an error.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../../../backends/IWorkerGPUContext.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include <stdexcept>

namespace llaminar2
{

    /**
     * @brief CUDA implementation of embedding kernel for FP32
     *
     * ## Workspace Support (REQUIRED)
     *
     * Implements IWorkspaceConsumer for allocation-free hot-path execution.
     * **Workspace MUST be bound via bindWorkspace() before calling apply_tensor().**
     */
    class CUDAEmbeddingKernelT : public ITensorEmbedding, public IWorkspaceConsumer
    {
    public:
        explicit CUDAEmbeddingKernelT(int device_idx = -1) : device_idx_(device_idx) {}

        /**
         * @brief Construct with device context (Phase 4 pattern)
         * @param ctx Device context for shared handles/streams
         */
        explicit CUDAEmbeddingKernelT(IWorkerGPUContext *ctx)
        {
            if (!ctx)
                throw std::runtime_error("CUDAEmbeddingKernelT: Device context is null");
            if (!ctx->isInitialized())
                throw std::runtime_error("CUDAEmbeddingKernelT: Device context not initialized");
            device_ctx_ = ctx;
            device_idx_ = ctx->deviceOrdinal();
        }

        ~CUDAEmbeddingKernelT() override;

        // ===== Device Context Support (Phase 4) =====
        void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
        IWorkerGPUContext *deviceContext() const { return device_ctx_; }
        bool hasDeviceContext() const { return device_ctx_ != nullptr; }
        void *getStream() const { return requireExplicitGPUStreamBinding(gpu_stream_, "GPU tensor kernel"); }

        // GPU stream for graph capture support
        void bindGPUStream(ExplicitGPUStream stream) override { gpu_stream_ = stream.get(); }
        void clearGPUStreamBinding() override { gpu_stream_ = nullptr; }
        void setPreparedEmbeddingHandle(const PreparedEmbeddingHandle *handle) override
        {
            prepared_embedding_handle_ = handle;
        }

        void setVocabRange(int vocab_offset, int local_vocab_size) override
        {
            explicit_vocab_range_ = true;
            vocab_offset_ = vocab_offset;
            local_vocab_size_ = local_vocab_size;
        }

        void setAllowOutOfRangeTokenIds(bool allow) override
        {
            allow_out_of_range_token_ids_ = allow;
        }

        bool supports_device(int device_idx) const override
        {
            return device_idx >= 0; // GPU only
        }

        /**
         * @brief Execute embedding lookup with FP32 output
         */
        bool apply(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            float *output,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1);

        /**
         * @brief Execute embedding lookup with BF16 output (not yet implemented)
         */
        bool apply_bf16(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            uint16_t *output,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1);

        /**
         * @brief Execute embedding lookup with FP16 output (not yet implemented)
         */
        bool apply_fp16(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            uint16_t *output,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1);

        /**
         * @brief Execute embedding lookup with Q8_1 output (not yet implemented)
         */
        bool apply_q8_1(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            void *output,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1);

        /**
         * @brief Apply embedding lookup using tensor objects with automatic type dispatch
         */
        bool apply_tensor(
            const TensorBase *embed_table,
            const int *token_ids,
            int num_tokens,
            int d_model,
            TensorBase *output,
            const IMPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        void setDynamicTokenIds(const int *token_ids, int num_tokens) override;
        void setDynamicDeviceTokenIds(const void *token_ids_device, int num_tokens) override;

        // =====================================================================
        // Session Lifecycle (ITensorKernel overrides)
        // =====================================================================

        /**
         * @brief Reset input-dependent cached state
         *
         * Clears the dynamic_params_active_ flag and token count so that the
         * next apply_tensor() call re-uploads token IDs from scratch.
         * Called by KernelFactory::resetAllDynamicState() on session boundary.
         */
        void resetDynamicState() override;

        /**
         * @brief Check if dynamic token state is cached
         * @return true if a previous setDynamicTokenIds() preload is active
         */
        bool hasDynamicStateActive() const override { return dynamic_params_active_; }

        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::embedding()
                .withWeight("embed_table", "embedding table [vocab_size, d_model]", KernelBufferDtype::FP32)
                .withInput("token_ids", "input token IDs [num_tokens]", KernelBufferDtype::INT32)
                .withOutput("output", "embedded output [num_tokens, d_model]", KernelBufferDtype::FP32)
                .withScalar("num_tokens", "number of tokens", KernelBufferDtype::INT32)
                .withScalar("d_model", "embedding dimension", KernelBufferDtype::INT32);
        }

        // =========================================================================
        // IWorkspaceConsumer Interface
        // =========================================================================

        /**
         * @brief Get workspace requirements for embedding lookup
         *
         * Returns the stable INT32 token-ID buffer used by prefill and by graph
         * replay when a device sampler has not supplied token IDs directly.
         * Embedding weights are persistent model state and never workspace.
         *
         * @param m Maximum sequence length (num_tokens)
         * @param n Not used (pass 0)
         * @param k Unused (embedding weights are prepared separately)
         * @return WorkspaceRequirements describing all needed buffers
         */
        WorkspaceRequirements getWorkspaceRequirements(
            int m, int n = 0, int k = 0) const override;

        /**
         * @brief Bind workspace manager (REQUIRED for apply_tensor)
         *
         * **MUST be called before apply_tensor().** After binding, the kernel uses
         * pre-allocated buffers from the workspace manager for allocation-free execution.
         *
         * @param workspace Pointer to workspace manager (NOT owned, must outlive kernel)
         */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;

        /**
         * @brief Check if a workspace is currently bound
         */
        bool hasWorkspace() const override;

        /**
         * @brief Get the currently bound workspace manager
         */
        DeviceWorkspaceManager *getWorkspace() const override;

    private:
        int device_idx_ = 0;
        IWorkerGPUContext *device_ctx_ = nullptr;
        void *gpu_stream_ = nullptr;
        const PreparedEmbeddingHandle *prepared_embedding_handle_ = nullptr;

        // IWorkspaceConsumer state
        DeviceWorkspaceManager *workspace_ = nullptr; ///< Bound workspace manager (not owned)

        int *h_token_ids_ = nullptr;
        int max_token_ids_ = 0;
        int dynamic_token_count_ = 0;
        bool dynamic_params_active_ = false;
        const int *device_token_ids_ = nullptr;
        int device_token_count_ = 0;
        bool device_token_ids_active_ = false;
        void *preload_stream_ = nullptr; ///< Stream used for the last setDynamicTokenIds H2D copy

        int vocab_offset_ = 0;
        int local_vocab_size_ = 0;
        bool explicit_vocab_range_ = false;
        bool allow_out_of_range_token_ids_ = false;
    };

    // Convenience alias
    using CUDAEmbeddingKernel = CUDAEmbeddingKernelT;

} // namespace llaminar2
