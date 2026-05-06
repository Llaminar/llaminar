/**
 * @file CUDAEmbeddingKernelT.h
 * @brief CUDA implementation of embedding lookup kernel
 *
 * Handles embedding table lookup on GPU. The embedding table is
 * uploaded to GPU memory and rows are looked up based on token IDs.
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
#include <unordered_map>
#include <mutex>
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
        void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

        // GPU stream for graph capture support
        void setGPUStream(void *stream) override { gpu_stream_ = stream; }
        void setPreparedEmbeddingHandle(const PreparedEmbeddingHandle *handle) override
        {
            prepared_embedding_handle_ = handle;
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
         * Returns buffers needed for embedding:
         * - embed_token_ids [max_seq_len]: INT32 token IDs on GPU
         * - embed_table_temp [vocab_size × d_model]: FP32 temp buffer for non-GPU embed tables
         *
         * @param m Maximum sequence length (num_tokens)
         * @param n Not used (pass 0)
         * @param k d_model dimension (embedding dimension)
         * @return WorkspaceRequirements describing all needed buffers
         *
         * @note For embed_table_temp, vocab_size is estimated as 151936 (Qwen2 vocab).
         *       Actual vocab size may be smaller; the buffer will be sufficient.
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

        /**
         * @brief Clear the cached embedding table pointer for this workspace
         *
         * Call this if the model changes or the workspace is reset.
         * The next apply_tensor() call will re-upload the embedding table.
         */
        void clearEmbeddingCache()
        {
            if (workspace_)
            {
                std::lock_guard<std::mutex> lock(s_embed_cache_mutex_);
                s_workspace_embed_cache_.erase(workspace_);
            }
        }

        /**
         * @brief Static method to clear ALL embedding caches (for model unload)
         */
        static void clearGlobalEmbeddingCache()
        {
            std::lock_guard<std::mutex> lock(s_embed_cache_mutex_);
            s_workspace_embed_cache_.clear();
        }

    private:
        int device_idx_ = 0;
        IWorkerGPUContext *device_ctx_ = nullptr;
        void *gpu_stream_ = nullptr;
        const PreparedEmbeddingHandle *prepared_embedding_handle_ = nullptr;

        // IWorkspaceConsumer state
        DeviceWorkspaceManager *workspace_ = nullptr; ///< Bound workspace manager (not owned)

        // Embedding table caching state (STATIC MAP - per-workspace cache)
        // This is critical for performance: kernel instances are recreated every forward pass
        // due to graph rebuild, but the embedding table in GPU workspace is persistent.
        // Using static ensures we don't re-upload 500+ MB every decode step.
        // KEY FIX: Use per-workspace cache to support LOCAL TP with multiple devices.
        // Each device has its own workspace, so we cache separately per workspace.
        // THREAD SAFETY: Protected by s_embed_cache_mutex_ for LocalTP multi-device access.
        static inline std::mutex s_embed_cache_mutex_;
        static inline std::unordered_map<DeviceWorkspaceManager *, const TensorBase *> s_workspace_embed_cache_;

        int *h_token_ids_ = nullptr;
        int max_token_ids_ = 0;
        int dynamic_token_count_ = 0;
        bool dynamic_params_active_ = false;
        void *preload_stream_ = nullptr; ///< Stream used for the last setDynamicTokenIds H2D copy
    };

    // Convenience alias
    using CUDAEmbeddingKernel = CUDAEmbeddingKernelT;

} // namespace llaminar2
