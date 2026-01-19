/**
 * @file ROCmEmbeddingKernelT.h
 * @brief ROCm implementation of ITensorEmbedding interface
 *
 * Provides FP32, BF16, FP16, and Q8_1 embedding lookup on AMD GPUs.
 */

#pragma once

#include "../../../tensors/TensorKernels.h"
#include "../../../interfaces/IWorkspaceConsumer.h"

namespace llaminar2
{

    /**
     * Standard buffer names used by embedding workspace consumers.
     */
    namespace EmbeddingWorkspaceBuffers
    {
        constexpr const char *TOKEN_IDS = "embed_token_ids";    ///< [max_seq_len] INT32 token IDs
        constexpr const char *EMBED_TABLE = "embed_table_temp"; ///< [vocab_size × d_model] FP32 temp for non-GPU embed tables
    }

    /**
     * @class ROCmEmbeddingKernelT
     * @brief ROCm/HIP embedding lookup kernel implementation
     *
     * Implements ITensorEmbedding interface for AMD GPUs using HIP.
     * Supports FP32, BF16, FP16, and Q8_1 output formats.
     *
     * ## Workspace Support (REQUIRED)
     *
     * Implements IWorkspaceConsumer for allocation-free hot-path execution.
     * **Workspace MUST be bound via bindWorkspace() before calling apply_tensor().**
     * The workspace provides pre-allocated buffers from DeviceWorkspaceManager.
     *
     * Without a bound workspace, apply_tensor() will return false with an error.
     */
    class ROCmEmbeddingKernelT : public ITensorEmbedding, public IWorkspaceConsumer
    {
    public:
        /**
         * @brief Construct with optional default device index
         * @param device_idx Default HIP device index (-1 for current device)
         */
        explicit ROCmEmbeddingKernelT(int device_idx = 0) : device_idx_(device_idx) {}

        ~ROCmEmbeddingKernelT() override = default;

        // ITensorKernel interface
        bool supports_device(int device_idx) const override
        {
            return device_idx >= 0; // ROCm supports any valid device index
        }

        // ITensorEmbedding interface - FP32 output
        bool apply(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            float *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        // ITensorEmbedding interface - BF16 output
        bool apply_bf16(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            uint16_t *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        // ITensorEmbedding interface - FP16 output
        bool apply_fp16(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            uint16_t *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        // ITensorEmbedding interface - Q8_1 output
        bool apply_q8_1(
            const float *embed_data,
            const int *token_ids,
            int num_tokens,
            int d_model,
            void *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

        // ITensorEmbedding interface - tensor-based dispatch
        bool apply_tensor(
            const TensorBase *embed_table,
            const int *token_ids,
            int num_tokens,
            int d_model,
            TensorBase *output,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override;

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

    private:
        int device_idx_;

        // IWorkspaceConsumer state
        DeviceWorkspaceManager *workspace_ = nullptr; ///< Bound workspace manager (not owned)
    };

} // namespace llaminar2
