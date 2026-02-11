/**
 * @file CUDARoPEKernelT.h
 * @brief CUDA RoPE kernel with typed precision support
 *
 * CUDA implementation of Rotary Positional Embeddings (RoPE) following
 * the same template pattern as CPURoPEKernelT. Specialized for FP32,
 * BF16, and FP16 precisions.
 *
 * Supports IWorkspaceConsumer for pre-allocated position_ids buffer.
 *
 * OPTIMIZATION PATHS (v5):
 * - DECODE (seq_len=1): Scalar position parameter - NO memcpy
 * - CONTIGUOUS (nullptr position_ids + pos_offset): Compute on GPU - ZERO memcpy
 * - NON-CONTIGUOUS: Use workspace buffer if bound, else fallback to malloc/memcpy
 *
 * @author Llaminar Team
 * @date 2025-01-14
 */

#pragma once

#include "../../../backends/IWorkerGPUContext.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/config/RuntimeConfig.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h" // For FP32Tensor, BF16Tensor, FP16Tensor
#include "../../../tensors/BlockStructures.h"
#include "../../../utils/Logger.h"
#include <cstdint>

namespace llaminar2
{
    namespace cuda
    {

        // =========================================================================
        // Primary Template (Unsupported Precisions)
        // =========================================================================

        template <ActivationPrecision Precision>
        class CUDARoPEKernelT
        {
            static_assert(Precision == ActivationPrecision::FP32 ||
                              Precision == ActivationPrecision::BF16 ||
                              Precision == ActivationPrecision::FP16,
                          "CUDARoPEKernelT only supports FP32, BF16, and FP16 precisions");
        };

        // =========================================================================
        // FP32 Specialization
        // =========================================================================

        template <>
        class CUDARoPEKernelT<ActivationPrecision::FP32> : public ITensorRoPE, public IWorkspaceConsumer
        {
        public:
            using StorageType = float;

            /// Maximum head_dim/2 for worst-case workspace allocation (covers head_dim up to 256)
            static constexpr int MAX_HALF_DIM = 128;

            explicit CUDARoPEKernelT(int device_idx = 0, float rope_theta = 10000.0f)
                : device_idx_(device_idx), rope_theta_(rope_theta), workspace_(nullptr),
                  inv_freq_initialized_(false), inv_freq_head_dim_(0), inv_freq_theta_(0.0f) {}
            ~CUDARoPEKernelT() override = default;

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            // ===== IWorkspaceConsumer interface =====
            WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override
            {
                (void)n;
                (void)k;
                WorkspaceRequirements reqs;
                // Position IDs buffer - m is max sequence length
                reqs.buffers.push_back({
                    RoPEWorkspaceBuffers::POSITION_IDS,
                    static_cast<size_t>(m) * sizeof(int),
                    256, // CUDA alignment
                    true // Required
                });
                // Inverse frequency table - allocated for worst-case head_dim
                reqs.buffers.push_back({
                    RoPEWorkspaceBuffers::INV_FREQ,
                    static_cast<size_t>(MAX_HALF_DIM) * sizeof(float),
                    256, // CUDA alignment
                    true // Required
                });
                return reqs;
            }

            void bindWorkspace(DeviceWorkspaceManager *workspace) override
            {
                // Only reset inv_freq state when workspace ACTUALLY changes.
                // See ROCmRoPEKernelT<FP32>::bindWorkspace() for full rationale.
                if (workspace_ != workspace)
                {
                    inv_freq_initialized_ = false;
                }
                workspace_ = workspace;
            }
            bool hasWorkspace() const override { return workspace_ != nullptr; }
            DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }

            // ===== GPU Stream Support (Graph Capture) =====
            void setGPUStream(void *stream) override { gpu_stream_ = stream; }

            // ===== ITensorRoPE interface =====
            bool apply(
                float *data, float *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, bool interleaved,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override;

            bool apply_bf16(
                uint16_t *data, uint16_t *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            bool apply_fp16(
                uint16_t *data, uint16_t *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            bool apply_q8_1(
                void *data, void *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            // ===== Typed API =====
            bool apply_typed(
                float *Q,
                float *K,
                const int *position_ids,
                int seq_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                float rope_theta,
                int device_idx = -1,
                int pos_offset = 0);

            // ===== Tensor-aware API (uses GPU memory, marks outputs dirty) =====
            bool apply_tensor(
                TensorBase *Q,
                TensorBase *K,
                const int *position_ids,
                int seq_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                float rope_theta,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                int pos_offset = 0) override
            {
                (void)mpi_ctx;

                LOG_DEBUG("[CUDARoPEKernelT<FP32>] apply_tensor called: seq_len=" << seq_len
                                                                                  << " n_heads=" << n_heads << " device_idx=" << device_idx
                                                                                  << " pos_offset=" << pos_offset);

                if (!Q || Q->native_type() != TensorType::FP32)
                {
                    LOG_ERROR("CUDARoPEKernelT<FP32>::apply_tensor: Q must be FP32Tensor");
                    return false;
                }

                auto *q_fp32 = dynamic_cast<FP32Tensor *>(Q);
                if (!q_fp32)
                {
                    LOG_ERROR("CUDARoPEKernelT<FP32>::apply_tensor: Q dynamic_cast failed");
                    return false;
                }

                // Get GPU pointer for Q - this may upload if needed via ensureOnDevice
                float *q_gpu = static_cast<float *>(q_fp32->gpu_data_ptr());
                if (!q_gpu)
                {
                    LOG_ERROR("CUDARoPEKernelT<FP32>::apply_tensor: Q has no GPU data");
                    return false;
                }

                // Handle K tensor if provided
                float *k_gpu = nullptr;
                FP32Tensor *k_fp32 = nullptr;
                if (K)
                {
                    if (K->native_type() != TensorType::FP32)
                    {
                        LOG_ERROR("CUDARoPEKernelT<FP32>::apply_tensor: K must be FP32Tensor");
                        return false;
                    }
                    k_fp32 = dynamic_cast<FP32Tensor *>(K);
                    if (!k_fp32)
                    {
                        LOG_ERROR("CUDARoPEKernelT<FP32>::apply_tensor: K dynamic_cast failed");
                        return false;
                    }
                    k_gpu = static_cast<float *>(k_fp32->gpu_data_ptr());
                    if (!k_gpu)
                    {
                        LOG_ERROR("CUDARoPEKernelT<FP32>::apply_tensor: K has no GPU data");
                        return false;
                    }
                }

                // Execute on GPU
                bool success = apply_typed(q_gpu, k_gpu, position_ids, seq_len, n_heads, n_kv_heads, head_dim, rope_theta, device_idx, pos_offset);

                // Mark tensors as modified on GPU
                if (success)
                {
                    q_fp32->mark_device_dirty();
                    if (k_fp32)
                    {
                        k_fp32->mark_device_dirty();
                    }
                }

                return success;
            }

            static constexpr ActivationPrecision precision() { return ActivationPrecision::FP32; }
            static const char *precision_name() { return "FP32"; }

        private:
            int device_idx_;
            float rope_theta_;
            DeviceWorkspaceManager *workspace_;
            IWorkerGPUContext *device_ctx_ = nullptr;
            void *gpu_stream_ = nullptr;

            // Inverse frequency cache state (for workspace-based allocation)
            mutable bool inv_freq_initialized_;
            mutable int inv_freq_head_dim_;
            mutable float inv_freq_theta_;
        };

        // =========================================================================
        // BF16 Specialization
        // =========================================================================

        template <>
        class CUDARoPEKernelT<ActivationPrecision::BF16> : public ITensorRoPE, public IWorkspaceConsumer
        {
        public:
            using StorageType = uint16_t;

            /// Maximum head_dim/2 for worst-case workspace allocation (covers head_dim up to 256)
            static constexpr int MAX_HALF_DIM = 128;

            explicit CUDARoPEKernelT(int device_idx = 0, float rope_theta = 10000.0f)
                : device_idx_(device_idx), rope_theta_(rope_theta), workspace_(nullptr),
                  inv_freq_initialized_(false), inv_freq_head_dim_(0), inv_freq_theta_(0.0f) {}

            /**
             * @brief Construct with device context (Phase 4 pattern)
             * @param ctx Device context for shared handles/streams
             * @param rope_theta RoPE theta parameter
             */
            explicit CUDARoPEKernelT(IWorkerGPUContext *ctx, float rope_theta = 10000.0f)
                : rope_theta_(rope_theta), workspace_(nullptr),
                  inv_freq_initialized_(false), inv_freq_head_dim_(0), inv_freq_theta_(0.0f)
            {
                if (!ctx)
                    throw std::runtime_error("CUDARoPEKernelT<BF16>: Device context is null");
                if (!ctx->isInitialized())
                    throw std::runtime_error("CUDARoPEKernelT<BF16>: Device context not initialized");
                device_ctx_ = ctx;
                device_idx_ = ctx->deviceOrdinal();
            }

            ~CUDARoPEKernelT() override = default;

            // ===== Device Context Support (Phase 4) =====
            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }
            void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

            // ===== GPU Stream Support (Graph Capture) =====
            void setGPUStream(void *stream) override { gpu_stream_ = stream; }

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            // ===== IWorkspaceConsumer interface =====
            WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override
            {
                (void)n;
                (void)k;
                WorkspaceRequirements reqs;
                // Position IDs buffer - m is max sequence length
                reqs.buffers.push_back({
                    RoPEWorkspaceBuffers::POSITION_IDS,
                    static_cast<size_t>(m) * sizeof(int),
                    256, // CUDA alignment
                    true // Required
                });
                // Inverse frequency table - allocated for worst-case head_dim
                reqs.buffers.push_back({
                    RoPEWorkspaceBuffers::INV_FREQ,
                    static_cast<size_t>(MAX_HALF_DIM) * sizeof(float),
                    256, // CUDA alignment
                    true // Required
                });
                return reqs;
            }

            void bindWorkspace(DeviceWorkspaceManager *workspace) override
            {
                // Only reset inv_freq state when workspace ACTUALLY changes.
                // See ROCmRoPEKernelT<FP32>::bindWorkspace() for full rationale.
                if (workspace_ != workspace)
                {
                    inv_freq_initialized_ = false;
                }
                workspace_ = workspace;
            }
            bool hasWorkspace() const override { return workspace_ != nullptr; }
            DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }

            // ===== ITensorRoPE interface =====
            bool apply(
                float *data, float *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, bool interleaved,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)interleaved;
                (void)mpi_ctx;
                (void)device_idx;
                return false;
            }

            bool apply_bf16(
                uint16_t *data, uint16_t *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override;

            bool apply_fp16(
                uint16_t *data, uint16_t *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            bool apply_q8_1(
                void *data, void *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            // ===== Typed API =====
            bool apply_typed(
                uint16_t *Q,
                uint16_t *K,
                const int *position_ids,
                int seq_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                float rope_theta,
                int device_idx = -1,
                int pos_offset = 0);

            // ===== Tensor-aware API (uses GPU memory, marks outputs dirty) =====
            bool apply_tensor(
                TensorBase *Q,
                TensorBase *K,
                const int *position_ids,
                int seq_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                float rope_theta,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                int pos_offset = 0) override
            {
                (void)mpi_ctx;

                if (!Q || Q->native_type() != TensorType::BF16)
                {
                    LOG_ERROR("CUDARoPEKernelT<BF16>::apply_tensor: Q must be BF16Tensor");
                    return false;
                }

                auto *q_bf16 = dynamic_cast<BF16Tensor *>(Q);
                if (!q_bf16)
                {
                    LOG_ERROR("CUDARoPEKernelT<BF16>::apply_tensor: Q dynamic_cast failed");
                    return false;
                }

                // Get GPU pointer for Q
                uint16_t *q_gpu = static_cast<uint16_t *>(q_bf16->gpu_data_ptr());
                if (!q_gpu)
                {
                    LOG_ERROR("CUDARoPEKernelT<BF16>::apply_tensor: Q has no GPU data");
                    return false;
                }

                // Handle K tensor if provided
                uint16_t *k_gpu = nullptr;
                BF16Tensor *k_bf16 = nullptr;
                if (K)
                {
                    if (K->native_type() != TensorType::BF16)
                    {
                        LOG_ERROR("CUDARoPEKernelT<BF16>::apply_tensor: K must be BF16Tensor");
                        return false;
                    }
                    k_bf16 = dynamic_cast<BF16Tensor *>(K);
                    if (!k_bf16)
                    {
                        LOG_ERROR("CUDARoPEKernelT<BF16>::apply_tensor: K dynamic_cast failed");
                        return false;
                    }
                    k_gpu = static_cast<uint16_t *>(k_bf16->gpu_data_ptr());
                    if (!k_gpu)
                    {
                        LOG_ERROR("CUDARoPEKernelT<BF16>::apply_tensor: K has no GPU data");
                        return false;
                    }
                }

                // Execute on GPU
                bool success = apply_typed(q_gpu, k_gpu, position_ids, seq_len, n_heads, n_kv_heads, head_dim, rope_theta, device_idx, pos_offset);

                // Mark tensors as modified on GPU
                if (success)
                {
                    q_bf16->mark_device_dirty();
                    if (k_bf16)
                    {
                        k_bf16->mark_device_dirty();
                    }
                }

                return success;
            }

            static constexpr ActivationPrecision precision() { return ActivationPrecision::BF16; }
            static const char *precision_name() { return "BF16"; }

        private:
            int device_idx_;
            float rope_theta_;
            DeviceWorkspaceManager *workspace_;
            IWorkerGPUContext *device_ctx_ = nullptr;
            void *gpu_stream_ = nullptr;

            // Inverse frequency cache state (for workspace-based allocation)
            mutable bool inv_freq_initialized_;
            mutable int inv_freq_head_dim_;
            mutable float inv_freq_theta_;
        };

        // =========================================================================
        // FP16 Specialization
        // =========================================================================

        template <>
        class CUDARoPEKernelT<ActivationPrecision::FP16> : public ITensorRoPE, public IWorkspaceConsumer
        {
        public:
            using StorageType = uint16_t;

            /// Maximum head_dim/2 for worst-case workspace allocation (covers head_dim up to 256)
            static constexpr int MAX_HALF_DIM = 128;

            explicit CUDARoPEKernelT(int device_idx = 0, float rope_theta = 10000.0f)
                : device_idx_(device_idx), rope_theta_(rope_theta), workspace_(nullptr),
                  inv_freq_initialized_(false), inv_freq_head_dim_(0), inv_freq_theta_(0.0f) {}

            /**
             * @brief Construct with device context (Phase 4 pattern)
             * @param ctx Device context for shared handles/streams
             * @param rope_theta RoPE theta parameter
             */
            explicit CUDARoPEKernelT(IWorkerGPUContext *ctx, float rope_theta = 10000.0f)
                : rope_theta_(rope_theta), workspace_(nullptr),
                  inv_freq_initialized_(false), inv_freq_head_dim_(0), inv_freq_theta_(0.0f)
            {
                if (!ctx)
                    throw std::runtime_error("CUDARoPEKernelT<FP16>: Device context is null");
                if (!ctx->isInitialized())
                    throw std::runtime_error("CUDARoPEKernelT<FP16>: Device context not initialized");
                device_ctx_ = ctx;
                device_idx_ = ctx->deviceOrdinal();
            }

            ~CUDARoPEKernelT() override = default;

            // ===== Device Context Support (Phase 4) =====
            void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }
            IWorkerGPUContext *deviceContext() const { return device_ctx_; }
            bool hasDeviceContext() const { return device_ctx_ != nullptr; }
            void *getStream() const { return device_ctx_ ? device_ctx_->defaultStream() : nullptr; }

            // ===== GPU Stream Support (Graph Capture) =====
            void setGPUStream(void *stream) override { gpu_stream_ = stream; }

            bool supports_device(int device_idx) const override { return device_idx >= 0; }

            // ===== IWorkspaceConsumer interface =====
            WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override
            {
                (void)n;
                (void)k;
                WorkspaceRequirements reqs;
                // Position IDs buffer - m is max sequence length
                reqs.buffers.push_back({
                    RoPEWorkspaceBuffers::POSITION_IDS,
                    static_cast<size_t>(m) * sizeof(int),
                    256, // CUDA alignment
                    true // Required
                });
                // Inverse frequency table - allocated for worst-case head_dim
                reqs.buffers.push_back({
                    RoPEWorkspaceBuffers::INV_FREQ,
                    static_cast<size_t>(MAX_HALF_DIM) * sizeof(float),
                    256, // CUDA alignment
                    true // Required
                });
                return reqs;
            }

            void bindWorkspace(DeviceWorkspaceManager *workspace) override
            {
                // Only reset inv_freq state when workspace ACTUALLY changes.
                // See ROCmRoPEKernelT<FP32>::bindWorkspace() for full rationale.
                if (workspace_ != workspace)
                {
                    inv_freq_initialized_ = false;
                }
                workspace_ = workspace;
            }
            bool hasWorkspace() const override { return workspace_ != nullptr; }
            DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }

            // ===== ITensorRoPE interface =====
            bool apply(
                float *data, float *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, bool interleaved,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)interleaved;
                (void)mpi_ctx;
                (void)device_idx;
                return false;
            }

            bool apply_bf16(
                uint16_t *data, uint16_t *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            bool apply_fp16(
                uint16_t *data, uint16_t *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override;

            bool apply_q8_1(
                void *data, void *output,
                const int *pos_ids,
                int batch_size, int seq_len, int head_dim, int num_heads,
                float theta_base, int device_idx) override
            {
                (void)data;
                (void)output;
                (void)pos_ids;
                (void)batch_size;
                (void)seq_len;
                (void)head_dim;
                (void)num_heads;
                (void)theta_base;
                (void)device_idx;
                return false;
            }

            // ===== Typed API =====
            bool apply_typed(
                uint16_t *Q,
                uint16_t *K,
                const int *position_ids,
                int seq_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                float rope_theta,
                int device_idx = -1,
                int pos_offset = 0);

            // ===== Tensor-aware API (uses GPU memory, marks outputs dirty) =====
            bool apply_tensor(
                TensorBase *Q,
                TensorBase *K,
                const int *position_ids,
                int seq_len,
                int n_heads,
                int n_kv_heads,
                int head_dim,
                float rope_theta,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1,
                int pos_offset = 0) override
            {
                (void)mpi_ctx;

                if (!Q || Q->native_type() != TensorType::FP16)
                {
                    LOG_ERROR("CUDARoPEKernelT<FP16>::apply_tensor: Q must be FP16Tensor");
                    return false;
                }

                auto *q_fp16 = dynamic_cast<FP16Tensor *>(Q);
                if (!q_fp16)
                {
                    LOG_ERROR("CUDARoPEKernelT<FP16>::apply_tensor: Q dynamic_cast failed");
                    return false;
                }

                // Get GPU pointer for Q
                uint16_t *q_gpu = static_cast<uint16_t *>(q_fp16->gpu_data_ptr());
                if (!q_gpu)
                {
                    LOG_ERROR("CUDARoPEKernelT<FP16>::apply_tensor: Q has no GPU data");
                    return false;
                }

                // Handle K tensor if provided
                uint16_t *k_gpu = nullptr;
                FP16Tensor *k_fp16 = nullptr;
                if (K)
                {
                    if (K->native_type() != TensorType::FP16)
                    {
                        LOG_ERROR("CUDARoPEKernelT<FP16>::apply_tensor: K must be FP16Tensor");
                        return false;
                    }
                    k_fp16 = dynamic_cast<FP16Tensor *>(K);
                    if (!k_fp16)
                    {
                        LOG_ERROR("CUDARoPEKernelT<FP16>::apply_tensor: K dynamic_cast failed");
                        return false;
                    }
                    k_gpu = static_cast<uint16_t *>(k_fp16->gpu_data_ptr());
                    if (!k_gpu)
                    {
                        LOG_ERROR("CUDARoPEKernelT<FP16>::apply_tensor: K has no GPU data");
                        return false;
                    }
                }

                // Execute on GPU
                bool success = apply_typed(q_gpu, k_gpu, position_ids, seq_len, n_heads, n_kv_heads, head_dim, rope_theta, device_idx, pos_offset);

                // Mark tensors as modified on GPU
                if (success)
                {
                    q_fp16->mark_device_dirty();
                    if (k_fp16)
                    {
                        k_fp16->mark_device_dirty();
                    }
                }

                return success;
            }

            static constexpr ActivationPrecision precision() { return ActivationPrecision::FP16; }
            static const char *precision_name() { return "FP16"; }

        private:
            int device_idx_;
            float rope_theta_;
            DeviceWorkspaceManager *workspace_;
            IWorkerGPUContext *device_ctx_ = nullptr;
            void *gpu_stream_ = nullptr;

            // Inverse frequency cache state (for workspace-based allocation)
            mutable bool inv_freq_initialized_;
            mutable int inv_freq_head_dim_;
            mutable float inv_freq_theta_;
        };

    } // namespace cuda
} // namespace llaminar2
