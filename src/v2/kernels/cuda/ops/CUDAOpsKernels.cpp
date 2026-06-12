/**
 * @file CUDAOpsKernels.cpp
 * @brief Implementation of CUDA ops kernel ITensorXxx methods
 *
 * This file contains the C++ method implementations for CUDARMSNormKernelT,
 * CUDASwiGLUKernelT, and CUDARoPEKernelT. It calls the extern "C" wrapper
 * functions defined in CUDAOpsKernels.cu.
 *
 * Separating .cu and .cpp allows nvcc to compile only the CUDA code without
 * encountering issues with MPI or other complex C++ headers.
 */

#include "CUDARMSNormKernelT.h"
#include "CUDASwiGLUKernelT.h"
#include "CUDARoPEKernelT.h"
#include "CUDAEmbeddingKernelT.h"
#include "../../../tensors/Tensors.h"
#include "../../../backends/DeviceId.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../utils/Logger.h"
#include "../../../utils/CUDAKernelProfiler.h"
#include "../../../utils/DebugEnv.h"
#include "../../common/EmbedQ8Repack.h"
#include "../../common/PreparedEmbeddingWeights.h"
#include "../../KernelFactory.h"
#include <climits>
#include "../../rope/RoPEDeviceParams.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

// =========================================================================
// Extern "C" declarations for CUDA kernel wrappers
// =========================================================================

extern "C"
{
    // RMSNorm
    bool cudaOps_rmsnorm_fp32(
        const float *input, const float *gamma, float *output,
        int rows, int cols, float epsilon, int device_idx, void *stream);
    bool cudaOps_rmsnorm_bf16(
        const uint16_t *input, const float *gamma, uint16_t *output,
        int rows, int cols, float epsilon, int device_idx, void *stream);
    bool cudaOps_rmsnorm_fp16(
        const uint16_t *input, const float *gamma, uint16_t *output,
        int rows, int cols, float epsilon, int device_idx, void *stream);

    // SwiGLU
    bool cudaOps_swiglu_fp32(
        const float *gate, const float *up, float *output,
        int size, int device_idx, void *stream);
    bool cudaOps_swiglu_bf16(
        const uint16_t *gate, const uint16_t *up, uint16_t *output,
        int size, int device_idx, void *stream);
    bool cudaOps_swiglu_fp16(
        const uint16_t *gate, const uint16_t *up, uint16_t *output,
        int size, int device_idx, void *stream);

    // RoPE WORKSPACE-AWARE (v3 - external inv_freq buffer)
    bool cudaOps_rope_populate_inv_freq(
        float *d_inv_freq, int head_dim, float freq_base, int device_idx, cudaStream_t stream);
    bool cudaOps_rope_fp32_v3(
        float *Q, float *K, const float *d_inv_freq, const int *position_ids,
        int seq_len, int n_heads, int n_kv_heads, int head_dim, int rotary_dim, int device_idx, cudaStream_t stream);
    bool cudaOps_rope_fp32_decode_v3(
        float *Q, float *K, const float *d_inv_freq, int pos,
        int n_heads, int n_kv_heads, int head_dim, int rotary_dim, int device_idx, cudaStream_t stream);
    bool cudaOps_rope_fp32_contiguous_v3(
        float *Q, float *K, const float *d_inv_freq, int pos_offset, int seq_len,
        int n_heads, int n_kv_heads, int head_dim, int rotary_dim, int device_idx, cudaStream_t stream,
        const llaminar2::rope::RoPEDeviceParams *device_params = nullptr);
    bool cudaOps_rope_bf16_v3(
        uint16_t *Q, uint16_t *K, const float *d_inv_freq, const int *position_ids,
        int seq_len, int n_heads, int n_kv_heads, int head_dim, int rotary_dim, int device_idx, cudaStream_t stream);
    bool cudaOps_rope_bf16_decode_v3(
        uint16_t *Q, uint16_t *K, const float *d_inv_freq, int pos,
        int n_heads, int n_kv_heads, int head_dim, int rotary_dim, int device_idx, cudaStream_t stream);
    bool cudaOps_rope_bf16_contiguous_v3(
        uint16_t *Q, uint16_t *K, const float *d_inv_freq, int pos_offset, int seq_len,
        int n_heads, int n_kv_heads, int head_dim, int rotary_dim, int device_idx, cudaStream_t stream,
        const llaminar2::rope::RoPEDeviceParams *device_params = nullptr);
    bool cudaOps_rope_fp16_v3(
        uint16_t *Q, uint16_t *K, const float *d_inv_freq, const int *position_ids,
        int seq_len, int n_heads, int n_kv_heads, int head_dim, int rotary_dim, int device_idx, cudaStream_t stream);
    bool cudaOps_rope_fp16_decode_v3(
        uint16_t *Q, uint16_t *K, const float *d_inv_freq, int pos,
        int n_heads, int n_kv_heads, int head_dim, int rotary_dim, int device_idx, cudaStream_t stream);
    bool cudaOps_rope_fp16_contiguous_v3(
        uint16_t *Q, uint16_t *K, const float *d_inv_freq, int pos_offset, int seq_len,
        int n_heads, int n_kv_heads, int head_dim, int rotary_dim, int device_idx, cudaStream_t stream,
        const llaminar2::rope::RoPEDeviceParams *device_params = nullptr);

    // Embedding lookup - FP32
    cudaError_t launch_embedding_lookup(
        const float *embed_data,
        const int *token_ids,
        float *output,
        int num_tokens,
        int d_model,
        int vocab_size,
        int vocab_offset,
        cudaStream_t stream);

    // Embedding lookup - EmbedQ8 (universal quantized format)
    cudaError_t launch_embedding_lookup_q8(
        const void *embed_q8,
        const int *token_ids,
        float *output,
        int num_tokens,
        int d_model,
        int blocks_per_row,
        int vocab_size,
        int vocab_offset,
        cudaStream_t stream);
}

namespace
{
    bool validateCudaPointerForDevice(const void *ptr,
                                      int expected_device,
                                      const char *ptr_name,
                                      bool fail_on_query_error)
    {
        if (!ptr)
        {
            LOG_ERROR("[CUDAEmbeddingKernelT] " << ptr_name << " is null");
            return false;
        }

        cudaPointerAttributes attr{};
        const cudaError_t attr_err = cudaPointerGetAttributes(&attr, ptr);
        if (attr_err != cudaSuccess)
        {
            cudaGetLastError();
            if (fail_on_query_error)
            {
                LOG_ERROR("[CUDAEmbeddingKernelT] Failed to query pointer attributes for "
                          << ptr_name
                          << " ptr=" << ptr
                          << " err=" << cudaGetErrorString(attr_err)
                          << " expected_device=" << expected_device);
                return false;
            }
            return true;
        }

        const bool memory_type_ok =
            attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;
        if (!memory_type_ok || attr.device != expected_device)
        {
            LOG_ERROR("[CUDAEmbeddingKernelT] " << ptr_name
                                                << " buffer has incompatible CUDA pointer attributes: ptr=" << ptr
                                                << " attr.device=" << attr.device
                                                << " expected=" << expected_device
                                                << " attr.type=" << static_cast<int>(attr.type)
                                                << " device_ptr=" << attr.devicePointer
                                                << " host_ptr=" << attr.hostPointer);
            return false;
        }

        return true;
    }

    bool validateCudaTokenIdsHost(const int *token_ids,
                                  int num_tokens,
                                  int vocab_size,
                                  bool fail_on_invalid)
    {
        if (!token_ids || num_tokens <= 0)
        {
            LOG_ERROR("[CUDAEmbeddingKernelT] Invalid token buffer on host: token_ids="
                      << token_ids << " num_tokens=" << num_tokens);
            return false;
        }

        int min_id = std::numeric_limits<int>::max();
        int max_id = std::numeric_limits<int>::min();
        int first_invalid_pos = -1;
        int first_invalid_id = -1;

        for (int i = 0; i < num_tokens; ++i)
        {
            const int value = token_ids[i];
            min_id = std::min(min_id, value);
            max_id = std::max(max_id, value);
            const bool has_upper_bound = vocab_size > 0;
            if ((value < 0 || (has_upper_bound && value >= vocab_size)) && first_invalid_pos < 0)
            {
                first_invalid_pos = i;
                first_invalid_id = value;
            }
        }

        LOG_DEBUG("[CUDAEmbeddingKernelT] Host token stats: num_tokens=" << num_tokens
                                                                         << " vocab_size=" << (vocab_size > 0 ? std::to_string(vocab_size) : std::string("unchecked"))
                                                                         << " min_id=" << min_id
                                                                         << " max_id=" << max_id
                                                                         << " first_id=" << token_ids[0]
                                                                         << " last_id=" << token_ids[num_tokens - 1]);

        if (first_invalid_pos >= 0)
        {
            LOG_ERROR("[CUDAEmbeddingKernelT] Host token out of range at pos="
                      << first_invalid_pos
                      << " token_id=" << first_invalid_id
                      << " vocab_size=" << vocab_size);
            return !fail_on_invalid;
        }

        return true;
    }

    bool checkCudaNoPriorError(const char *context)
    {
        const cudaError_t prior = cudaPeekAtLastError();
        if (prior == cudaSuccess)
            return true;

        LOG_ERROR("[CUDAEmbeddingKernelT] Pre-existing CUDA error before "
                  << context << ": " << cudaGetErrorString(prior));
        return false;
    }

    bool isCudaStreamCapturing(cudaStream_t stream, const char *context)
    {
        if (llaminar2::isGraphCaptureActive())
            return true;

        if (!stream)
            return false;

        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        const cudaError_t err = cudaStreamIsCapturing(stream, &status);
        if (err != cudaSuccess)
        {
            llaminar2::Logger::getInstance().log(llaminar2::LogLevel::ERROR,
                                                 std::string("[") + context + "] cudaStreamIsCapturing failed: " +
                                                     cudaGetErrorString(err));
            return true;
        }
        return status == cudaStreamCaptureStatusActive;
    }

    bool uploadCudaRoPEDeviceParams(
        llaminar2::DeviceWorkspaceManager *workspace,
        llaminar2::rope::RoPEDeviceParams *host_params,
        cudaStream_t stream,
        bool &device_valid,
        int &device_offset,
        int pos_offset,
        const char *context)
    {
        device_valid = false;

        if (!stream)
        {
            LOG_ERROR("[" << context << "] Cannot upload RoPE params on a null/default CUDA stream");
            return false;
        }
        if (!workspace)
        {
            LOG_ERROR("[" << context << "] Cannot upload RoPE params without a bound workspace");
            return false;
        }
        if (isCudaStreamCapturing(stream, context))
        {
            LOG_ERROR("[" << context << "] Refusing to record RoPE-param H2D inside CUDA graph capture");
            return false;
        }

        auto *d_params = workspace->getBuffer(llaminar2::RoPEWorkspaceBuffers::DEVICE_PARAMS);
        if (!d_params)
        {
            LOG_ERROR("[" << context << "] Missing workspace buffer "
                          << llaminar2::RoPEWorkspaceBuffers::DEVICE_PARAMS);
            return false;
        }

        const cudaError_t copy_err =
            cudaMemcpyAsync(d_params, host_params, sizeof(llaminar2::rope::RoPEDeviceParams),
                            cudaMemcpyHostToDevice, stream);
        if (copy_err != cudaSuccess)
        {
            LOG_ERROR("[" << context << "] cudaMemcpyAsync failed for RoPE params: "
                          << cudaGetErrorString(copy_err));
            return false;
        }

        device_offset = pos_offset;
        device_valid = true;
        return true;
    }
} // namespace

namespace llaminar2
{

    CUDAEmbeddingKernelT::~CUDAEmbeddingKernelT()
    {
        if (h_token_ids_)
        {
            cudaFreeHost(h_token_ids_);
            h_token_ids_ = nullptr;
        }
    }
    namespace cuda
    {

        // =========================================================================
        // CUDARMSNormKernelT<FP32> Implementation
        // =========================================================================

        bool CUDARMSNormKernelT<ActivationPrecision::FP32>::apply(
            const float *input, const float *weight, float *output,
            int rows, int cols,
            float epsilon,
            bool use_bf16,
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            (void)use_bf16;
            (void)mpi_ctx;
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            return apply_typed(input, weight, output, rows, cols, epsilon, dev);
        }

        bool CUDARMSNormKernelT<ActivationPrecision::FP32>::apply_tensor(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float epsilon,
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            (void)mpi_ctx;
            if (!input || !weight || !output)
                return false;
            if (input->native_type() != TensorType::FP32 || output->native_type() != TensorType::FP32)
                return false;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            // Cast to FP32Tensor for GPU operations
            auto *input_fp32 = const_cast<FP32Tensor *>(dynamic_cast<const FP32Tensor *>(input));
            auto *weight_fp32 = const_cast<FP32Tensor *>(dynamic_cast<const FP32Tensor *>(weight));
            auto *output_fp32 = dynamic_cast<FP32Tensor *>(output);

            if (!input_fp32 || !weight_fp32 || !output_fp32)
                return false;

            // Coherence handled automatically by DeviceGraphExecutor

            // Get device pointers
            const float *d_input = static_cast<const float *>(input_fp32->gpu_data_ptr());
            const float *d_weight = static_cast<const float *>(weight_fp32->gpu_data_ptr());
            float *d_output = static_cast<float *>(output_fp32->gpu_data_ptr());

            // Launch kernel asynchronously - no sync needed since all ops are on default stream
            // Stream ordering guarantees subsequent kernels wait for this one
            CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::RMS_NORM, gpu_stream_);
            return cudaOps_rmsnorm_fp32(d_input, d_weight, d_output, rows, cols, epsilon, dev, gpu_stream_);
        }

        bool CUDARMSNormKernelT<ActivationPrecision::FP32>::apply_typed(
            const float *input,
            const float *gamma,
            float *output,
            int rows, int cols,
            float epsilon,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            bool ok = cudaOps_rmsnorm_fp32(input, gamma, output, rows, cols, epsilon, dev, gpu_stream_);
            if (ok)
                cudaDeviceSynchronize();
            return ok;
        }

        // =========================================================================
        // CUDARMSNormKernelT<BF16> Implementation
        // =========================================================================

        bool CUDARMSNormKernelT<ActivationPrecision::BF16>::apply_bf16(
            const uint16_t *input, const float *weight, uint16_t *output,
            int rows, int cols, float epsilon, int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            return apply_typed(input, weight, output, rows, cols, epsilon, dev);
        }

        bool CUDARMSNormKernelT<ActivationPrecision::BF16>::apply_tensor(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float epsilon,
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            (void)mpi_ctx;
            if (!input || !weight || !output)
                return false;
            if (input->native_type() != TensorType::BF16 || output->native_type() != TensorType::BF16)
                return false;

            auto *in_bf16 = const_cast<BF16Tensor *>(dynamic_cast<const BF16Tensor *>(input));
            auto *weight_fp32 = const_cast<FP32Tensor *>(dynamic_cast<const FP32Tensor *>(weight));
            auto *out_bf16 = dynamic_cast<BF16Tensor *>(output);
            if (!in_bf16 || !weight_fp32 || !out_bf16)
            {
                LOG_ERROR("[CUDARMSNormKernelT<BF16>] Dynamic cast failed - weight must be FP32");
                return false;
            }

            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            // Coherence handled automatically by DeviceGraphExecutor

            // Get device pointers - use gpu_data_ptr() for proper GPU pointer handling
            const uint16_t *d_input = static_cast<const uint16_t *>(in_bf16->gpu_data_ptr());
            const float *d_weight = static_cast<const float *>(weight_fp32->gpu_data_ptr());
            uint16_t *d_output = static_cast<uint16_t *>(out_bf16->gpu_data_ptr());

            // No sync needed - DeviceGraphExecutor handles async execution via stream ordering
            return cudaOps_rmsnorm_bf16(d_input, d_weight, d_output, rows, cols, epsilon, dev, gpu_stream_);
        }

        bool CUDARMSNormKernelT<ActivationPrecision::BF16>::apply_typed(
            const uint16_t *input,
            const float *gamma,
            uint16_t *output,
            int rows, int cols,
            float epsilon,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            bool ok = cudaOps_rmsnorm_bf16(input, gamma, output, rows, cols, epsilon, dev, gpu_stream_);
            if (ok)
                cudaDeviceSynchronize();
            return ok;
        }

        // =========================================================================
        // CUDARMSNormKernelT<FP16> Implementation
        // =========================================================================

        bool CUDARMSNormKernelT<ActivationPrecision::FP16>::apply_fp16(
            const uint16_t *input, const float *weight, uint16_t *output,
            int rows, int cols, float epsilon, int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            return apply_typed(input, weight, output, rows, cols, epsilon, dev);
        }

        bool CUDARMSNormKernelT<ActivationPrecision::FP16>::apply_tensor(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float epsilon,
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            (void)mpi_ctx;
            if (!input || !weight || !output)
                return false;
            if (input->native_type() != TensorType::FP16 || output->native_type() != TensorType::FP16)
                return false;

            auto *in_fp16 = const_cast<FP16Tensor *>(dynamic_cast<const FP16Tensor *>(input));
            auto *weight_fp32 = const_cast<FP32Tensor *>(dynamic_cast<const FP32Tensor *>(weight));
            auto *out_fp16 = dynamic_cast<FP16Tensor *>(output);
            if (!in_fp16 || !weight_fp32 || !out_fp16)
            {
                LOG_ERROR("[CUDARMSNormKernelT<FP16>] Dynamic cast failed - weight must be FP32");
                return false;
            }

            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            // Coherence handled automatically by DeviceGraphExecutor

            // Get device pointers - use gpu_data_ptr() for proper GPU pointer handling
            const uint16_t *d_input = static_cast<const uint16_t *>(in_fp16->gpu_data_ptr());
            const float *d_weight = static_cast<const float *>(weight_fp32->gpu_data_ptr());
            uint16_t *d_output = static_cast<uint16_t *>(out_fp16->gpu_data_ptr());

            // No sync needed - DeviceGraphExecutor handles async execution via stream ordering
            return cudaOps_rmsnorm_fp16(d_input, d_weight, d_output, rows, cols, epsilon, dev, gpu_stream_);
        }

        bool CUDARMSNormKernelT<ActivationPrecision::FP16>::apply_typed(
            const uint16_t *input,
            const float *gamma,
            uint16_t *output,
            int rows, int cols,
            float epsilon,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            bool ok = cudaOps_rmsnorm_fp16(input, gamma, output, rows, cols, epsilon, dev, gpu_stream_);
            if (ok)
                cudaDeviceSynchronize();
            return ok;
        }

        // =========================================================================
        // CUDASwiGLUKernelT<FP32> Implementation
        // =========================================================================

        bool CUDASwiGLUKernelT<ActivationPrecision::FP32>::apply(
            const float *gate, const float *up, float *output,
            int rows, int cols,
            bool add_residual,
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            (void)add_residual; // TODO: implement residual addition
            (void)mpi_ctx;
            int size = rows * cols;
            return apply_typed(gate, up, output, size);
        }

        bool CUDASwiGLUKernelT<ActivationPrecision::FP32>::apply_tensor(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int rows, int cols,
            bool add_residual,
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            (void)add_residual;
            (void)mpi_ctx;
            if (!gate || !up || !output)
                return false;
            if (gate->native_type() != TensorType::FP32 ||
                up->native_type() != TensorType::FP32 ||
                output->native_type() != TensorType::FP32)
                return false;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            // Cast to FP32Tensor for GPU operations
            auto *gate_fp32 = const_cast<FP32Tensor *>(dynamic_cast<const FP32Tensor *>(gate));
            auto *up_fp32 = const_cast<FP32Tensor *>(dynamic_cast<const FP32Tensor *>(up));
            auto *output_fp32 = dynamic_cast<FP32Tensor *>(output);

            if (!gate_fp32 || !up_fp32 || !output_fp32)
                return false;

            // Coherence handled automatically by DeviceGraphExecutor

            // Get device pointers
            const float *d_gate = static_cast<const float *>(gate_fp32->gpu_data_ptr());
            const float *d_up = static_cast<const float *>(up_fp32->gpu_data_ptr());
            float *d_output = static_cast<float *>(output_fp32->gpu_data_ptr());

            int size = rows * cols;
            // Launch kernel asynchronously - stream ordering handles dependencies
            CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::SWIGLU, gpu_stream_);
            return cudaOps_swiglu_fp32(d_gate, d_up, d_output, size, dev, gpu_stream_);
        }

        bool CUDASwiGLUKernelT<ActivationPrecision::FP32>::apply_typed(
            const float *gate,
            const float *up,
            float *output,
            int size,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            // Launch kernel asynchronously
            return cudaOps_swiglu_fp32(gate, up, output, size, dev, gpu_stream_);
        }

        // =========================================================================
        // CUDASwiGLUKernelT<BF16> Implementation
        // =========================================================================

        bool CUDASwiGLUKernelT<ActivationPrecision::BF16>::apply_bf16(
            const uint16_t *gate, const uint16_t *up, uint16_t *output,
            int rows, int cols,
            bool add_residual,
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            (void)add_residual;
            (void)mpi_ctx;
            (void)device_idx;
            int size = rows * cols;
            return apply_typed(gate, up, output, size);
        }

        bool CUDASwiGLUKernelT<ActivationPrecision::BF16>::apply_tensor(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int rows, int cols,
            bool add_residual,
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            (void)add_residual;
            (void)mpi_ctx;
            if (!gate || !up || !output)
                return false;
            if (gate->native_type() != TensorType::BF16 ||
                up->native_type() != TensorType::BF16 ||
                output->native_type() != TensorType::BF16)
                return false;

            auto *gate_bf16 = const_cast<BF16Tensor *>(dynamic_cast<const BF16Tensor *>(gate));
            auto *up_bf16 = const_cast<BF16Tensor *>(dynamic_cast<const BF16Tensor *>(up));
            auto *out_bf16 = dynamic_cast<BF16Tensor *>(output);
            if (!gate_bf16 || !up_bf16 || !out_bf16)
                return false;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            // Coherence handled automatically by DeviceGraphExecutor

            // Get device pointers
            const uint16_t *d_gate = static_cast<const uint16_t *>(gate_bf16->gpu_data_ptr());
            const uint16_t *d_up = static_cast<const uint16_t *>(up_bf16->gpu_data_ptr());
            uint16_t *d_output = static_cast<uint16_t *>(out_bf16->gpu_data_ptr());

            int size = rows * cols;
            // No sync needed - DeviceGraphExecutor handles async execution via stream ordering
            return cudaOps_swiglu_bf16(d_gate, d_up, d_output, size, dev, gpu_stream_);
        }

        bool CUDASwiGLUKernelT<ActivationPrecision::BF16>::apply_typed(
            const uint16_t *gate,
            const uint16_t *up,
            uint16_t *output,
            int size,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            bool ok = cudaOps_swiglu_bf16(gate, up, output, size, dev, gpu_stream_);
            if (ok)
                cudaDeviceSynchronize();
            return ok;
        }

        // =========================================================================
        // CUDASwiGLUKernelT<FP16> Implementation
        // =========================================================================

        bool CUDASwiGLUKernelT<ActivationPrecision::FP16>::apply_fp16(
            const uint16_t *gate, const uint16_t *up, uint16_t *output,
            int rows, int cols,
            bool add_residual,
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            (void)add_residual;
            (void)mpi_ctx;
            (void)device_idx;
            int size = rows * cols;
            return apply_typed(gate, up, output, size);
        }

        bool CUDASwiGLUKernelT<ActivationPrecision::FP16>::apply_tensor(
            const TensorBase *gate,
            const TensorBase *up,
            TensorBase *output,
            int rows, int cols,
            bool add_residual,
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            (void)add_residual;
            (void)mpi_ctx;
            if (!gate || !up || !output)
                return false;
            if (gate->native_type() != TensorType::FP16 ||
                up->native_type() != TensorType::FP16 ||
                output->native_type() != TensorType::FP16)
                return false;

            auto *gate_fp16 = const_cast<FP16Tensor *>(dynamic_cast<const FP16Tensor *>(gate));
            auto *up_fp16 = const_cast<FP16Tensor *>(dynamic_cast<const FP16Tensor *>(up));
            auto *out_fp16 = dynamic_cast<FP16Tensor *>(output);
            if (!gate_fp16 || !up_fp16 || !out_fp16)
                return false;

            int dev = (device_idx >= 0) ? device_idx : device_idx_;

            // Coherence handled automatically by DeviceGraphExecutor

            // Get device pointers
            const uint16_t *d_gate = static_cast<const uint16_t *>(gate_fp16->gpu_data_ptr());
            const uint16_t *d_up = static_cast<const uint16_t *>(up_fp16->gpu_data_ptr());
            uint16_t *d_output = static_cast<uint16_t *>(out_fp16->gpu_data_ptr());

            int size = rows * cols;
            // No sync needed - DeviceGraphExecutor handles async execution via stream ordering
            return cudaOps_swiglu_fp16(d_gate, d_up, d_output, size, dev, gpu_stream_);
        }

        bool CUDASwiGLUKernelT<ActivationPrecision::FP16>::apply_typed(
            const uint16_t *gate,
            const uint16_t *up,
            uint16_t *output,
            int size,
            int device_idx)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            bool ok = cudaOps_swiglu_fp16(gate, up, output, size, dev, gpu_stream_);
            if (ok)
                cudaDeviceSynchronize();
            return ok;
        }

        // =========================================================================
        // CUDARoPEKernelT<FP32> Implementation
        // =========================================================================

        CUDARoPEKernelT<ActivationPrecision::FP32>::~CUDARoPEKernelT()
        {
            if (h_device_params_)
            {
                cudaFreeHost(h_device_params_);
                h_device_params_ = nullptr;
            }
        }

        void CUDARoPEKernelT<ActivationPrecision::FP32>::setDynamicPosOffset(int pos_offset)
        {
            if (!h_device_params_)
            {
                cudaMallocHost(reinterpret_cast<void **>(&h_device_params_), sizeof(rope::RoPEDeviceParams));
            }
            if (h_device_params_)
            {
                h_device_params_->pos_offset = pos_offset;
                uploadCudaRoPEDeviceParams(
                    workspace_, h_device_params_, static_cast<cudaStream_t>(gpu_stream_),
                    dynamic_pos_device_valid_, dynamic_pos_offset_, pos_offset,
                    "CUDARoPEKernelT<FP32>");
            }
        }

        bool CUDARoPEKernelT<ActivationPrecision::FP32>::apply_typed(
            float *Q,
            float *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            int device_idx,
            int pos_offset,
            int rotary_dim)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::ROPE, gpu_stream_);
            cudaStream_t stream = static_cast<cudaStream_t>(gpu_stream_);
            const bool sync_after = (stream == nullptr);

            // Effective rotary dimension: 0 means full rotation (=head_dim)
            const int eff_rotary = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;

            // Require workspace to be bound
            if (!workspace_)
            {
                LOG_ERROR("[CUDARoPEKernelT<FP32>] Workspace not bound. Call bindWorkspace() first.");
                return false;
            }

            float *d_inv_freq = static_cast<float *>(workspace_->getBuffer(RoPEWorkspaceBuffers::INV_FREQ));
            if (!d_inv_freq)
            {
                LOG_ERROR("[CUDARoPEKernelT<FP32>] INV_FREQ buffer not allocated in workspace");
                return false;
            }

            // Initialize inv_freq if needed (lazy initialization)
            // Use eff_rotary for frequency computation: inv_freq[i] = 1/(theta^(2i/eff_rotary))
            if (!inv_freq_initialized_ || inv_freq_head_dim_ != eff_rotary || inv_freq_theta_ != rope_theta)
            {
                if (!cudaOps_rope_populate_inv_freq(d_inv_freq, eff_rotary, rope_theta, dev, stream))
                {
                    LOG_ERROR("[CUDARoPEKernelT<FP32>] Failed to populate inv_freq");
                    return false;
                }
                inv_freq_initialized_ = true;
                inv_freq_head_dim_ = eff_rotary;
                inv_freq_theta_ = rope_theta;
            }

            const bool force_device_positions = (gpu_stream_ != nullptr && position_ids != nullptr);

            // ZERO-COPY PATH: If position_ids is nullptr, use contiguous kernel
            // GRAPH CAPTURE: Skip decode fast path when gpu_stream_ is set — the scalar `pos`
            // argument would be frozen in the captured graph. Fall through to contiguous path
            // which uses pre-uploaded device_params for graph replay.
            if (seq_len == 1 && !force_device_positions && !gpu_stream_)
            {
                int pos = position_ids ? position_ids[0] : pos_offset;
                bool ok = cudaOps_rope_fp32_decode_v3(Q, K, d_inv_freq, pos,
                                                      n_heads, n_kv_heads, head_dim, eff_rotary, dev, stream);
                if (ok && sync_after)
                    cudaDeviceSynchronize();
                return ok;
            }

            {
                bool is_contiguous = (position_ids == nullptr) && !force_device_positions;
                if (!is_contiguous && position_ids)
                {
                    is_contiguous = true;
                    for (int i = 0; i < seq_len; ++i)
                    {
                        if (position_ids[i] != pos_offset + i)
                        {
                            is_contiguous = false;
                            break;
                        }
                    }
                }
                if (is_contiguous)
                {
                    // For graph capture/replay, use the pre-uploaded device params buffer
                    // so pos_offset can change without recording H2D nodes.
                    const rope::RoPEDeviceParams *d_params = nullptr;
                    if (gpu_stream_ && workspace_)
                    {
                        d_params = static_cast<const rope::RoPEDeviceParams *>(
                            workspace_->getBuffer(RoPEWorkspaceBuffers::DEVICE_PARAMS));
                        if (!d_params)
                        {
                            LOG_ERROR("[CUDARoPEKernelT<FP32>] Missing workspace buffer "
                                      << RoPEWorkspaceBuffers::DEVICE_PARAMS);
                            return false;
                        }
                        if (isGraphCaptureActive())
                        {
                            if (!dynamic_pos_device_valid_ || dynamic_pos_offset_ != pos_offset)
                            {
                                LOG_ERROR("[CUDARoPEKernelT<FP32>] RoPE device params were not ready before CUDA graph capture"
                                          << " requested_pos=" << pos_offset
                                          << " prepared_pos=" << dynamic_pos_offset_
                                          << " device_valid=" << dynamic_pos_device_valid_);
                                return false;
                            }
                        }
                        else if (!dynamic_pos_device_valid_ || dynamic_pos_offset_ != pos_offset)
                        {
                            setDynamicPosOffset(pos_offset);
                            if (!dynamic_pos_device_valid_)
                                return false;
                        }
                    }
                    bool ok = cudaOps_rope_fp32_contiguous_v3(Q, K, d_inv_freq, pos_offset, seq_len,
                                                              n_heads, n_kv_heads, head_dim, eff_rotary, dev, stream, d_params);
                    if (ok && sync_after)
                        cudaDeviceSynchronize();
                    return ok;
                }
            }

            int *d_position_ids = static_cast<int *>(workspace_->getBuffer(RoPEWorkspaceBuffers::POSITION_IDS));
            if (!d_position_ids)
            {
                LOG_ERROR("[CUDARoPEKernelT<FP32>] POSITION_IDS buffer not allocated in workspace");
                return false;
            }

            size_t pos_bytes = static_cast<size_t>(seq_len) * sizeof(int);
            if (isGraphCaptureActive())
            {
                LOG_ERROR("[CUDARoPEKernelT<FP32>] Refusing to record position_ids H2D inside CUDA graph capture");
                return false;
            }
            cudaError_t copy_err = cudaMemcpyAsync(d_position_ids, position_ids, pos_bytes,
                                                   cudaMemcpyHostToDevice, stream);
            if (copy_err != cudaSuccess)
            {
                LOG_ERROR("[CUDARoPEKernelT<FP32>] Failed to copy position_ids to GPU: "
                          << cudaGetErrorString(copy_err));
                return false;
            }

            bool ok = cudaOps_rope_fp32_v3(Q, K, d_inv_freq, d_position_ids, seq_len,
                                           n_heads, n_kv_heads, head_dim, eff_rotary, dev, stream);
            if (ok && sync_after)
                cudaDeviceSynchronize();
            return ok;
        }

        // =========================================================================
        // CUDARoPEKernelT<BF16> Implementation
        // =========================================================================

        CUDARoPEKernelT<ActivationPrecision::BF16>::~CUDARoPEKernelT()
        {
            if (h_device_params_)
            {
                cudaFreeHost(h_device_params_);
                h_device_params_ = nullptr;
            }
        }

        void CUDARoPEKernelT<ActivationPrecision::BF16>::setDynamicPosOffset(int pos_offset)
        {
            if (!h_device_params_)
            {
                cudaMallocHost(reinterpret_cast<void **>(&h_device_params_), sizeof(rope::RoPEDeviceParams));
            }
            if (h_device_params_)
            {
                h_device_params_->pos_offset = pos_offset;
                uploadCudaRoPEDeviceParams(
                    workspace_, h_device_params_, static_cast<cudaStream_t>(gpu_stream_),
                    dynamic_pos_device_valid_, dynamic_pos_offset_, pos_offset,
                    "CUDARoPEKernelT<BF16>");
            }
        }

        bool CUDARoPEKernelT<ActivationPrecision::BF16>::apply_typed(
            uint16_t *Q,
            uint16_t *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            int device_idx,
            int pos_offset,
            int rotary_dim)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            cudaStream_t stream = static_cast<cudaStream_t>(gpu_stream_);
            const bool sync_after = (stream == nullptr);

            // Effective rotary dimension: 0 means full rotation (=head_dim)
            const int eff_rotary = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;

            // Require workspace to be bound
            if (!workspace_)
            {
                LOG_ERROR("[CUDARoPEKernelT<BF16>] Workspace not bound. Call bindWorkspace() first.");
                return false;
            }

            float *d_inv_freq = static_cast<float *>(workspace_->getBuffer(RoPEWorkspaceBuffers::INV_FREQ));
            if (!d_inv_freq)
            {
                LOG_ERROR("[CUDARoPEKernelT<BF16>] INV_FREQ buffer not allocated in workspace");
                return false;
            }

            // Initialize inv_freq if needed (lazy initialization)
            if (!inv_freq_initialized_ || inv_freq_head_dim_ != eff_rotary || inv_freq_theta_ != rope_theta)
            {
                if (!cudaOps_rope_populate_inv_freq(d_inv_freq, eff_rotary, rope_theta, dev, stream))
                {
                    LOG_ERROR("[CUDARoPEKernelT<BF16>] Failed to populate inv_freq");
                    return false;
                }
                inv_freq_initialized_ = true;
                inv_freq_head_dim_ = eff_rotary;
                inv_freq_theta_ = rope_theta;
            }

            const bool force_device_positions = (gpu_stream_ != nullptr && position_ids != nullptr);

            // GRAPH CAPTURE: Skip decode fast path when gpu_stream_ is set — scalar pos frozen in graph.
            if (seq_len == 1 && !force_device_positions && !gpu_stream_)
            {
                int pos = position_ids ? position_ids[0] : pos_offset;
                bool ok = cudaOps_rope_bf16_decode_v3(Q, K, d_inv_freq, pos,
                                                      n_heads, n_kv_heads, head_dim, eff_rotary, dev, stream);
                if (ok && sync_after)
                    cudaDeviceSynchronize();
                return ok;
            }

            {
                bool is_contiguous = (position_ids == nullptr) && !force_device_positions;
                if (!is_contiguous && position_ids)
                {
                    is_contiguous = true;
                    for (int i = 0; i < seq_len; ++i)
                    {
                        if (position_ids[i] != pos_offset + i)
                        {
                            is_contiguous = false;
                            break;
                        }
                    }
                }
                if (is_contiguous)
                {
                    const rope::RoPEDeviceParams *d_params = nullptr;
                    if (gpu_stream_ && workspace_)
                    {
                        d_params = static_cast<const rope::RoPEDeviceParams *>(
                            workspace_->getBuffer(RoPEWorkspaceBuffers::DEVICE_PARAMS));
                        if (!d_params)
                        {
                            LOG_ERROR("[CUDARoPEKernelT<BF16>] Missing workspace buffer "
                                      << RoPEWorkspaceBuffers::DEVICE_PARAMS);
                            return false;
                        }
                        if (isGraphCaptureActive())
                        {
                            if (!dynamic_pos_device_valid_ || dynamic_pos_offset_ != pos_offset)
                            {
                                LOG_ERROR("[CUDARoPEKernelT<BF16>] RoPE device params were not ready before CUDA graph capture"
                                          << " requested_pos=" << pos_offset
                                          << " prepared_pos=" << dynamic_pos_offset_
                                          << " device_valid=" << dynamic_pos_device_valid_);
                                return false;
                            }
                        }
                        else if (!dynamic_pos_device_valid_ || dynamic_pos_offset_ != pos_offset)
                        {
                            setDynamicPosOffset(pos_offset);
                            if (!dynamic_pos_device_valid_)
                                return false;
                        }
                    }
                    bool ok = cudaOps_rope_bf16_contiguous_v3(Q, K, d_inv_freq, pos_offset, seq_len,
                                                              n_heads, n_kv_heads, head_dim, eff_rotary, dev, stream, d_params);
                    if (ok && sync_after)
                        cudaDeviceSynchronize();
                    return ok;
                }
            }

            int *d_position_ids = static_cast<int *>(workspace_->getBuffer(RoPEWorkspaceBuffers::POSITION_IDS));
            if (!d_position_ids)
            {
                LOG_ERROR("[CUDARoPEKernelT<BF16>] POSITION_IDS buffer not allocated in workspace");
                return false;
            }

            size_t pos_bytes = static_cast<size_t>(seq_len) * sizeof(int);
            if (isGraphCaptureActive())
            {
                LOG_ERROR("[CUDARoPEKernelT<BF16>] Refusing to record position_ids H2D inside CUDA graph capture");
                return false;
            }
            cudaError_t copy_err = cudaMemcpyAsync(d_position_ids, position_ids, pos_bytes,
                                                   cudaMemcpyHostToDevice, stream);
            if (copy_err != cudaSuccess)
            {
                LOG_ERROR("[CUDARoPEKernelT<BF16>] Failed to copy position_ids to GPU: "
                          << cudaGetErrorString(copy_err));
                return false;
            }

            bool ok = cudaOps_rope_bf16_v3(Q, K, d_inv_freq, d_position_ids, seq_len,
                                           n_heads, n_kv_heads, head_dim, eff_rotary, dev, stream);
            if (ok && sync_after)
                cudaDeviceSynchronize();
            return ok;
        }

        // =========================================================================
        // CUDARoPEKernelT<FP16> Implementation
        // =========================================================================

        CUDARoPEKernelT<ActivationPrecision::FP16>::~CUDARoPEKernelT()
        {
            if (h_device_params_)
            {
                cudaFreeHost(h_device_params_);
                h_device_params_ = nullptr;
            }
        }

        void CUDARoPEKernelT<ActivationPrecision::FP16>::setDynamicPosOffset(int pos_offset)
        {
            if (!h_device_params_)
            {
                cudaMallocHost(reinterpret_cast<void **>(&h_device_params_), sizeof(rope::RoPEDeviceParams));
            }
            if (h_device_params_)
            {
                h_device_params_->pos_offset = pos_offset;
                uploadCudaRoPEDeviceParams(
                    workspace_, h_device_params_, static_cast<cudaStream_t>(gpu_stream_),
                    dynamic_pos_device_valid_, dynamic_pos_offset_, pos_offset,
                    "CUDARoPEKernelT<FP16>");
            }
        }

        bool CUDARoPEKernelT<ActivationPrecision::FP16>::apply_typed(
            uint16_t *Q,
            uint16_t *K,
            const int *position_ids,
            int seq_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            float rope_theta,
            int device_idx,
            int pos_offset,
            int rotary_dim)
        {
            int dev = (device_idx >= 0) ? device_idx : device_idx_;
            cudaStream_t stream = static_cast<cudaStream_t>(gpu_stream_);
            const bool sync_after = (stream == nullptr);

            // Effective rotary dimension: 0 means full rotation (=head_dim)
            const int eff_rotary = (rotary_dim > 0 && rotary_dim < head_dim) ? rotary_dim : head_dim;

            // Require workspace to be bound
            if (!workspace_)
            {
                LOG_ERROR("[CUDARoPEKernelT<FP16>] Workspace not bound. Call bindWorkspace() first.");
                return false;
            }

            float *d_inv_freq = static_cast<float *>(workspace_->getBuffer(RoPEWorkspaceBuffers::INV_FREQ));
            if (!d_inv_freq)
            {
                LOG_ERROR("[CUDARoPEKernelT<FP16>] INV_FREQ buffer not allocated in workspace");
                return false;
            }

            // Initialize inv_freq if needed (lazy initialization)
            if (!inv_freq_initialized_ || inv_freq_head_dim_ != eff_rotary || inv_freq_theta_ != rope_theta)
            {
                if (!cudaOps_rope_populate_inv_freq(d_inv_freq, eff_rotary, rope_theta, dev, stream))
                {
                    LOG_ERROR("[CUDARoPEKernelT<FP16>] Failed to populate inv_freq");
                    return false;
                }
                inv_freq_initialized_ = true;
                inv_freq_head_dim_ = eff_rotary;
                inv_freq_theta_ = rope_theta;
            }

            const bool force_device_positions = (gpu_stream_ != nullptr && position_ids != nullptr);

            // GRAPH CAPTURE: Skip decode fast path when gpu_stream_ is set — scalar pos frozen in graph.
            if (seq_len == 1 && !force_device_positions && !gpu_stream_)
            {
                int pos = position_ids ? position_ids[0] : pos_offset;
                bool ok = cudaOps_rope_fp16_decode_v3(Q, K, d_inv_freq, pos,
                                                      n_heads, n_kv_heads, head_dim, eff_rotary, dev, stream);
                if (ok && sync_after)
                    cudaDeviceSynchronize();
                return ok;
            }

            {
                bool is_contiguous = (position_ids == nullptr) && !force_device_positions;
                if (!is_contiguous && position_ids)
                {
                    is_contiguous = true;
                    for (int i = 0; i < seq_len; ++i)
                    {
                        if (position_ids[i] != pos_offset + i)
                        {
                            is_contiguous = false;
                            break;
                        }
                    }
                }
                if (is_contiguous)
                {
                    const rope::RoPEDeviceParams *d_params = nullptr;
                    if (gpu_stream_ && workspace_)
                    {
                        d_params = static_cast<const rope::RoPEDeviceParams *>(
                            workspace_->getBuffer(RoPEWorkspaceBuffers::DEVICE_PARAMS));
                        if (!d_params)
                        {
                            LOG_ERROR("[CUDARoPEKernelT<FP16>] Missing workspace buffer "
                                      << RoPEWorkspaceBuffers::DEVICE_PARAMS);
                            return false;
                        }
                        if (isGraphCaptureActive())
                        {
                            if (!dynamic_pos_device_valid_ || dynamic_pos_offset_ != pos_offset)
                            {
                                LOG_ERROR("[CUDARoPEKernelT<FP16>] RoPE device params were not ready before CUDA graph capture"
                                          << " requested_pos=" << pos_offset
                                          << " prepared_pos=" << dynamic_pos_offset_
                                          << " device_valid=" << dynamic_pos_device_valid_);
                                return false;
                            }
                        }
                        else if (!dynamic_pos_device_valid_ || dynamic_pos_offset_ != pos_offset)
                        {
                            setDynamicPosOffset(pos_offset);
                            if (!dynamic_pos_device_valid_)
                                return false;
                        }
                    }
                    bool ok = cudaOps_rope_fp16_contiguous_v3(Q, K, d_inv_freq, pos_offset, seq_len,
                                                              n_heads, n_kv_heads, head_dim, eff_rotary, dev, stream, d_params);
                    if (ok && sync_after)
                        cudaDeviceSynchronize();
                    return ok;
                }
            }

            int *d_position_ids = static_cast<int *>(workspace_->getBuffer(RoPEWorkspaceBuffers::POSITION_IDS));
            if (!d_position_ids)
            {
                LOG_ERROR("[CUDARoPEKernelT<FP16>] POSITION_IDS buffer not allocated in workspace");
                return false;
            }

            size_t pos_bytes = static_cast<size_t>(seq_len) * sizeof(int);
            if (isGraphCaptureActive())
            {
                LOG_ERROR("[CUDARoPEKernelT<FP16>] Refusing to record position_ids H2D inside CUDA graph capture");
                return false;
            }
            cudaError_t copy_err = cudaMemcpyAsync(d_position_ids, position_ids, pos_bytes,
                                                   cudaMemcpyHostToDevice, stream);
            if (copy_err != cudaSuccess)
            {
                LOG_ERROR("[CUDARoPEKernelT<FP16>] Failed to copy position_ids to GPU: "
                          << cudaGetErrorString(copy_err));
                return false;
            }

            bool ok = cudaOps_rope_fp16_v3(Q, K, d_inv_freq, d_position_ids, seq_len,
                                           n_heads, n_kv_heads, head_dim, eff_rotary, dev, stream);
            if (ok && sync_after)
                cudaDeviceSynchronize();
            return ok;
        }

    } // namespace cuda

    // =========================================================================
    // CUDAEmbeddingKernelT Implementation (in llaminar2 namespace, not cuda)
    // =========================================================================

    bool CUDAEmbeddingKernelT::apply(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        float *output,
        const IMPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        int dev = (device_idx >= 0) ? device_idx : device_idx_;

        // Set device context before kernel launch (not capturable in CUDA graphs)
        if (!gpu_stream_)
        {
            cudaError_t set_err = cudaSetDevice(dev);
            if (set_err != cudaSuccess)
            {
                fprintf(stderr, "[CUDAEmbeddingKernelT] cudaSetDevice(%d) failed: %s\n",
                        dev, cudaGetErrorString(set_err));
                return false;
            }
        }

        CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::EMBEDDING_LOOKUP, gpu_stream_);
        const int launch_vocab_size = explicit_vocab_range_ && local_vocab_size_ > 0
                                          ? local_vocab_size_
                                          : INT_MAX;
        const int launch_vocab_offset = explicit_vocab_range_ ? vocab_offset_ : 0;
        cudaError_t err = launch_embedding_lookup(embed_data, token_ids, output,
                                                  num_tokens, d_model,
                                                  launch_vocab_size, launch_vocab_offset,
                                                  static_cast<cudaStream_t>(gpu_stream_));
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[CUDAEmbeddingKernelT] Kernel launch failed: %s\n",
                    cudaGetErrorString(err));
            return false;
        }
        // Async - stream ordering handles dependencies
        return true;
    }

    bool CUDAEmbeddingKernelT::apply_bf16(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        uint16_t *output,
        const IMPIContext *mpi_ctx,
        int device_idx)
    {
        (void)embed_data;
        (void)token_ids;
        (void)num_tokens;
        (void)d_model;
        (void)output;
        (void)mpi_ctx;
        (void)device_idx;
        fprintf(stderr, "[CUDAEmbeddingKernelT] BF16 output not yet implemented\n");
        return false;
    }

    bool CUDAEmbeddingKernelT::apply_fp16(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        uint16_t *output,
        const IMPIContext *mpi_ctx,
        int device_idx)
    {
        (void)embed_data;
        (void)token_ids;
        (void)num_tokens;
        (void)d_model;
        (void)output;
        (void)mpi_ctx;
        (void)device_idx;
        fprintf(stderr, "[CUDAEmbeddingKernelT] FP16 output not yet implemented\n");
        return false;
    }

    bool CUDAEmbeddingKernelT::apply_q8_1(
        const float *embed_data,
        const int *token_ids,
        int num_tokens,
        int d_model,
        void *output,
        const IMPIContext *mpi_ctx,
        int device_idx)
    {
        (void)embed_data;
        (void)token_ids;
        (void)num_tokens;
        (void)d_model;
        (void)output;
        (void)mpi_ctx;
        (void)device_idx;
        fprintf(stderr, "[CUDAEmbeddingKernelT] Q8_1 output not yet implemented\n");
        return false;
    }

    void CUDAEmbeddingKernelT::setDynamicTokenIds(const int *token_ids, int num_tokens)
    {
        dynamic_params_active_ = false;
        dynamic_token_count_ = 0;
        device_token_ids_active_ = false;
        device_token_ids_ = nullptr;
        device_token_count_ = 0;

        if (!token_ids || num_tokens <= 0)
        {
            return;
        }

        if (num_tokens > max_token_ids_)
        {
            if (h_token_ids_)
            {
                cudaFreeHost(h_token_ids_);
                h_token_ids_ = nullptr;
            }

            cudaError_t alloc_err = cudaMallocHost(reinterpret_cast<void **>(&h_token_ids_),
                                                   static_cast<size_t>(num_tokens) * sizeof(int));
            if (alloc_err != cudaSuccess)
            {
                fprintf(stderr, "[CUDAEmbeddingKernelT] Failed to allocate pinned token buffer: %s\n",
                        cudaGetErrorString(alloc_err));
                return;
            }
            max_token_ids_ = num_tokens;
        }

        std::memcpy(h_token_ids_, token_ids, static_cast<size_t>(num_tokens) * sizeof(int));

        if (!workspace_ || !workspace_->isAllocated())
        {
            return;
        }
        if (!gpu_stream_)
        {
            fprintf(stderr, "[CUDAEmbeddingKernelT] Dynamic token preload requires an explicit non-null stream\n");
            return;
        }

        int *d_token_ids = static_cast<int *>(workspace_->getBuffer(EmbeddingWorkspaceBuffers::TOKEN_IDS));
        if (!d_token_ids)
        {
            return;
        }

        cudaError_t copy_err = cudaMemcpyAsync(d_token_ids, h_token_ids_,
                                               static_cast<size_t>(num_tokens) * sizeof(int),
                                               cudaMemcpyHostToDevice,
                                               static_cast<cudaStream_t>(gpu_stream_));
        if (copy_err != cudaSuccess)
        {
            fprintf(stderr, "[CUDAEmbeddingKernelT] Failed to preload token_ids to GPU: %s\n",
                    cudaGetErrorString(copy_err));
            return;
        }

        dynamic_token_count_ = num_tokens;
        dynamic_params_active_ = true;
        preload_stream_ = gpu_stream_;
    }

    void CUDAEmbeddingKernelT::setDynamicDeviceTokenIds(
        const void *token_ids_device,
        int num_tokens)
    {
        dynamic_params_active_ = false;
        dynamic_token_count_ = 0;
        preload_stream_ = nullptr;
        device_token_ids_ = static_cast<const int *>(token_ids_device);
        device_token_count_ = num_tokens;
        device_token_ids_active_ = token_ids_device && num_tokens > 0;
    }

    void CUDAEmbeddingKernelT::resetDynamicState()
    {
        dynamic_params_active_ = false;
        dynamic_token_count_ = 0;
        device_token_ids_active_ = false;
        device_token_ids_ = nullptr;
        device_token_count_ = 0;
        preload_stream_ = nullptr;
        // h_token_ids_ buffer is preserved — it's reusable for the next session
    }

    bool CUDAEmbeddingKernelT::apply_tensor(
        const TensorBase *embed_table,
        const int *token_ids,
        int num_tokens,
        int d_model,
        TensorBase *output,
        const IMPIContext *mpi_ctx,
        int device_idx)
    {
        // Output must be FP32 (common for all embedding operations)
        if (output->native_type() != TensorType::FP32)
        {
            fprintf(stderr, "[CUDAEmbeddingKernelT] Output must be FP32 tensor\n");
            return false;
        }

        auto *output_fp32 = dynamic_cast<FP32Tensor *>(output);
        if (!output_fp32)
        {
            fprintf(stderr, "[CUDAEmbeddingKernelT] Output tensor cast to FP32 failed\n");
            return false;
        }

        // Determine target CUDA device and set context
        int dev = (device_idx >= 0) ? device_idx : device_idx_;
        if (!gpu_stream_)
        {
            cudaError_t set_err = cudaSetDevice(dev);
            if (set_err != cudaSuccess)
            {
                fprintf(stderr, "[CUDAEmbeddingKernelT] cudaSetDevice(%d) failed: %s\n",
                        dev, cudaGetErrorString(set_err));
                return false;
            }
        }

        // =====================================================================
        // Step 1: Get token_ids buffer from workspace and copy data
        // =====================================================================
        if (!hasWorkspace())
        {
            fprintf(stderr, "[CUDAEmbeddingKernelT] Workspace not bound - hot-path allocation disabled. "
                            "Call bindWorkspace() before apply_tensor()\n");
            return false;
        }

        int *workspace_token_ids = static_cast<int *>(workspace_->getBuffer(EmbeddingWorkspaceBuffers::TOKEN_IDS));
        if (!workspace_token_ids)
        {
            fprintf(stderr, "[CUDAEmbeddingKernelT] Workspace buffer '%s' not found\n",
                    EmbeddingWorkspaceBuffers::TOKEN_IDS);
            return false;
        }

        const bool use_device_token_ids =
            device_token_ids_active_ &&
            device_token_count_ == num_tokens &&
            device_token_ids_ != nullptr;
        int *d_token_ids = use_device_token_ids
                               ? const_cast<int *>(device_token_ids_)
                               : workspace_token_ids;

        const bool validate_gpu_ptrs = debugEnv().validation.validate_gpu_ptrs;
        if (!use_device_token_ids && validate_gpu_ptrs &&
            !validateCudaTokenIdsHost(token_ids, num_tokens, /*vocab_size=*/0, /*fail_on_invalid=*/true))
        {
            return false;
        }
        if (validate_gpu_ptrs &&
            !validateCudaPointerForDevice(d_token_ids, dev, "TOKEN_IDS", /*fail_on_query_error=*/true))
        {
            return false;
        }

        size_t token_bytes = static_cast<size_t>(num_tokens) * sizeof(int);
        cudaError_t err = cudaSuccess;
        // Verify preloaded data matches current request to prevent stale tokens
        // after clear_cache(). The kernel is cached in KernelFactory and
        // dynamic_params_active_ persists across graph rebuilds.
        // Also verify stream match: setDynamicTokenIds() may have run on a
        // different stream than the current gpu_stream_ if the graph capture
        // controller reassigned stage streams after updateDynamicParams().
        const bool token_ids_preloaded = dynamic_params_active_ &&
                                         dynamic_token_count_ == num_tokens &&
                                         preload_stream_ == gpu_stream_ &&
                                         token_ids &&
                                         h_token_ids_ &&
                                         std::memcmp(h_token_ids_, token_ids, token_bytes) == 0;
        if (!use_device_token_ids && !token_ids_preloaded)
        {
            if (isGraphCaptureActive())
            {
                fprintf(stderr,
                        "[CUDAEmbeddingKernelT] Token IDs were not preloaded before graph capture "
                        "(num_tokens=%d, stream=%p, active=%d, cached_tokens=%d, preload_stream=%p)\n",
                        num_tokens,
                        gpu_stream_,
                        dynamic_params_active_ ? 1 : 0,
                        dynamic_token_count_,
                        preload_stream_);
                return false;
            }
            if (!gpu_stream_)
            {
                fprintf(stderr, "[CUDAEmbeddingKernelT] Token ID upload requires an explicit non-null stream\n");
                return false;
            }
            if (!token_ids)
            {
                fprintf(stderr, "[CUDAEmbeddingKernelT] Host token IDs are null and no device token source is active\n");
                return false;
            }
            dynamic_params_active_ = false;
            dynamic_token_count_ = 0;
            err = cudaMemcpyAsync(workspace_token_ids, token_ids, token_bytes, cudaMemcpyHostToDevice,
                                  static_cast<cudaStream_t>(gpu_stream_));
            if (err != cudaSuccess)
            {
                fprintf(stderr, "[CUDAEmbeddingKernelT] Failed to copy token_ids to GPU: %s\n",
                        cudaGetErrorString(err));
                return false;
            }
        }

        // =====================================================================
        // Step 2: Get GPU pointer for output (coherence handled by DeviceGraphExecutor)
        // =====================================================================
        float *d_output = static_cast<float *>(output_fp32->gpu_data_ptr());
        if (!d_output)
        {
            fprintf(stderr, "[CUDAEmbeddingKernelT] Output GPU pointer is null\n");
            return false;
        }
        if (validate_gpu_ptrs &&
            !validateCudaPointerForDevice(d_output, dev, "OUTPUT", /*fail_on_query_error=*/true))
        {
            return false;
        }

        // =====================================================================
        // Step 3: Route by embedding table format
        // =====================================================================

        // --- Fast path: FP32 tensor already on GPU (no upload needed) ---
        auto *embed_fp32 = dynamic_cast<const FP32Tensor *>(embed_table);
        if (embed_fp32 && embed_fp32->isOnGPU())
        {
            float *d_embed = const_cast<float *>(static_cast<const float *>(embed_fp32->gpu_data_ptr()));
            if (validate_gpu_ptrs &&
                !validateCudaPointerForDevice(d_embed, dev, "EMBED_FP32", /*fail_on_query_error=*/true))
            {
                return false;
            }
            if (validate_gpu_ptrs && !checkCudaNoPriorError("EmbedFP32 launch"))
            {
                return false;
            }
            const int launch_vocab_size = explicit_vocab_range_ && local_vocab_size_ > 0
                                              ? local_vocab_size_
                                              : static_cast<int>(embed_fp32->rows());
            const int launch_vocab_offset = explicit_vocab_range_ ? vocab_offset_ : 0;
            CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::EMBEDDING_LOOKUP, gpu_stream_);
            err = launch_embedding_lookup(d_embed, d_token_ids, d_output,
                                          num_tokens, d_model,
                                          launch_vocab_size, launch_vocab_offset,
                                          static_cast<cudaStream_t>(gpu_stream_));
            if (err != cudaSuccess)
            {
                fprintf(stderr, "[CUDAEmbeddingKernelT] FP32 kernel launch failed: %s\n",
                        cudaGetErrorString(err));
                return false;
            }
            return true;
        }

        // --- Quantized path: repack to EmbedQ8 via IINT8Unpackable ---
        const auto *unpackable = dynamic_cast<const IINT8Unpackable *>(embed_table);
        if (unpackable)
        {
            // --- Preferred path: use model-owned PreparedWeightStore handle ---
            using namespace llaminar::v2::kernels;
            const DeviceId dev_id = DeviceId::cuda(dev);
            const PreparedEmbeddingHandle *prepared = nullptr;
            if (prepared_embedding_handle_ && prepared_embedding_handle_->device_id == dev_id)
                prepared = prepared_embedding_handle_;

            void *d_embed_q8 = nullptr;
            size_t blocks_per_row = 0;
            int vocab_offset = 0;
            int local_vocab_size = static_cast<int>(embed_table->rows());

            if (prepared && prepared->weights && prepared->weights->device_data)
            {
                // Fast path: GPU-resident prepared data from weight loading
                d_embed_q8 = prepared->weights->device_data;
                blocks_per_row = prepared->weights->blocks_per_row;
                vocab_offset = static_cast<int>(prepared->weights->vocab_offset);
                local_vocab_size = static_cast<int>(prepared->weights->vocab_size);
            }
            else
            {
                // Fallback: workspace-based lazy repack (for tests, CPU-only, etc.)
                if (!prepared)
                {
                    LOG_DEBUG("[CUDAEmbeddingKernelT] Prepared embedding lookup miss: "
                              << "tensor_ptr=" << static_cast<const void *>(embed_table)
                              << " device=" << dev_id.to_string()
                              << " — using workspace fallback");
                }
                d_embed_q8 = workspace_ ? workspace_->getBuffer(EmbeddingWorkspaceBuffers::EMBED_TABLE) : nullptr;
                if (!d_embed_q8)
                {
                    fprintf(stderr, "[CUDAEmbeddingKernelT] No prepared embedding weights and no workspace EMBED_TABLE buffer\n");
                    return false;
                }

                // Check if we need to repack + upload (first call or different tensor for THIS workspace)
                bool needs_upload = false;
                {
                    std::lock_guard<std::mutex> lock(s_embed_cache_mutex_);
                    auto it = s_workspace_embed_cache_.find(workspace_);
                    needs_upload = (it == s_workspace_embed_cache_.end()) || (it->second != embed_table);
                }
                if (needs_upload)
                {
                    // CPU-side repack: any quant format → EmbedQ8Block via IINT8Unpackable
                    auto repacked = repackEmbeddingToQ8(embed_table, d_model);

                    err = cudaMemcpyAsync(d_embed_q8, repacked.data.data(), repacked.byte_size,
                                          cudaMemcpyHostToDevice,
                                          static_cast<cudaStream_t>(gpu_stream_));
                    if (err != cudaSuccess)
                    {
                        fprintf(stderr, "[CUDAEmbeddingKernelT] Failed to upload EmbedQ8 data: %s\n",
                                cudaGetErrorString(err));
                        return false;
                    }

                    blocks_per_row = repacked.blocks_per_row;

                    {
                        std::lock_guard<std::mutex> lock(s_embed_cache_mutex_);
                        s_workspace_embed_cache_[workspace_] = embed_table;
                    }
                    LOG_DEBUG("[CUDAEmbeddingKernelT] Uploaded EmbedQ8 embedding (workspace fallback): "
                             << tensorTypeName(embed_table->native_type()) << " "
                             << repacked.vocab_size << "x" << d_model
                             << " → " << (repacked.byte_size / (1024 * 1024)) << " MB"
                             << " (" << repacked.blocks_per_row << " blocks/row)");
                }
                else
                {
                    blocks_per_row = (static_cast<size_t>(d_model) + 31) / 32;
                }
            }

            if (validate_gpu_ptrs)
            {
                if (!validateCudaPointerForDevice(d_embed_q8, dev, "EMBED_TABLE", /*fail_on_query_error=*/true))
                {
                    return false;
                }
                if (!checkCudaNoPriorError("EmbedQ8 launch"))
                {
                    LOG_ERROR("[CUDAEmbeddingKernelT] EmbedQ8 launch context: dev=" << dev
                                                                                    << " stream=" << gpu_stream_
                                                                                    << " d_embed_q8=" << d_embed_q8
                                                                                    << " d_token_ids=" << static_cast<void *>(d_token_ids)
                                                                                    << " d_output=" << static_cast<void *>(d_output)
                                                                                    << " num_tokens=" << num_tokens
                                                                                    << " d_model=" << d_model
                                                                                    << " blocks_per_row=" << blocks_per_row
                                                                                    << " local_vocab_size=" << local_vocab_size
                                                                                    << " vocab_offset=" << vocab_offset);
                    return false;
                }
            }
            // Validation readbacks require D2H plus stream synchronization, both
            // illegal inside CUDA graph capture. The launch itself still consumes
            // the device token IDs and remains graph-capturable.
            if (use_device_token_ids &&
                debugEnv().validation.validate_buffers &&
                num_tokens > 0 &&
                !isGraphCaptureActive())
            {
                std::vector<int> sampled_tokens(static_cast<size_t>(num_tokens), -1);
                cudaError_t token_copy_err = cudaMemcpyAsync(
                    sampled_tokens.data(),
                    d_token_ids,
                    static_cast<size_t>(num_tokens) * sizeof(int),
                    cudaMemcpyDeviceToHost,
                    static_cast<cudaStream_t>(gpu_stream_));
                if (token_copy_err != cudaSuccess)
                {
                    LOG_ERROR("[CUDAEmbeddingKernelT] Failed to read device token IDs for validation: "
                              << cudaGetErrorString(token_copy_err));
                    return false;
                }
                token_copy_err = cudaStreamSynchronize(static_cast<cudaStream_t>(gpu_stream_));
                if (token_copy_err != cudaSuccess)
                {
                    LOG_ERROR("[CUDAEmbeddingKernelT] Failed to synchronize device token validation: "
                              << cudaGetErrorString(token_copy_err));
                    return false;
                }
                const bool zero_token_rows_allowed =
                    allow_out_of_range_token_ids_ ||
                    (mpi_ctx && mpi_ctx->world_size() > 1);
                for (int i = 0; i < num_tokens; ++i)
                {
                    const int token_id = sampled_tokens[static_cast<size_t>(i)];
                    const bool in_local_range =
                        token_id >= vocab_offset &&
                        token_id < vocab_offset + local_vocab_size;
                    if (!in_local_range && !zero_token_rows_allowed)
                    {
                        LOG_ERROR("[CUDAEmbeddingKernelT] Device-token embedding would zero single-device token="
                                  << token_id << " local_vocab_size=" << local_vocab_size
                                  << " vocab_offset=" << vocab_offset
                                  << " num_tokens=" << num_tokens);
                        return false;
                    }
                    LOG_DEBUG("[CUDAEmbeddingKernelT] Device-token embedding validation token="
                              << token_id << " local_vocab_size=" << local_vocab_size
                              << " vocab_offset=" << vocab_offset
                              << " in_local_range=" << (in_local_range ? 1 : 0));
                }
            }

            // Launch EmbedQ8 kernel
            CUDA_KERNEL_PROFILE_SCOPE_STREAM(CUDAKernelType::EMBEDDING_LOOKUP, gpu_stream_);
            err = launch_embedding_lookup_q8(d_embed_q8, d_token_ids, d_output,
                                             num_tokens, d_model,
                                             static_cast<int>(blocks_per_row),
                                             local_vocab_size, vocab_offset,
                                             static_cast<cudaStream_t>(gpu_stream_));
            if (err != cudaSuccess)
            {
                fprintf(stderr, "[CUDAEmbeddingKernelT] EmbedQ8 kernel failed: %s\n",
                        cudaGetErrorString(err));
                if (validate_gpu_ptrs)
                {
                    LOG_ERROR("[CUDAEmbeddingKernelT] EmbedQ8 failure context: dev=" << dev
                                                                                     << " stream=" << gpu_stream_
                                                                                     << " d_embed_q8=" << d_embed_q8
                                                                                     << " d_token_ids=" << static_cast<void *>(d_token_ids)
                                                                                     << " d_output=" << static_cast<void *>(d_output)
                                                                                     << " num_tokens=" << num_tokens
                                                                                     << " d_model=" << d_model
                                                                                     << " blocks_per_row=" << blocks_per_row
                                                                                     << " local_vocab_size=" << local_vocab_size
                                                                                     << " vocab_offset=" << vocab_offset);
                }
                return false;
            }
            return true;
        }

        // No FP32 fallback — embedding table must be either FP32-on-GPU or IINT8Unpackable
        fprintf(stderr, "[CUDAEmbeddingKernelT] Embedding table type %s is not FP32-on-GPU "
                        "and does not implement IINT8Unpackable\n",
                tensorTypeName(embed_table->native_type()));
        return false;
    }

    // =============================================================================
    // IWorkspaceConsumer Interface Implementation
    // =============================================================================

    WorkspaceRequirements CUDAEmbeddingKernelT::getWorkspaceRequirements(
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
            256, // Alignment for CUDA
            true // Required
        });

        // Buffer 2: Embedding table temp [vocab_size × blocks_per_row × sizeof(EmbedQ8Block)]
        // Only needed when PreparedEmbeddingWeights are NOT available (test/fallback path).
        // When weights are prepared during loading, the prepared data lives in its own
        // GPU allocation and this workspace buffer is unused.
        if (!prepared_embedding_handle_)
        {
            constexpr size_t DEFAULT_VOCAB_SIZE = 151936;
            size_t d_model_size = (k > 0) ? static_cast<size_t>(k) : 896;
            size_t blocks_per_row = (d_model_size + 31) / 32;
            size_t embed_table_bytes = DEFAULT_VOCAB_SIZE * blocks_per_row * sizeof(EmbedQ8Block);
            reqs.buffers.push_back({EmbeddingWorkspaceBuffers::EMBED_TABLE,
                                    embed_table_bytes,
                                    256,
                                    true});
        }

        return reqs;
    }

    void CUDAEmbeddingKernelT::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        workspace_ = workspace;
    }

    bool CUDAEmbeddingKernelT::hasWorkspace() const
    {
        return workspace_ != nullptr && workspace_->isAllocated();
    }

    DeviceWorkspaceManager *CUDAEmbeddingKernelT::getWorkspace() const
    {
        return workspace_;
    }

} // namespace llaminar2
