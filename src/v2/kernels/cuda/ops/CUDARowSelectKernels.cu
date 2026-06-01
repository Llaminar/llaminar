/**
 * @file CUDARowSelectKernels.cu
 * @brief CUDA implementation of graph-capturable hidden-state row selection.
 *
 * The row copy uses a fixed kernel signature and launch shape for a fixed
 * bucket. The selected row is read from a device scalar that is updated by a
 * captured host-to-device copy, allowing one captured graph executable to replay
 * with different real prompt lengths inside the same bucket.
 */

#include "CUDARowSelectKernels.h"

#ifdef HAVE_CUDA

#include <algorithm>
#include <cuda_runtime.h>

namespace llaminar2::cuda
{
    namespace
    {
        /// @brief Copy one selected FP32 row using a grid-stride loop over columns.
        __global__ void rowSelectFP32Kernel(
            const float *__restrict__ input,
            float *__restrict__ output,
            const int *__restrict__ selected_row_ptr,
            int seq_len,
            int d_model)
        {
            const int raw_selected_row = selected_row_ptr ? *selected_row_ptr : 0;
            const int upper_bound_row = seq_len - 1;
            const int selected_row = raw_selected_row < 0
                                         ? 0
                                         : (raw_selected_row > upper_bound_row ? upper_bound_row : raw_selected_row);
            const size_t source_base = static_cast<size_t>(selected_row) * static_cast<size_t>(d_model);

            const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = blockDim.x * gridDim.x;
            for (int column = thread_index; column < d_model; column += stride)
            {
                output[column] = input[source_base + static_cast<size_t>(column)];
            }
        }

        /// @brief Concatenate two [rows, hidden_dim] matrices row-wise as [embedding, hidden].
        __global__ void mtpConcatFP32Kernel(
            const float *__restrict__ hidden,
            const float *__restrict__ embedding,
            float *__restrict__ output,
            int rows,
            int hidden_dim)
        {
            const int total = rows * hidden_dim;
            const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = blockDim.x * gridDim.x;
            for (int idx = thread_index; idx < total; idx += stride)
            {
                const int row = idx / hidden_dim;
                const int col = idx - row * hidden_dim;
                const size_t src_offset = static_cast<size_t>(idx);
                const size_t dst_offset =
                    (static_cast<size_t>(row) * static_cast<size_t>(hidden_dim) * 2) +
                    static_cast<size_t>(col);
                output[dst_offset] = embedding[src_offset];
                output[dst_offset + static_cast<size_t>(hidden_dim)] = hidden[src_offset];
            }
        }

        /// @brief Return true if a CUDA status is successful.
        bool ok(cudaError_t status)
        {
            return status == cudaSuccess;
        }
    }

    bool allocateRowSelectParam(
        int device_ordinal,
        int **host_selected_row,
        int **device_selected_row)
    {
        if (!host_selected_row || !device_selected_row)
            return false;

        *host_selected_row = nullptr;
        *device_selected_row = nullptr;

        // Pin the host scalar so the captured cudaMemcpyAsync is legal and
        // replays from a stable address without pageable-memory staging.
        if (!ok(cudaSetDevice(device_ordinal)))
            return false;
        if (!ok(cudaHostAlloc(reinterpret_cast<void **>(host_selected_row), sizeof(int), cudaHostAllocDefault)))
            return false;
        if (!ok(cudaMalloc(reinterpret_cast<void **>(device_selected_row), sizeof(int))))
        {
            cudaFreeHost(*host_selected_row);
            *host_selected_row = nullptr;
            return false;
        }
        return true;
    }

    void freeRowSelectParam(
        int device_ordinal,
        int *host_selected_row,
        int *device_selected_row)
    {
        cudaSetDevice(device_ordinal);
        if (device_selected_row)
            cudaFree(device_selected_row);
        if (host_selected_row)
            cudaFreeHost(host_selected_row);
    }

    bool uploadRowSelectParam(
        int *device_selected_row,
        const int *host_selected_row,
        void *stream)
    {
        if (!device_selected_row || !host_selected_row)
            return false;
        auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        return ok(cudaMemcpyAsync(
            device_selected_row,
            host_selected_row,
            sizeof(int),
            cudaMemcpyHostToDevice,
            cuda_stream));
    }

    bool launchRowSelectFP32(
        const float *input,
        float *output,
        const int *device_selected_row,
        int seq_len,
        int d_model,
        void *stream)
    {
        if (!input || !output || !device_selected_row || seq_len <= 0 || d_model <= 0)
            return false;

        constexpr int threads_per_block = 256;
        const int blocks = std::max(1, std::min(1024, (d_model + threads_per_block - 1) / threads_per_block));
        auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        rowSelectFP32Kernel<<<blocks, threads_per_block, 0, cuda_stream>>>(
            input,
            output,
            device_selected_row,
            seq_len,
            d_model);
        return ok(cudaGetLastError());
    }

    bool launchMTPConcatFP32(
        const float *hidden,
        const float *embedding,
        float *output,
        int rows,
        int hidden_dim,
        void *stream)
    {
        if (!hidden || !embedding || !output || rows <= 0 || hidden_dim <= 0)
            return false;

        constexpr int threads_per_block = 256;
        const int total = rows * hidden_dim;
        const int blocks = std::max(1, std::min(1024, (total + threads_per_block - 1) / threads_per_block));
        auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
        mtpConcatFP32Kernel<<<blocks, threads_per_block, 0, cuda_stream>>>(
            hidden,
            embedding,
            output,
            rows,
            hidden_dim);
        return ok(cudaGetLastError());
    }

} // namespace llaminar2::cuda

#endif // HAVE_CUDA
